"""Diamond-Si 4x4x4 valence-plus-conduction DLWF calculation.

The example has separate SCF and localization stages so that each calculation
uses an appropriate parallel model:

* ``--stage scf`` runs one threaded PySCF process and saves its orbitals.
* ``--stage localize`` loads those orbitals on MPI rank 0, broadcasts them,
  and runs the Wannier90-v4 localization collectively.
* ``--stage all`` (the default) runs both stages serially for interactive use.

Four occupied valence bands are frozen.  Four additional conduction-like
Wannier functions are disentangled from the next eight Bloch bands, giving an
eight-function manifold initialized by sp3 projections on the two Si atoms.
"""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path

import numpy as np
from pyscf import lib
from pyscf.data import nist
from pyscf.pbc import dft, gto

from pywannier90_v4 import W90


try:
    from mpi4py import MPI
except ImportError:
    MPI = None

MPI_COMM = None if MPI is None else MPI.COMM_WORLD
MPI_RANK = 0 if MPI_COMM is None else MPI_COMM.Get_rank()
MPI_SIZE = 1 if MPI_COMM is None else MPI_COMM.Get_size()

LATTICE_CONSTANT_ANG = 5.43
MP_GRID = [4, 4, 4]
NUM_OCCUPIED = 4
NUM_CONDUCTION_WANN = 4
NUM_WANN = NUM_OCCUPIED + NUM_CONDUCTION_WANN
NUM_BLOCH_BANDS = 12
PLOT_WANNIER_FUNCTIONS = True
WF_PLOT_SUPERCELL = [3, 3, 3]
WF_PLOT_GRID = [20, 20, 20]

OUTPUT = Path("si-wannier90-v4-valence-conduction")
SCF_CHECKPOINT = OUTPUT / "si_444_vc_scf.chk"
SCF_ORBITALS = OUTPUT / "si_444_vc_orbitals.npz"

# Quantum ESPRESSO ibrav=2 primitive vectors, expressed in Angstrom.
LATTICE = (
    0.5
    * LATTICE_CONSTANT_ANG
    * np.array(
        [
            [-1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [-1.0, 1.0, 0.0],
        ]
    )
)


def build_cell(verbose: int):
    """Construct the common primitive cell on every process."""

    second_si = np.array([0.25, 0.25, 0.25]) @ LATTICE
    return gto.Cell(
        atom=[("Si", (0.0, 0.0, 0.0)), ("Si", second_si)],
        a=LATTICE,
        unit="A",
        basis="gth-dzvp-molopt-sr",
        pseudo="gth-pbe",
        mesh=[31, 31, 31],
        precision=1.0e-9,
        verbose=verbose,
    ).build()


def build_mean_field(cell):
    """Construct the KRKS object without running its SCF kernel."""

    kpts = cell.make_kpts(MP_GRID)
    kmf = dft.KRKS(cell, kpts).density_fit()
    kmf.xc = "PBE"
    kmf.conv_tol = 1.0e-10
    kmf.max_cycle = 100
    return kmf


def _positive_environment_integer(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def save_orbitals(kmf, path: Path) -> None:
    """Atomically save the numerical SCF state needed by localization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez(
            temporary,
            converged=np.asarray(bool(kmf.converged)),
            e_tot=np.asarray(float(kmf.e_tot)),
            kpts=np.asarray(kmf.kpts),
            mo_energy=np.asarray(kmf.mo_energy),
            mo_coeff=np.asarray(kmf.mo_coeff),
            mo_occ=np.asarray(kmf.mo_occ),
            mp_grid=np.asarray(MP_GRID, dtype=np.int32),
            lattice_angstrom=np.asarray(LATTICE),
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_scf_stage(orbital_path: Path) -> None:
    """Run a single PySCF process with shared-memory threading."""

    if MPI_SIZE != 1:
        raise RuntimeError(
            "The SCF stage must have one process. Run it without mpiexec, then "
            "launch the localization stage with MPI."
        )

    scf_threads = _positive_environment_integer("PYSCF_SCF_THREADS", 1)
    lib.num_threads(scf_threads)
    print(
        f"PySCF SCF stage: host={socket.gethostname()}, "
        f"processes=1, OpenMP threads={lib.num_threads()}"
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cell = build_cell(verbose=4)
    kmf = build_mean_field(cell)
    kmf.chkfile = str(SCF_CHECKPOINT)
    kmf.kernel()
    if not kmf.converged:
        raise RuntimeError("The PySCF KRKS calculation did not converge")

    save_orbitals(kmf, orbital_path)
    print("SCF energy (Hartree/cell):", kmf.e_tot)
    print("PySCF checkpoint:", SCF_CHECKPOINT.resolve())
    print("Localization orbital data:", orbital_path.resolve())


def load_and_share_orbitals(kmf, path: Path) -> None:
    """Load on rank 0 and broadcast the SCF arrays to all MPI ranks."""

    payload = None
    if MPI_RANK == 0:
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "converged",
                    "e_tot",
                    "kpts",
                    "mo_energy",
                    "mo_coeff",
                    "mo_occ",
                    "mp_grid",
                    "lattice_angstrom",
                }
                missing = sorted(required.difference(archive.files))
                if missing:
                    raise RuntimeError(
                        "orbital file is missing entries: " + ", ".join(missing)
                    )
                if not np.array_equal(archive["mp_grid"], MP_GRID):
                    raise RuntimeError("orbital file uses a different MP grid")
                if not np.allclose(archive["lattice_angstrom"], LATTICE):
                    raise RuntimeError("orbital file uses a different lattice")
                if not np.allclose(archive["kpts"], kmf.kpts):
                    raise RuntimeError("orbital file k-points do not match this input")
                payload = {
                    "error": None,
                    "converged": bool(archive["converged"].item()),
                    "e_tot": float(archive["e_tot"].item()),
                    "mo_energy": np.array(archive["mo_energy"]),
                    "mo_coeff": np.array(archive["mo_coeff"]),
                    "mo_occ": np.array(archive["mo_occ"]),
                }
        except Exception as exc:
            payload = {"error": f"{type(exc).__name__}: {exc}"}

    if MPI_COMM is not None:
        payload = MPI_COMM.bcast(payload, root=0)
    if payload["error"] is not None:
        raise RuntimeError(f"Unable to load PySCF orbital data: {payload['error']}")

    kmf.converged = payload["converged"]
    kmf.e_tot = payload["e_tot"]
    kmf.mo_energy = payload["mo_energy"]
    kmf.mo_coeff = payload["mo_coeff"]
    kmf.mo_occ = payload["mo_occ"]
    kmf.chkfile = None


def run_localization_stage(orbital_path: Path) -> None:
    """Run the Wannier90 stage using the active MPI communicator."""

    expected_ranks = _positive_environment_integer(
        "PYWANNIER90_EXPECT_MPI_RANKS", MPI_SIZE
    )
    if MPI_SIZE != expected_ranks:
        raise RuntimeError(
            f"Expected {expected_ranks} MPI ranks but MPI.COMM_WORLD has {MPI_SIZE}. "
            "Check the mpiexec command and mpi4py installation."
        )

    # Every MPI worker must remain single-threaded to avoid oversubscription.
    lib.num_threads(1)
    if MPI_RANK == 0:
        print(
            f"Wannier90 stage: host={socket.gethostname()}, "
            f"MPI ranks={MPI_SIZE}, threads per rank={lib.num_threads()}"
        )
        OUTPUT.mkdir(parents=True, exist_ok=True)
    if MPI_COMM is not None:
        MPI_COMM.Barrier()

    cell = build_cell(verbose=4 if MPI_RANK == 0 else 0)
    kmf = build_mean_field(cell)
    load_and_share_orbitals(kmf, orbital_path)
    if not kmf.converged:
        raise RuntimeError("The saved PySCF KRKS calculation is not converged")

    num_bands = min(np.asarray(energy).size for energy in kmf.mo_energy)
    if num_bands < NUM_BLOCH_BANDS:
        raise RuntimeError(
            f"Need at least {NUM_BLOCH_BANDS} Bloch bands, but PySCF returned "
            f"{num_bands}"
        )

    # Retain bands 1--12.  The valence bands 1--4 are frozen, while Wannier90
    # selects a smooth four-dimensional conduction subspace from bands 5--12.
    exclude_bands = list(range(NUM_BLOCH_BANDS + 1, num_bands + 1))
    energies_ev = np.asarray(kmf.mo_energy)[:, :NUM_BLOCH_BANDS] * nist.HARTREE2EV
    vbm_ev = float(np.max(energies_ev[:, NUM_OCCUPIED - 1]))
    cbm_ev = float(np.min(energies_ev[:, NUM_OCCUPIED]))
    frozen_window_max_ev = 0.5 * (vbm_ev + cbm_ev)
    outer_window_min_ev = float(np.min(energies_ev)) - 1.0
    outer_window_max_ev = float(np.max(energies_ev)) + 1.0

    # Si:sp3 expands to four projections on each of the two Si atoms, exactly
    # matching the requested eight Wannier functions.
    projections = ["Si:sp3"]

    with W90(
        kmf,
        cell,
        MP_GRID,
        num_wann=NUM_WANN,
        seedname=OUTPUT / "si_valence_conduction_dlwf",
        projections=projections,
        exclude_bands=exclude_bands,
        mpi_comm=MPI_COMM if MPI_SIZE > 1 else None,
        options={
            "iprint": 2,
            "dis_num_iter": 200,
            "dis_mix_ratio": 1.0,
            "dis_win_min": outer_window_min_ev,
            "dis_win_max": outer_window_max_ev,
            "dis_froz_max": frozen_window_max_ev,
            "num_iter": 200,
            "conv_window": 5,
            "conv_tol": 1.0e-8,
            "sp_en_mix": 0.5,
            "sp_en_scale": 0.3781820678533007,
            "sp_en_power": 2.0,
            "sp_en_trust_radius": 0.2,
            "write_info": True,
            # num_occ refers to the retained band list, where bands 1--4 are
            # occupied and bands 5--12 are unoccupied.
            "num_occ": NUM_OCCUPIED,
        },
    ) as w90:
        w90.setup()
        local_counts = (
            [w90.local_kpoint_indices.size]
            if MPI_COMM is None
            else MPI_COMM.gather(w90.local_kpoint_indices.size, root=0)
        )
        if MPI_RANK == 0:
            print("Wannier90 k-points assigned per MPI rank:", local_counts)
            print(f"VBM = {vbm_ev:.8f} eV; CBM = {cbm_ev:.8f} eV")
            print(f"Frozen-window maximum = {frozen_window_max_ev:.8f} eV")
            print(
                f"Outer window = [{outer_window_min_ev:.8f}, "
                f"{outer_window_max_ev:.8f}] eV"
            )
        w90.run()

        if MPI_RANK == 0:
            print("SCF energy (Hartree/cell):", kmf.e_tot)
            print("Converged:", w90.converged)
            print("Centers (Angstrom):\n", w90.wann_centres)
            print("Spatial spreads (Angstrom^2):\n", w90.wann_spreads)
            print(
                "Per-orbital space-energy data:",
                OUTPUT / "si_valence_conduction_dlwf.info",
            )
            if PLOT_WANNIER_FUNCTIONS:
                plot_files = w90.plot_wf(
                    outfile=OUTPUT / "si_valence_conduction_wf",
                    wf_list=range(NUM_WANN),
                    supercell=WF_PLOT_SUPERCELL,
                    grid=WF_PLOT_GRID,
                )
                print("Wannier-function XSF files:", *plot_files, sep="\n  ")
                print(
                    "Wannier-function max imaginary/real ratios:",
                    w90.wannier_plot_imaginary_ratios,
                )

        # Non-root ranks wait while rank 0 writes the optional XSF files.
        if MPI_COMM is not None:
            MPI_COMM.Barrier()


def parse_arguments() -> argparse.Namespace:
    default_stage = os.environ.get("PYWANNIER90_STAGE", "all").lower()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("scf", "localize", "all"),
        default=default_stage,
        help=(
            "workflow stage; defaults to PYWANNIER90_STAGE or 'all'. The "
            "'all' mode is serial and intended for interactive testing"
        ),
    )
    parser.add_argument(
        "--orbital-file",
        type=Path,
        default=SCF_ORBITALS,
        help="shared NPZ file used to transfer converged PySCF orbitals",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    orbital_path = args.orbital_file.expanduser().resolve()
    if args.stage == "scf":
        run_scf_stage(orbital_path)
    elif args.stage == "localize":
        run_localization_stage(orbital_path)
    else:
        if MPI_SIZE != 1:
            raise RuntimeError(
                "--stage all is serial. Use separate --stage scf and "
                "--stage localize invocations for MPI localization."
            )
        run_scf_stage(orbital_path)
        run_localization_stage(orbital_path)


if __name__ == "__main__":
    main()
