"""Diamond-Si 4x4x4 valence-plus-conduction DLWF calculation.

Four occupied valence bands are frozen.  Four additional conduction-like
Wannier functions are disentangled from the next eight Bloch bands, giving an
eight-function manifold initialized by sp3 projections on the two Si atoms.
"""

from pathlib import Path

import numpy as np
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


def run_scf_once_and_share(kmf):
    """Run PySCF on rank 0 and replicate the orbitals needed by W90."""

    scf_data = None
    if MPI_RANK == 0:
        try:
            kmf.kernel()
        except Exception as exc:
            scf_data = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            scf_data = {
                "error": None,
                "converged": bool(kmf.converged),
                "e_tot": float(kmf.e_tot),
                "mo_energy": np.asarray(kmf.mo_energy),
                "mo_coeff": np.asarray(kmf.mo_coeff),
                "mo_occ": np.asarray(kmf.mo_occ),
            }
    if MPI_COMM is not None:
        scf_data = MPI_COMM.bcast(scf_data, root=0)
    if scf_data["error"] is not None:
        raise RuntimeError(f"Rank-0 PySCF calculation failed: {scf_data['error']}")

    kmf.converged = scf_data["converged"]
    kmf.e_tot = scf_data["e_tot"]
    kmf.mo_energy = scf_data["mo_energy"]
    kmf.mo_coeff = scf_data["mo_coeff"]
    kmf.mo_occ = scf_data["mo_occ"]


LATTICE_CONSTANT_ANG = 5.43
MP_GRID = [4, 4, 4]
NUM_OCCUPIED = 4
NUM_CONDUCTION_WANN = 4
NUM_WANN = NUM_OCCUPIED + NUM_CONDUCTION_WANN
NUM_BLOCH_BANDS = 12
PLOT_WANNIER_FUNCTIONS = True
WF_PLOT_SUPERCELL = [3, 3, 3]
WF_PLOT_GRID = [20, 20, 20]

# Quantum ESPRESSO ibrav=2 primitive vectors, expressed in Angstrom.
lattice = 0.5 * LATTICE_CONSTANT_ANG * np.array(
    [
        [-1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [-1.0, 1.0, 0.0],
    ]
)
second_si = np.array([0.25, 0.25, 0.25]) @ lattice

cell = gto.Cell(
    atom=[("Si", (0.0, 0.0, 0.0)), ("Si", second_si)],
    a=lattice,
    unit="A",
    basis="gth-dzvp-molopt-sr",
    pseudo="gth-pbe",
    mesh=[31, 31, 31],
    precision=1.0e-9,
    verbose=4 if MPI_RANK == 0 else 0,
).build()

kpts = cell.make_kpts(MP_GRID)
kmf = dft.KRKS(cell, kpts).density_fit()
kmf.xc = "PBE"
kmf.conv_tol = 1.0e-10
kmf.max_cycle = 100

output = Path("si-wannier90-v4-valence-conduction")
if MPI_RANK == 0:
    output.mkdir(exist_ok=True)
if MPI_COMM is not None:
    MPI_COMM.Barrier()
kmf.chkfile = str(output / "si_444_vc_scf.chk") if MPI_RANK == 0 else None
run_scf_once_and_share(kmf)
if not kmf.converged:
    raise RuntimeError("The PySCF KRKS calculation did not converge")

num_bands = min(np.asarray(energy).size for energy in kmf.mo_energy)
if num_bands < NUM_BLOCH_BANDS:
    raise RuntimeError(
        f"Need at least {NUM_BLOCH_BANDS} Bloch bands, but PySCF returned {num_bands}"
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
    seedname=output / "si_valence_conduction_dlwf",
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
            output / "si_valence_conduction_dlwf.info",
        )
        if PLOT_WANNIER_FUNCTIONS:
            plot_files = w90.plot_wf(
                outfile=output / "si_valence_conduction_wf",
                wf_list=range(NUM_WANN),
                supercell=WF_PLOT_SUPERCELL,
                grid=WF_PLOT_GRID,
            )
            print("Wannier-function XSF files:", *plot_files, sep="\n  ")
            print(
                "Wannier-function max imaginary/real ratios:",
                w90.wannier_plot_imaginary_ratios,
            )
