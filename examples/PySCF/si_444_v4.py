"""Diamond-Si 4x4x4 PySCF input for Wannier90-v4 space-energy localization.

The cell and unshifted k mesh match a Quantum ESPRESSO ``ibrav=2`` cell with
``A=5.43`` Angstrom and atoms at crystal coordinates (0, 0, 0) and
(1/4, 1/4, 1/4).  PySCF uses a Gaussian/GTH representation, so its numerical
DFT results are not expected to be identical to an ONCV/plane-wave result.
"""

from pathlib import Path

import numpy as np
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
        except Exception as exc:  # propagate a root failure to every rank
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

# Quantum ESPRESSO ibrav=2 primitive vectors, expressed in Angstrom.
lattice = 0.5 * LATTICE_CONSTANT_ANG * np.array(
    [
        [-1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [-1.0, 1.0, 0.0],
    ]
)

# The second position is crystal (1/4, 1/4, 1/4) in the lattice above.
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

# make_kpts produces the unshifted 4x4x4 full mesh required by the adapter.
kpts = cell.make_kpts(MP_GRID)
kmf = dft.KRKS(cell, kpts).density_fit()
kmf.xc = "PBE"
kmf.conv_tol = 1.0e-10
kmf.max_cycle = 100

output = Path("si-wannier90-v4")
if MPI_RANK == 0:
    output.mkdir(exist_ok=True)
if MPI_COMM is not None:
    MPI_COMM.Barrier()
kmf.chkfile = str(output / "si_444_scf.chk") if MPI_RANK == 0 else None
run_scf_once_and_share(kmf)
if not kmf.converged:
    raise RuntimeError("The PySCF KRKS calculation did not converge")

# The GTH Gaussian basis contains virtual bands.  Exclude all of them so this
# run tests only localization of the four occupied Si valence bands, without
# also testing Wannier90 disentanglement.  Wannier90 band indices are 1-based.
num_bands = min(np.asarray(energy).size for energy in kmf.mo_energy)
exclude_bands = list(range(NUM_OCCUPIED + 1, num_bands + 1))

# Four bond-centred s trial functions, in fractional primitive-cell
# coordinates.  Their count must equal num_wann for the current PySCF builder.
bond_projections = [
    "f= 0.125, 0.125, 0.125:s",
    "f=-0.375, 0.125, 0.125:s",
    "f= 0.125,-0.375, 0.125:s",
    "f= 0.125, 0.125,-0.375:s",
]

with W90(
    kmf,
    cell,
    MP_GRID,
    num_wann=NUM_OCCUPIED,
    seedname=output / "si_valence_dlwf",
    projections=bond_projections,
    exclude_bands=exclude_bands,
    mpi_comm=MPI_COMM if MPI_SIZE > 1 else None,
    options={
        "iprint": 2,
        "num_iter": 200,
        "conv_window": 5,
        "conv_tol": 1.0e-8,
        "sp_en_mix": 0.5,
        "sp_en_scale": 0.3781820678533007,
        "sp_en_power": 2.0,
        "sp_en_trust_radius": 0.2,
        "write_info": True,
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
    w90.run()
    if MPI_RANK == 0:
        print("SCF energy (Hartree/cell):", kmf.e_tot)
        print("Converged:", w90.converged)
        print("Centers (Angstrom):\n", w90.wann_centres)
        print("Spatial spreads (Angstrom^2):\n", w90.wann_spreads)
        print("Per-orbital space-energy data:", output / "si_valence_dlwf.info")
