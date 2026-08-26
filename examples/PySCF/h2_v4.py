"""Minimal PySCF 2.9 + Wannier90 v4 space-energy localization example.

Derived from the original pyWannier90 PySCF examples by Hung Q. Pham.
SPDX-License-Identifier: BSD-3-Clause
"""

from pathlib import Path

import numpy as np
from pyscf.pbc import gto, scf

from pywannier90_v4 import W90


cell = gto.Cell(
    atom="H 0 0 0; H 0 0 1.4",
    a=np.eye(3) * 5.0,
    unit="A",
    basis="sto-3g",
    mesh=[13, 13, 13],
    verbose=4,
).build()

mp_grid = [2, 1, 1]
kmf = scf.KRHF(cell, cell.make_kpts(mp_grid)).density_fit()
kmf.conv_tol = 1.0e-9
kmf.kernel()

output = Path("h2-wannier90-v4")
output.mkdir(exist_ok=True)

with W90(
    kmf,
    cell,
    mp_grid,
    num_wann=2,
    seedname=output / "h2_dlwf",
    projections=["H:s"],
    options={
        "num_iter": 50,
        "conv_window": 3,
        "conv_tol": 1.0e-8,
        "sp_en_mix": 0.5,
        "sp_en_scale": 0.3781820678533007,
        "sp_en_power": 2.0,
    },
) as w90:
    w90.kernel()
    print("Converged:", w90.converged)
    print("Centers (Angstrom):\n", w90.wann_centres)
    print("Spreads (Angstrom^2):\n", w90.wann_spreads)
