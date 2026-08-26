"""Tests for the PySCF/Wannier90-v4 interface.

The native test is enabled by setting PYWANNIER90_TEST_LIB to a Wannier90 v4
shared library built with WANNIER90_WITH_C=ON.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pywannier90_v4 import W90, _parse_legacy_keywords, projection_function


class ProjectionTests(unittest.TestCase):
    def test_s_projection_is_finite_at_its_center(self):
        values = projection_function(
            np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.1]]),
            site=[0.0, 0.0, 0.0],
            l=0,
            mr=1,
            radial=1,
            zona=1.0,
        )
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(values[0], values[1])

    def test_projection_axes_must_be_orthogonal(self):
        with self.assertRaisesRegex(ValueError, "orthogonal"):
            projection_function(
                np.ones((1, 3)),
                site=[0.0, 0.0, 0.0],
                l=0,
                mr=1,
                radial=1,
                zona=1.0,
                x_axis=[1.0, 0.0, 0.0],
                z_axis=[1.0, 0.0, 0.0],
            )


class LegacyKeywordTests(unittest.TestCase):
    def test_common_original_syntax(self):
        parsed = _parse_legacy_keywords(
            """
            exclude_bands : 3, 4
            begin projections
            H:s
            end projections
            use_bloch_phases = false
            sp_en_mix = 5.0d-1
            """
        )
        self.assertEqual(parsed["exclude_bands"], [3, 4])
        self.assertEqual(parsed["projections"], ["H:s"])
        self.assertFalse(parsed["use_bloch_phases"])
        self.assertAlmostEqual(parsed["sp_en_mix"], 0.5)


class PySCFObjectCompatibilityTests(unittest.TestCase):
    class _Cell:
        natm = 1

        @staticmethod
        def lattice_vectors():
            return np.eye(3) * 8.0

        @staticmethod
        def get_scaled_kpts(kpts):
            return np.asarray(kpts)

        @staticmethod
        def atom_symbol(index):
            return "H"

        @staticmethod
        def atom_coords():
            return np.zeros((1, 3))

    class _UnrestrictedMeanField:
        kpts = np.zeros((2, 3))
        mo_energy = (
            np.array([[-1.0, 0.1], [-0.9, 0.2]]),
            np.array([[-0.8, 0.3], [-0.7, 0.4]]),
        )
        mo_coeff = (
            np.repeat(np.eye(2)[None, :, :], 2, axis=0),
            np.repeat((2.0 * np.eye(2))[None, :, :], 2, axis=0),
        )

    def test_unrestricted_spin_selection_uses_canonical_pyscf_attributes(self):
        adapter = W90(
            self._UnrestrictedMeanField(),
            self._Cell(),
            [2, 1, 1],
            num_wann=1,
            spin="down",
            use_bloch_phases=True,
        )
        np.testing.assert_allclose(adapter.mo_energy[:, 0], [-0.8, -0.7])
        np.testing.assert_allclose(adapter.mo_coeff[:, 0, 0], [2.0, 2.0])

    def test_dlwf_gamma_only_is_rejected_before_native_call(self):
        with self.assertRaisesRegex(ValueError, "gamma_only"):
            W90(
                self._UnrestrictedMeanField(),
                self._Cell(),
                [2, 1, 1],
                num_wann=1,
                gamma=True,
                spin="up",
                use_bloch_phases=True,
                options={"sp_en_mix": 0.5},
            )


@unittest.skipUnless(
    os.environ.get("PYWANNIER90_TEST_LIB"),
    "set PYWANNIER90_TEST_LIB to run the native Wannier90-v4 test",
)
class NativePySCF29Tests(unittest.TestCase):
    def test_space_energy_localization(self):
        try:
            from pyscf.pbc import gto, scf
        except ImportError as exc:  # pragma: no cover - conditional environment
            self.skipTest(str(exc))

        cell = gto.Cell(
            atom="H 0 0 0; H 0 0 1.4",
            a=np.eye(3) * 5.0,
            unit="A",
            basis="sto-3g",
            mesh=[13, 13, 13],
            verbose=0,
        ).build()
        mp_grid = [2, 1, 1]
        kmf = scf.KRHF(cell, cell.make_kpts(mp_grid)).density_fit()
        kmf.conv_tol = 1.0e-9
        kmf.kernel()
        self.assertTrue(kmf.converged)

        with tempfile.TemporaryDirectory() as directory:
            with W90(
                kmf,
                cell,
                mp_grid,
                num_wann=2,
                library_path=os.environ["PYWANNIER90_TEST_LIB"],
                seedname=Path(directory) / "h2_dlwf",
                projections=["H:s"],
                options={
                    "iprint": 0,
                    "num_iter": 10,
                    "conv_window": 3,
                    "conv_tol": 1.0e-8,
                    "sp_en_mix": 0.5,
                    "sp_en_scale": 0.3781820678533007,
                    "sp_en_power": 2.0,
                    "sp_en_trust_radius": 0.2,
                },
            ) as w90:
                w90.kernel()
                self.assertEqual(w90.M_matrix.shape, (2, 2, 10, 2))
                self.assertEqual(w90.A_matrix.shape, (2, 2, 2))
                self.assertEqual(w90.eigenvalues.shape, (2, 2))
                self.assertEqual(w90.wann_centres.shape, (2, 3))
                self.assertTrue(np.isfinite(w90.wann_centres).all())
                self.assertTrue(np.isfinite(w90.wann_spreads).all())
                hamiltonian = w90.get_hamiltonian_kpts()
                np.testing.assert_allclose(
                    hamiltonian,
                    hamiltonian.conj().transpose(0, 2, 1),
                    atol=1.0e-11,
                )


if __name__ == "__main__":
    unittest.main()
