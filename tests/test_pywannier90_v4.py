"""Tests for the PySCF/Wannier90-v4 interface.

The native test is enabled by setting PYWANNIER90_TEST_LIB to a Wannier90 v4
shared library built with WANNIER90_WITH_C=ON.
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from pywannier90_v4 import (
    SymmetryData,
    W90,
    Wannier90V4Library,
    _W90Data,
    _default_kpoint_distribution,
    _parse_legacy_keywords,
    build_symmetry_data,
    projection_function,
    symmetrize_projection_matrix,
    validate_symmetry_data,
    write_dmn,
)


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


class MPISupportTests(unittest.TestCase):
    def test_default_distribution_is_contiguous_and_balanced(self):
        np.testing.assert_array_equal(
            _default_kpoint_distribution(7, 3),
            np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int32),
        )

    def test_fortran_communicator_handle_is_passed_to_wannier90(self):
        class FakeComm:
            @staticmethod
            def py2f():
                return 37

            @staticmethod
            def Get_rank():
                return 1

            @staticmethod
            def Get_size():
                return 3

        class FakeLibrary:
            communicator = None

            def w90_set_comm_f(self, data, handle):
                del data
                self.communicator = ctypes.cast(
                    handle, ctypes.POINTER(ctypes.c_int)
                ).contents.value

        wrapper = object.__new__(Wannier90V4Library)
        wrapper.is_mpi = True
        wrapper.lib = FakeLibrary()
        wrapper.data = _W90Data(ctypes.c_void_p(1))
        wrapper.set_communicator(FakeComm())

        self.assertEqual(wrapper.lib.communicator, 37)
        self.assertEqual(wrapper.mpi_rank, 1)
        self.assertEqual(wrapper.mpi_size, 3)

    def test_global_m_matrix_is_packed_for_the_local_rank(self):
        class FakeLibrary:
            attached_m = None

            def attach_matrices(self, eigenvalues, m_matrix, u_matrix, u_opt):
                del eigenvalues, u_matrix, u_opt
                self.attached_m = m_matrix

            @staticmethod
            def disentangle():
                pass

            @staticmethod
            def project_overlap():
                pass

            @staticmethod
            def wannierise():
                return 0

            @staticmethod
            def get_centres_and_spreads(num_wann):
                return np.zeros((3, num_wann)), np.zeros(num_wann)

        adapter = object.__new__(W90)
        adapter.library = FakeLibrary()
        adapter.num_bands = 1
        adapter.num_wann = 1
        adapter.nntot = 1
        adapter.num_kpts = 4
        adapter.local_kpoint_indices = np.array([1, 3], dtype=np.int32)
        adapter.mo_energy = np.zeros((4, 1))
        adapter.band_included_list = [0]
        global_m = np.arange(4, dtype=np.complex128).reshape(1, 1, 1, 4)

        adapter.run(
            m_matrix=global_m,
            a_matrix=np.ones((1, 1, 4), dtype=np.complex128),
            eigenvalues=np.zeros((1, 4)),
        )

        np.testing.assert_array_equal(
            adapter.library.attached_m.reshape(-1), np.array([1.0, 3.0])
        )
        self.assertTrue(adapter.M_matrix_local.flags.f_contiguous)


class SiteSymmetryTests(unittest.TestCase):
    @staticmethod
    def _identity_data(num_kpts=1):
        return SymmetryData(
            ik2ir=np.arange(num_kpts, dtype=np.int32),
            ir2ik=np.arange(num_kpts, dtype=np.int32),
            kptsym=np.arange(num_kpts, dtype=np.int32)[None, :],
            d_matrix_wann=np.ones(
                (1, 1, 1, num_kpts), dtype=np.complex128
            ),
            d_matrix_band=np.ones(
                (1, 1, 1, num_kpts), dtype=np.complex128
            ),
        )

    def test_dmn_writer_uses_one_based_maps_and_native_header(self):
        data = self._identity_data(2)
        with tempfile.TemporaryDirectory() as directory:
            path = write_dmn(Path(directory) / "test.dmn", data, atomic=True)
            lines = path.read_text().splitlines()
        self.assertEqual([int(value) for value in lines[1].split()], [1, 1, 2, 2])
        integer_lines = [line.split() for line in lines[3:11] if line.strip()]
        self.assertEqual(integer_lines[:4], [["1", "2"], ["1", "2"], ["1"], ["2"]])
        self.assertFalse(any(path.parent.glob(".test.dmn.tmp-*")))

    def test_nonunitary_representation_is_rejected(self):
        data = self._identity_data()
        data.d_matrix_band[0, 0, 0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "not unitary"):
            validate_symmetry_data(data)

    def test_projection_seed_is_transported_over_each_kpoint_star(self):
        phase = np.exp(0.37j)
        data = SymmetryData(
            ik2ir=np.array([0, 0], dtype=np.int32),
            ir2ik=np.array([0], dtype=np.int32),
            kptsym=np.array([[0], [1]], dtype=np.int32),
            d_matrix_wann=np.ones((1, 1, 2, 1), dtype=np.complex128),
            d_matrix_band=np.array([[[[1.0], [phase]]]]),
        )
        raw = np.array([[[2.0, -3.0j]]], dtype=np.complex128)
        prepared = symmetrize_projection_matrix(raw, data)
        self.assertAlmostEqual(abs(prepared[0, 0, 0]), 1.0)
        np.testing.assert_allclose(prepared[0, 0, 1], phase * prepared[0, 0, 0])

    def test_rank_zero_writes_then_barriers(self):
        class FakeComm:
            barriers = 0

            @staticmethod
            def bcast(value, root):
                self.assertEqual(root, 0)
                return value

            def Barrier(self):
                self.barriers += 1

        adapter = object.__new__(W90)
        adapter.options = {"site_symmetry": True, "sp_en_mix": 0.5}
        adapter.num_bands = 1
        adapter.num_wann = 1
        adapter.num_kpts = 1
        adapter.is_mpi = True
        adapter.mpi_rank = 0
        adapter.mpi_comm = FakeComm()
        adapter.requested_symmetry_data = self._identity_data()
        adapter.symmetry_data = None
        adapter.symmetry_projection_tolerance = 1.0e-6
        adapter.symmetry_band_tolerance = 1.0e-6
        with tempfile.TemporaryDirectory() as directory:
            adapter.seedname = str(Path(directory) / "rank_zero")
            adapter._prepare_symmetry_file(np.ones((1, 1, 1)))
            self.assertTrue(Path(adapter.seedname + ".dmn").is_file())
        self.assertEqual(adapter.mpi_comm.barriers, 1)
        self.assertIs(adapter.symmetry_data, adapter.requested_symmetry_data)

    def test_dlwf_site_symmetry_is_accepted_by_python_validation(self):
        adapter = object.__new__(W90)
        adapter.options = {"sp_en_mix": 0.5, "site_symmetry": True}
        adapter.gamma_only = False
        adapter._validate_dlwf_options()

    def test_dlwf_site_symmetry_rejects_frozen_window(self):
        adapter = object.__new__(W90)
        adapter.options = {
            "sp_en_mix": 0.5,
            "site_symmetry": True,
            "dis_froz_max": 3.0,
        }
        adapter.gamma_only = False
        with self.assertRaisesRegex(ValueError, "frozen"):
            adapter._validate_dlwf_options()

    def test_pyscf_builder_produces_closed_unitary_data(self):
        try:
            from pyscf.pbc import gto
        except ImportError as exc:  # pragma: no cover - dependency guard
            self.skipTest(str(exc))

        cell = gto.Cell(
            atom="H 0 0 0",
            a=np.eye(3) * 5.0,
            unit="A",
            basis="sto-3g",
            mesh=[7, 7, 7],
            spin=1,
            verbose=0,
        ).build()
        kpts = cell.make_kpts([2, 2, 2])
        overlaps = np.asarray(cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts))
        coefficients = np.empty((8, 1, 1), dtype=np.complex128)
        coefficients[:, 0, 0] = 1.0 / np.sqrt(overlaps[:, 0, 0].real)
        projections = np.ones((1, 1, 8), dtype=np.complex128)
        projection_data = {
            "site": np.zeros((3, 1)),
            "l": np.array([0]),
            "m": np.array([1]),
            "radial": np.array([1]),
            "zona": np.array([1.0]),
            "x_axis": np.array([[1.0], [0.0], [0.0]]),
            "z_axis": np.array([[0.0], [0.0], [1.0]]),
        }
        data = build_symmetry_data(
            cell,
            kpts,
            coefficients,
            projections,
            projection_data=projection_data,
        )
        validate_symmetry_data(data)
        self.assertEqual(data.ik2ir.size, 8)
        self.assertEqual(data.d_matrix_band.shape[:2], (1, 1))
        self.assertGreater(data.kptsym.shape[0], 1)


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


class WannierPlotTests(unittest.TestCase):
    class _Cell:
        @staticmethod
        def lattice_vectors():
            return np.eye(3)

    @staticmethod
    def _adapter():
        adapter = object.__new__(W90)
        adapter.cell = WannierPlotTests._Cell()
        adapter.real_lattice = np.eye(3)
        adapter.atom_symbols = ["H"]
        adapter.atoms_cart = np.zeros((1, 3))
        adapter.num_wann = 1
        adapter.num_kpts = 1
        adapter.kpts = np.zeros((1, 3))
        adapter.band_included_list = [0]
        adapter.mo_coeff = np.array([[[2.0 + 0.0j]]])
        adapter.u_matrix_opt = np.array([[[1.0 + 0.0j]]])
        adapter.u_matrix = np.array([[[1.0 + 0.0j]]])
        adapter.is_mpi = False
        adapter.mpi_rank = 0
        return adapter

    @staticmethod
    def _fake_pyscf_modules():
        numint = types.ModuleType("pyscf.pbc.dft.numint")
        numint.eval_ao = lambda cell, coordinates, kpt: np.ones(
            (len(coordinates), 1), dtype=np.complex128
        )
        dft = types.ModuleType("pyscf.pbc.dft")
        dft.__path__ = []
        dft.numint = numint
        pbc = types.ModuleType("pyscf.pbc")
        pbc.__path__ = []
        pbc.dft = dft
        pyscf = types.ModuleType("pyscf")
        pyscf.__path__ = []
        pyscf.pbc = pbc
        return {
            "pyscf": pyscf,
            "pyscf.pbc": pbc,
            "pyscf.pbc.dft": dft,
            "pyscf.pbc.dft.numint": numint,
        }

    def test_get_wannier_applies_final_transform_on_xsf_ordered_grid(self):
        adapter = self._adapter()
        with mock.patch.dict(sys.modules, self._fake_pyscf_modules()):
            values = adapter.get_wannier(grid=(2, 2, 2), chunk_size=3)
        self.assertEqual(values.shape, (8, 1))
        np.testing.assert_allclose(values, 2.0)
        np.testing.assert_allclose(adapter.wannier_plot_imaginary_ratios, 0.0)

    def test_plot_wf_writes_xsf_on_root_only(self):
        adapter = self._adapter()
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "WF"
            with mock.patch.dict(sys.modules, self._fake_pyscf_modules()):
                written = adapter.plot_wf(prefix, grid=(2, 2, 2))
            self.assertEqual(written, [Path(f"{prefix}-0.xsf")])
            contents = written[0].read_text()
            self.assertIn("BEGIN_DATAGRID_3D_UNKNOWN", contents)
            self.assertIn("2 2 2", contents)

            adapter.is_mpi = True
            adapter.mpi_rank = 1
            self.assertEqual(adapter.plot_wf(prefix, grid=(2, 2, 2)), [])


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
                self.assertEqual(
                    w90.M_matrix.shape,
                    (2, 2, 10, w90.local_kpoint_indices.size),
                )
                if w90.is_mpi:
                    distributed = w90.mpi_comm.allgather(
                        w90.local_kpoint_indices.tolist()
                    )
                    self.assertEqual(
                        sorted(index for indices in distributed for index in indices),
                        list(range(w90.num_kpts)),
                    )
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
