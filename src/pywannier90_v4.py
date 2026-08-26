"""PySCF interface to the Wannier90 v4 library API.

This module is derived from pyWannier90, originally written by Hung Q. Pham:
https://github.com/hungpham2017/pyWannier90

Copyright (c) 2017 Hung Pham
Copyright (c) 2026 contributors to the Wannier90 v4 interface

SPDX-License-Identifier: BSD-3-Clause

The Wannier90 library is a separate work licensed under LGPL-2.1-or-later.
It is discovered at runtime and is not included with this Python module.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class Wannier90Error(RuntimeError):
    """Raised when the Wannier90 library reports an error."""


class _W90Data(ctypes.Structure):
    _fields_ = [("cptr", ctypes.c_void_p)]


def _array_pointer(array: np.ndarray) -> ctypes.c_void_p:
    return ctypes.c_void_p(array.ctypes.data)


def _candidate_library_paths(explicit_path: str | os.PathLike[str] | None) -> list[str]:
    serial_names = (
        "libwannier90.4.dylib",
        "libwannier90.dylib",
        "libwannier90.so.4",
        "libwannier90.so",
    )
    mpi_names = (
        "libwannier90_mpi.4.dylib",
        "libwannier90_mpi.dylib",
        "libwannier90_mpi.so.4",
        "libwannier90_mpi.so",
    )
    names = serial_names + mpi_names

    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if path.is_dir():
            matches = [str(path / name) for name in names if (path / name).is_file()]
            if matches:
                return matches
            raise FileNotFoundError(f"No Wannier90 v4 shared library found in {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Wannier90 shared library does not exist: {path}")
        return [str(path)]

    candidates: list[str] = []
    for variable in ("PYWANNIER90_LIB", "WANNIER90_MPI_LIB"):
        configured = os.environ.get(variable)
        if not configured:
            continue
        configured_path = Path(configured).expanduser()
        if configured_path.is_dir():
            candidates.extend(str(configured_path / name) for name in names)
        else:
            candidates.append(str(configured_path))

    for variable in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        for directory in os.environ.get(variable, "").split(os.pathsep):
            if directory:
                candidates.extend(str(Path(directory) / name) for name in names)

    for library_name in ("wannier90", "wannier90_mpi"):
        found = ctypes.util.find_library(library_name)
        if found:
            candidates.append(found)
    candidates.extend(names)

    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


class Wannier90V4Library:
    """Thin owner-aware wrapper around Wannier90's version-4 C API."""

    def __init__(
        self,
        library_path: str | os.PathLike[str] | None = None,
        mpi_comm: Any | None = None,
    ):
        self.path, self.lib = self._load(library_path)
        self._configure_signatures()
        self.is_mpi = hasattr(self.lib, "w90_set_comm_f")
        self.data = _W90Data()
        self.lib.w90_create(ctypes.byref(self.data))
        if not self.data.cptr:
            raise Wannier90Error("w90_create did not return a valid Wannier90 object")
        self._closed = False
        self._matrix_owners: tuple[np.ndarray, ...] = ()
        self.mpi_comm: Any | None = None
        self.mpi_rank = 0
        self.mpi_size = 1
        try:
            if self.is_mpi:
                self.set_communicator(mpi_comm)
            elif mpi_comm is not None:
                raise ValueError(
                    "mpi_comm was supplied, but the selected Wannier90 library is serial"
                )
        except Exception:
            self.close()
            raise

    @staticmethod
    def _load(
        library_path: str | os.PathLike[str] | None,
    ) -> tuple[str, ctypes.CDLL]:
        errors: list[str] = []
        for candidate in _candidate_library_paths(library_path):
            try:
                library = ctypes.CDLL(candidate)
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")
                continue
            if not hasattr(library, "w90_create"):
                errors.append(f"{candidate}: missing the Wannier90 v4 symbol w90_create")
                continue
            return candidate, library

        detail = "\n  ".join(errors) if errors else "no candidate paths were found"
        raise OSError(
            "Unable to load a Wannier90 v4 shared library built with "
            "WANNIER90_WITH_C=ON. Set PYWANNIER90_LIB or "
            "WANNIER90_MPI_LIB, or pass library_path.\n  "
            + detail
        )

    def _configure_signatures(self) -> None:
        data = _W90Data
        void_p = ctypes.c_void_p
        int_p = ctypes.POINTER(ctypes.c_int)
        double_p = ctypes.POINTER(ctypes.c_double)

        signatures: dict[str, list[Any]] = {
            "w90_create": [ctypes.POINTER(data)],
            "w90_delete": [ctypes.POINTER(data)],
            "w90_set_option_double": [data, ctypes.c_char_p, ctypes.c_double],
            "w90_set_option_double1d": [data, ctypes.c_char_p, void_p, ctypes.c_int],
            "w90_set_option_double2d": [data, ctypes.c_char_p, void_p, ctypes.c_int, ctypes.c_int],
            "w90_set_option_int": [data, ctypes.c_char_p, ctypes.c_int],
            "w90_set_option_int1d": [data, ctypes.c_char_p, void_p, ctypes.c_int],
            "w90_set_option_int2d": [data, ctypes.c_char_p, void_p, ctypes.c_int, ctypes.c_int],
            "w90_set_option_logical": [data, ctypes.c_char_p, ctypes.c_bool],
            "w90_set_option_text": [data, ctypes.c_char_p, ctypes.c_char_p],
            "w90_set_option_text2d": [
                data,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.c_int,
            ],
            "w90_input_setopt": [data, ctypes.c_char_p, int_p],
            "w90_input_reader": [data, int_p],
            "w90_get_nk": [data, int_p],
            "w90_get_nw": [data, int_p],
            "w90_get_nn": [data, int_p, int_p],
            "w90_get_nnkp": [data, void_p, int_p],
            "w90_get_gkpb": [data, void_p, int_p],
            "w90_get_nproj": [data, int_p],
            "w90_get_proj": [
                data,
                int_p,
                void_p,
                void_p,
                void_p,
                void_p,
                void_p,
                void_p,
                void_p,
                void_p,
                void_p,
                int_p,
            ],
            "w90_get_num_excl_bands": [data, int_p],
            "w90_get_excl_bands": [data, void_p, int_p],
            "w90_set_eigval": [data, void_p],
            "w90_set_m_local": [data, void_p],
            "w90_set_u_matrix": [data, void_p],
            "w90_set_u_opt": [data, void_p],
            "w90_disentangle": [data, int_p],
            "w90_project_overlap": [data, int_p],
            "w90_wannierise": [data, int_p],
            "w90_print_info": [data, int_p],
            "w90_get_centres": [data, double_p],
            "w90_get_spreads": [data, double_p],
        }
        missing = [name for name in signatures if not hasattr(self.lib, name)]
        if missing:
            raise OSError(
                f"{self.path} is missing required Wannier90 v4 C symbols: "
                + ", ".join(missing)
            )
        for name, argtypes in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = argtypes
            function.restype = None
        if hasattr(self.lib, "w90_set_comm_f"):
            self.lib.w90_set_comm_f.argtypes = [data, int_p]
            self.lib.w90_set_comm_f.restype = None

    def set_communicator(self, mpi_comm: Any | None = None) -> None:
        """Attach an mpi4py-compatible communicator to an MPI Wannier90 build."""

        if not self.is_mpi:
            raise Wannier90Error("The loaded Wannier90 library was built without MPI")
        if mpi_comm is None:
            try:
                from mpi4py import MPI
            except ImportError as exc:
                raise ImportError(
                    "An MPI-enabled Wannier90 library requires mpi4py built against "
                    "the same MPI implementation. Install pywannier90-v4-interface[mpi] "
                    "after loading the compiler/MPI module stack."
                ) from exc
            if MPI.Is_finalized():
                raise RuntimeError("MPI has already been finalized")
            if not MPI.Is_initialized():
                MPI.Init()
            mpi_comm = MPI.COMM_WORLD

        required = ("py2f", "Get_rank", "Get_size")
        missing = [name for name in required if not hasattr(mpi_comm, name)]
        if missing:
            raise TypeError(
                "mpi_comm must be an mpi4py-compatible communicator; missing "
                + ", ".join(missing)
            )
        try:
            rank = int(mpi_comm.Get_rank())
            size = int(mpi_comm.Get_size())
            fortran_handle = ctypes.c_int(int(mpi_comm.py2f()))
        except Exception as exc:
            raise ValueError("mpi_comm is not a valid active MPI communicator") from exc
        if size < 1 or rank < 0 or rank >= size:
            raise ValueError(f"Invalid MPI communicator rank/size: rank={rank}, size={size}")

        self.lib.w90_set_comm_f(self.data, ctypes.byref(fortran_handle))
        self.mpi_comm = mpi_comm
        self.mpi_rank = rank
        self.mpi_size = size

    def set_option(self, key: str, value: Any) -> None:
        """Set a Wannier90 option, dispatching to its type-specific C function."""

        encoded_key = str(key).encode()
        if isinstance(value, (bool, np.bool_)):
            self.lib.w90_set_option_logical(self.data, encoded_key, bool(value))
            return
        if isinstance(value, (int, np.integer)):
            self.lib.w90_set_option_int(self.data, encoded_key, int(value))
            return
        if isinstance(value, (float, np.floating)):
            self.lib.w90_set_option_double(self.data, encoded_key, float(value))
            return
        if isinstance(value, (str, os.PathLike)):
            self.lib.w90_set_option_text(
                self.data, encoded_key, os.fspath(value).encode()
            )
            return
        if isinstance(value, (list, tuple)) and len(value) > 0 and all(
            isinstance(item, str) for item in value
        ):
            encoded = [item.encode() for item in value]
            if str(key).lower() == "projections":
                # Projection definitions are separate settings entries in the
                # native Fortran examples and parser.
                for item in encoded:
                    self.lib.w90_set_option_text(self.data, encoded_key, item)
            else:
                # Character vectors such as symbols are a single c2d entry.
                string_array = (ctypes.c_char_p * len(encoded))(*encoded)
                self.lib.w90_set_option_text2d(
                    self.data, encoded_key, string_array, len(encoded)
                )
            return

        array = np.asarray(value)
        if array.ndim not in (1, 2):
            raise TypeError(f"Wannier90 option {key!r} must be scalar, 1-D, or 2-D")
        if np.issubdtype(array.dtype, np.integer):
            array = np.ascontiguousarray(array, dtype=np.int32)
            if array.ndim == 1:
                self.lib.w90_set_option_int1d(
                    self.data, encoded_key, _array_pointer(array), array.shape[0]
                )
            else:
                self.lib.w90_set_option_int2d(
                    self.data,
                    encoded_key,
                    _array_pointer(array),
                    array.shape[0],
                    array.shape[1],
                )
            return
        if np.issubdtype(array.dtype, np.number):
            array = np.ascontiguousarray(array, dtype=np.float64)
            if array.ndim == 1:
                self.lib.w90_set_option_double1d(
                    self.data, encoded_key, _array_pointer(array), array.shape[0]
                )
            else:
                self.lib.w90_set_option_double2d(
                    self.data,
                    encoded_key,
                    _array_pointer(array),
                    array.shape[0],
                    array.shape[1],
                )
            return
        raise TypeError(f"Unsupported value for Wannier90 option {key!r}: {value!r}")

    def input_setopt(self, seedname: str) -> None:
        self._status_call("w90_input_setopt", seedname.encode())

    def input_reader(self) -> None:
        self._status_call("w90_input_reader")

    def _status_call(self, name: str, *args: Any, allow_negative: bool = False) -> int:
        ierr = ctypes.c_int(0)
        getattr(self.lib, name)(self.data, *args, ctypes.byref(ierr))
        if ierr.value > 0 or (ierr.value < 0 and not allow_negative):
            raise Wannier90Error(f"{name} failed with Wannier90 error code {ierr.value}")
        return ierr.value

    def get_count(self, name: str) -> int:
        result = ctypes.c_int(0)
        getattr(self.lib, name)(self.data, ctypes.byref(result))
        return result.value

    def get_neighbours(self, num_kpts: int) -> tuple[np.ndarray, np.ndarray]:
        nntot = ctypes.c_int(0)
        ierr = ctypes.c_int(0)
        self.lib.w90_get_nn(self.data, ctypes.byref(nntot), ctypes.byref(ierr))
        if ierr.value != 0:
            raise Wannier90Error(f"w90_get_nn failed with error code {ierr.value}")
        nnkp = np.empty((num_kpts, nntot.value), dtype=np.int32, order="F")
        gkpb = np.empty((3, num_kpts, nntot.value), dtype=np.int32, order="F")
        self.lib.w90_get_nnkp(self.data, _array_pointer(nnkp), ctypes.byref(ierr))
        if ierr.value != 0:
            raise Wannier90Error(f"w90_get_nnkp failed with error code {ierr.value}")
        self.lib.w90_get_gkpb(self.data, _array_pointer(gkpb), ctypes.byref(ierr))
        if ierr.value != 0:
            raise Wannier90Error(f"w90_get_gkpb failed with error code {ierr.value}")
        return nnkp, gkpb

    def get_excluded_bands(self) -> np.ndarray:
        count = ctypes.c_int(0)
        self.lib.w90_get_num_excl_bands(self.data, ctypes.byref(count))
        if count.value == 0:
            return np.empty(0, dtype=np.int32)
        result = np.empty(count.value, dtype=np.int32)
        ierr = ctypes.c_int(0)
        self.lib.w90_get_excl_bands(
            self.data, _array_pointer(result), ctypes.byref(ierr)
        )
        if ierr.value != 0:
            raise Wannier90Error(
                f"w90_get_excl_bands failed with error code {ierr.value}"
            )
        return result

    def get_projections(self) -> dict[str, np.ndarray]:
        count = ctypes.c_int(0)
        self.lib.w90_get_nproj(self.data, ctypes.byref(count))
        if count.value <= 0:
            raise Wannier90Error(
                "Wannier90 has no parsed projections; provide projections or set "
                "use_bloch_phases=True"
            )
        nproj = count.value
        site = np.empty((3, nproj), dtype=np.float64, order="F")
        x_axis = np.empty((3, nproj), dtype=np.float64, order="F")
        z_axis = np.empty((3, nproj), dtype=np.float64, order="F")
        spin_axis = np.empty((3, nproj), dtype=np.float64, order="F")
        l = np.empty(nproj, dtype=np.int32)
        m = np.empty(nproj, dtype=np.int32)
        spin = np.empty(nproj, dtype=np.int32)
        radial = np.empty(nproj, dtype=np.int32)
        zona = np.empty(nproj, dtype=np.float64)
        ierr = ctypes.c_int(0)
        self.lib.w90_get_proj(
            self.data,
            ctypes.byref(count),
            _array_pointer(site),
            _array_pointer(l),
            _array_pointer(m),
            _array_pointer(spin),
            _array_pointer(radial),
            _array_pointer(x_axis),
            _array_pointer(z_axis),
            _array_pointer(spin_axis),
            _array_pointer(zona),
            ctypes.byref(ierr),
        )
        if ierr.value != 0:
            raise Wannier90Error(f"w90_get_proj failed with error code {ierr.value}")
        return {
            "site": site,
            "l": l,
            "m": m,
            "spin": spin,
            "radial": radial,
            "x_axis": x_axis,
            "z_axis": z_axis,
            "spin_axis": spin_axis,
            "zona": zona,
        }

    def attach_matrices(
        self,
        eigenvalues: np.ndarray,
        m_matrix: np.ndarray,
        u_matrix: np.ndarray,
        u_opt: np.ndarray,
    ) -> None:
        self._matrix_owners = (eigenvalues, m_matrix, u_matrix, u_opt)
        self.lib.w90_set_eigval(self.data, _array_pointer(eigenvalues))
        self.lib.w90_set_m_local(self.data, _array_pointer(m_matrix))
        self.lib.w90_set_u_matrix(self.data, _array_pointer(u_matrix))
        self.lib.w90_set_u_opt(self.data, _array_pointer(u_opt))

    def disentangle(self) -> None:
        self._status_call("w90_disentangle")

    def project_overlap(self) -> None:
        self._status_call("w90_project_overlap")

    def wannierise(self) -> int:
        return self._status_call("w90_wannierise", allow_negative=True)

    def get_centres_and_spreads(self, num_wann: int) -> tuple[np.ndarray, np.ndarray]:
        centres = np.empty((3, num_wann), dtype=np.float64, order="F")
        spreads = np.empty(num_wann, dtype=np.float64)
        self.lib.w90_get_centres(
            self.data, centres.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        self.lib.w90_get_spreads(
            self.data, spreads.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return centres, spreads

    def close(self) -> None:
        if not self._closed:
            self.lib.w90_delete(ctypes.byref(self.data))
            self._matrix_owners = ()
            self._closed = True

    def __enter__(self) -> "Wannier90V4Library":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _transform_axes(x_axis: Sequence[float], z_axis: Sequence[float]) -> np.ndarray:
    x = np.array(x_axis, dtype=float, copy=True)
    z = np.array(z_axis, dtype=float, copy=True)
    x_norm = np.linalg.norm(x)
    z_norm = np.linalg.norm(z)
    if x_norm < 1.0e-14 or z_norm < 1.0e-14:
        raise ValueError("Projection axes must be nonzero")
    x /= x_norm
    z /= z_norm
    if not np.isclose(np.dot(x, z), 0.0, atol=1.0e-10):
        raise ValueError("Projection x_axis and z_axis must be orthogonal")
    y = -np.cross(x, z)
    return np.vstack((x, y, z))


def _radial_function(radius: np.ndarray, radial: int, zona: float) -> np.ndarray:
    if radial == 1:
        return 2.0 * zona**1.5 * np.exp(-zona * radius)
    if radial == 2:
        return (
            zona**1.5
            * (2.0 - zona * radius)
            * np.exp(-zona * radius / 2.0)
            / (2.0 * np.sqrt(2.0))
        )
    if radial == 3:
        return (
            np.sqrt(4.0 / 27.0)
            * zona**1.5
            * (1.0 - 2.0 * zona * radius / 3.0 + 2.0 * zona**2 * radius**2 / 27.0)
            * np.exp(-zona * radius / 3.0)
        )
    raise ValueError(f"Unsupported Wannier90 radial index: {radial}")


def _angular_function(name: str, cost: np.ndarray, phi: np.ndarray) -> np.ndarray:
    sint = np.sqrt(np.maximum(0.0, 1.0 - cost**2))
    functions = {
        "s": np.ones_like(cost) / np.sqrt(4.0 * np.pi),
        "pz": np.sqrt(3.0 / (4.0 * np.pi)) * cost,
        "px": np.sqrt(3.0 / (4.0 * np.pi)) * sint * np.cos(phi),
        "py": np.sqrt(3.0 / (4.0 * np.pi)) * sint * np.sin(phi),
        "dz2": np.sqrt(5.0 / (16.0 * np.pi)) * (3.0 * cost**2 - 1.0),
        "dxz": np.sqrt(15.0 / (4.0 * np.pi)) * sint * cost * np.cos(phi),
        "dyz": np.sqrt(15.0 / (4.0 * np.pi)) * sint * cost * np.sin(phi),
        "dx2-y2": np.sqrt(15.0 / (16.0 * np.pi)) * sint**2 * np.cos(2.0 * phi),
        "dxy": np.sqrt(15.0 / (16.0 * np.pi)) * sint**2 * np.sin(2.0 * phi),
        "fz3": np.sqrt(7.0) * (5.0 * cost**3 - 3.0 * cost) / (4.0 * np.sqrt(np.pi)),
        "fxz2": np.sqrt(21.0) * (5.0 * cost**2 - 1.0) * sint * np.cos(phi) / (4.0 * np.sqrt(2.0 * np.pi)),
        "fyz2": np.sqrt(21.0) * (5.0 * cost**2 - 1.0) * sint * np.sin(phi) / (4.0 * np.sqrt(2.0 * np.pi)),
        "fz(x2-y2)": np.sqrt(105.0) * sint**2 * cost * np.cos(2.0 * phi) / (4.0 * np.sqrt(np.pi)),
        "fxyz": np.sqrt(105.0) * sint**2 * cost * np.sin(2.0 * phi) / (4.0 * np.sqrt(np.pi)),
        "fx(x2-3y2)": np.sqrt(35.0) * sint**3 * (np.cos(phi) ** 2 - 3.0 * np.sin(phi) ** 2) * np.cos(phi) / (4.0 * np.sqrt(2.0 * np.pi)),
        "fy(3x2-y2)": np.sqrt(35.0) * sint**3 * (3.0 * np.cos(phi) ** 2 - np.sin(phi) ** 2) * np.sin(phi) / (4.0 * np.sqrt(2.0 * np.pi)),
    }
    try:
        return functions[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported angular projection {name!r}") from exc


def _theta_lmr(l: int, mr: int, cost: np.ndarray, phi: np.ndarray) -> np.ndarray:
    orbital_names = {
        0: ("s",),
        1: ("pz", "px", "py"),
        2: ("dz2", "dxz", "dyz", "dx2-y2", "dxy"),
        3: ("fz3", "fxz2", "fyz2", "fz(x2-y2)", "fxyz", "fx(x2-3y2)", "fy(3x2-y2)"),
    }
    if l >= 0:
        try:
            return _angular_function(orbital_names[l][mr - 1], cost, phi)
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Unsupported Wannier90 projection (l={l}, mr={mr})") from exc

    s = _angular_function("s", cost, phi)
    px = _angular_function("px", cost, phi)
    py = _angular_function("py", cost, phi)
    pz = _angular_function("pz", cost, phi)
    dz2 = _angular_function("dz2", cost, phi)
    dx2y2 = _angular_function("dx2-y2", cost, phi)
    hybrids: dict[int, tuple[np.ndarray, ...]] = {
        -1: ((s + px) / np.sqrt(2.0), (s - px) / np.sqrt(2.0)),
        -2: (
            s / np.sqrt(3.0) - px / np.sqrt(6.0) + py / np.sqrt(2.0),
            s / np.sqrt(3.0) - px / np.sqrt(6.0) - py / np.sqrt(2.0),
            s / np.sqrt(3.0) + 2.0 * px / np.sqrt(6.0),
        ),
        -3: (
            (s + px + py + pz) / 2.0,
            (s + px - py - pz) / 2.0,
            (s - px + py - pz) / 2.0,
            (s - px - py + pz) / 2.0,
        ),
        -4: (
            s / np.sqrt(3.0) - px / np.sqrt(6.0) + py / np.sqrt(2.0),
            s / np.sqrt(3.0) - px / np.sqrt(6.0) - py / np.sqrt(2.0),
            s / np.sqrt(3.0) + 2.0 * px / np.sqrt(6.0),
            (pz + dz2) / np.sqrt(2.0),
            (-pz + dz2) / np.sqrt(2.0),
        ),
        -5: (
            s / np.sqrt(6.0) - px / np.sqrt(2.0) - dz2 / np.sqrt(12.0) + dx2y2 / 2.0,
            s / np.sqrt(6.0) + px / np.sqrt(2.0) - dz2 / np.sqrt(12.0) + dx2y2 / 2.0,
            s / np.sqrt(6.0) - py / np.sqrt(2.0) - dz2 / np.sqrt(12.0) - dx2y2 / 2.0,
            s / np.sqrt(6.0) + py / np.sqrt(2.0) - dz2 / np.sqrt(12.0) - dx2y2 / 2.0,
            s / np.sqrt(6.0) - pz / np.sqrt(2.0) + dz2 / np.sqrt(3.0),
            s / np.sqrt(6.0) + pz / np.sqrt(2.0) + dz2 / np.sqrt(3.0),
        ),
    }
    try:
        return hybrids[l][mr - 1]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unsupported Wannier90 hybrid projection (l={l}, mr={mr})") from exc


def projection_function(
    coordinates: np.ndarray,
    site: Sequence[float],
    l: int,
    mr: int,
    radial: int,
    zona: float,
    x_axis: Sequence[float] = (1.0, 0.0, 0.0),
    z_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Evaluate a Wannier90 hydrogenic projection on a PySCF Bohr grid."""

    relative = np.asarray(coordinates, dtype=float) - np.asarray(site, dtype=float)
    relative = relative @ _transform_axes(x_axis, z_axis).T
    radius = np.linalg.norm(relative, axis=1)
    safe_radius = np.where(radius < 1.0e-12, 1.0e-12, radius)
    cost = np.clip(relative[:, 2] / safe_radius, -1.0, 1.0)
    phi = np.arctan2(relative[:, 1], relative[:, 0])
    return _theta_lmr(l, mr, cost, phi) * _radial_function(
        safe_radius, radial, zona
    )


def _parse_legacy_keywords(text: str | None) -> dict[str, Any]:
    """Parse the common subset of the original ``other_keywords`` format."""

    if not text:
        return {}
    lines = text.splitlines()
    parsed: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        line = re.split(r"[#!]", lines[index], maxsplit=1)[0].strip()
        index += 1
        if not line:
            continue
        block_match = re.match(r"(?i)^begin\s+([a-z0-9_]+)$", line)
        if block_match:
            key = block_match.group(1).lower()
            values: list[str] = []
            while index < len(lines):
                block_line = re.split(r"[#!]", lines[index], maxsplit=1)[0].strip()
                index += 1
                if re.match(rf"(?i)^end\s+{re.escape(key)}$", block_line):
                    break
                if block_line:
                    values.append(block_line)
            else:
                raise ValueError(f"Unterminated Wannier90 block: {key}")
            parsed[key] = values
            continue

        match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s*(?:=|:)\s*(.*?)\s*$", line)
        if not match:
            raise ValueError(f"Cannot parse Wannier90 option line: {line!r}")
        key, raw_value = match.group(1).lower(), match.group(2).strip()
        lowered = raw_value.lower()
        if lowered in {"true", ".true.", "t"}:
            parsed[key] = True
            continue
        if lowered in {"false", ".false.", "f"}:
            parsed[key] = False
            continue
        tokens = [token for token in re.split(r"[\s,]+", raw_value) if token]
        try:
            integers = [int(token) for token in tokens]
        except ValueError:
            integers = []
        if integers and len(integers) == len(tokens):
            parsed[key] = integers[0] if len(integers) == 1 else integers
            continue
        try:
            numbers = [float(token.replace("d", "e").replace("D", "E")) for token in tokens]
        except ValueError:
            numbers = []
        if numbers and len(numbers) == len(tokens):
            parsed[key] = numbers[0] if len(numbers) == 1 else numbers
            continue
        parsed[key] = raw_value.strip("'\"")
    return parsed


def _default_kpoint_distribution(num_kpts: int, mpi_size: int) -> np.ndarray:
    """Return Wannier90's contiguous, balanced k-point-to-rank mapping."""

    if num_kpts < 1:
        raise ValueError("num_kpts must be positive")
    if mpi_size < 1:
        raise ValueError("mpi_size must be positive")
    counts = np.full(mpi_size, num_kpts // mpi_size, dtype=np.int32)
    counts[: num_kpts % mpi_size] += 1
    return np.repeat(np.arange(mpi_size, dtype=np.int32), counts)


class W90:
    """Wannier90 v4 workflow for periodic PySCF mean-field objects.

    Parameters largely follow the original :class:`pywannier90.W90` API. New
    code should prefer ``options`` and ``projections`` over the legacy
    ``other_keywords`` string.
    """

    _RESERVED_OPTIONS = {
        "kpoints",
        "mp_grid",
        "total_bands",
        "num_bands",
        "num_wann",
        "unit_cell_cart",
        "symbols",
        "atoms_cart",
        "atoms_frac",
        "distk",
    }

    def __init__(
        self,
        kmf: Any,
        cell: Any,
        mp_grid: Sequence[int],
        num_wann: int,
        gamma: bool = False,
        spinors: bool = False,
        spin: str = "up",
        other_keywords: str | None = None,
        *,
        library_path: str | os.PathLike[str] | None = None,
        seedname: str | os.PathLike[str] = "wannier90",
        projections: Sequence[str] | None = None,
        exclude_bands: Sequence[int] | None = None,
        use_bloch_phases: bool = False,
        options: Mapping[str, Any] | None = None,
        mpi_comm: Any | None = None,
        kpoint_distribution: Sequence[int] | None = None,
    ):
        if spinors:
            raise NotImplementedError("Spinor PySCF wavefunctions are not yet supported")
        self.kmf = kmf
        self.cell = cell
        self.mp_grid = np.asarray(mp_grid, dtype=np.int32)
        if self.mp_grid.shape != (3,) or np.any(self.mp_grid <= 0):
            raise ValueError("mp_grid must contain three positive integers")
        self.num_wann = int(num_wann)
        self.gamma_only = bool(gamma)
        self.spinors = bool(spinors)
        self.spin = spin.lower()
        self.library_path = library_path
        self.mpi_comm = mpi_comm
        self.requested_kpoint_distribution = (
            None
            if kpoint_distribution is None
            else np.asarray(kpoint_distribution, dtype=np.int32)
        )
        self.seedname = str(Path(seedname).expanduser().resolve())

        legacy_options = _parse_legacy_keywords(other_keywords)
        legacy_projections = legacy_options.pop("projections", None)
        legacy_excluded = legacy_options.pop("exclude_bands", None)
        legacy_bloch = legacy_options.pop("use_bloch_phases", False)
        merged_options = dict(legacy_options)
        merged_options.update(options or {})
        overlap = self._RESERVED_OPTIONS.intersection(key.lower() for key in merged_options)
        if overlap:
            raise ValueError(
                "These options are managed by the PySCF adapter and cannot be overridden: "
                + ", ".join(sorted(overlap))
            )
        self.options = merged_options
        self.projections = list(projections if projections is not None else legacy_projections or [])
        excluded = exclude_bands if exclude_bands is not None else legacy_excluded or []
        if isinstance(excluded, (int, np.integer)):
            excluded = [int(excluded)]
        self.requested_exclude_bands = np.asarray(excluded, dtype=np.int32)
        self.use_bloch_phases = bool(use_bloch_phases or legacy_bloch)

        self.kpts = np.asarray(kmf.kpts, dtype=np.float64)
        if self.kpts.ndim == 1:
            self.kpts = self.kpts.reshape(1, 3)
        self.num_kpts = self.kpts.shape[0]
        if self.num_kpts != int(np.prod(self.mp_grid)):
            raise ValueError(
                f"PySCF has {self.num_kpts} k-points but mp_grid contains "
                f"{int(np.prod(self.mp_grid))} points"
            )

        from pyscf.data import nist

        self.bohr_to_angstrom = float(nist.BOHR)
        self.hartree_to_ev = float(nist.HARTREE2EV)
        self.real_lattice = np.asarray(cell.lattice_vectors(), dtype=np.float64) * self.bohr_to_angstrom
        self.kpt_latt = np.asarray(cell.get_scaled_kpts(self.kpts), dtype=np.float64)
        self.atom_symbols = [cell.atom_symbol(index) for index in range(cell.natm)]
        self.atoms_cart = np.asarray(cell.atom_coords(), dtype=np.float64) * self.bohr_to_angstrom

        self.mo_energy, self.mo_coeff = self._extract_orbitals(kmf)
        self.num_bands_total = min(energy.size for energy in self.mo_energy)
        self.mo_energy = np.asarray(
            [energy[: self.num_bands_total] for energy in self.mo_energy], dtype=np.float64
        )
        self.mo_coeff = np.asarray(
            [coeff[:, : self.num_bands_total] for coeff in self.mo_coeff],
            dtype=np.complex128,
        )

        self.library: Wannier90V4Library | None = None
        self.is_mpi = False
        self.mpi_rank = 0
        self.mpi_size = 1
        self.dist_kpoints = np.zeros(self.num_kpts, dtype=np.int32)
        self.local_kpoint_indices = np.arange(self.num_kpts, dtype=np.int32)
        self.nntot: int | None = None
        self.nnkp: np.ndarray | None = None
        self.gkpb: np.ndarray | None = None
        self.exclude_bands = np.empty(0, dtype=np.int32)
        self.band_included_list: list[int] = []
        self.num_bands: int | None = None
        self.projection_data: dict[str, np.ndarray] | None = None

        self.M_matrix: np.ndarray | None = None
        self.M_matrix_local: np.ndarray | None = None
        self.A_matrix: np.ndarray | None = None
        self.eigenvalues: np.ndarray | None = None
        self.u_matrix: np.ndarray | None = None
        self.u_matrix_opt: np.ndarray | None = None
        self.U_matrix: np.ndarray | None = None
        self.U_matrix_opt: np.ndarray | None = None
        self.wann_centres: np.ndarray | None = None
        self.wann_spreads: np.ndarray | None = None
        self.converged: bool | None = None

        self._validate_dlwf_options()

    def _extract_orbitals(self, kmf: Any) -> tuple[list[np.ndarray], list[np.ndarray]]:
        energies = getattr(kmf, "mo_energy", None)
        coefficients = getattr(kmf, "mo_coeff", None)
        if energies is None:
            energies = getattr(kmf, "mo_energy_kpts", None)
        if coefficients is None:
            coefficients = getattr(kmf, "mo_coeff_kpts", None)
        if energies is None or coefficients is None:
            raise TypeError("The PySCF object must provide mo_energy and mo_coeff")

        coeff_array = np.asarray(coefficients)
        unrestricted = coeff_array.ndim == 4 or (
            isinstance(coefficients, (list, tuple))
            and len(coefficients) == 2
            and np.asarray(coefficients[0]).ndim == 3
        )
        if unrestricted:
            if self.spin not in {"up", "down"}:
                raise ValueError("Unrestricted calculations require spin='up' or spin='down'")
            spin_index = 0 if self.spin == "up" else 1
            energies = energies[spin_index]
            coefficients = coefficients[spin_index]
        elif self.spin not in {"up", "down"}:
            raise ValueError("spin must be 'up' or 'down'")

        energy_array = np.asarray(energies)
        coefficient_array = np.asarray(coefficients)
        if self.num_kpts == 1 and energy_array.ndim == 1:
            energy_array = energy_array[None, :]
        if self.num_kpts == 1 and coefficient_array.ndim == 2:
            coefficient_array = coefficient_array[None, :, :]
        if energy_array.shape[0] != self.num_kpts or coefficient_array.shape[0] != self.num_kpts:
            raise ValueError("MO energies/coefficients do not match the PySCF k-point count")
        return [np.asarray(item) for item in energy_array], [np.asarray(item) for item in coefficient_array]

    def _validate_dlwf_options(self) -> None:
        mixing = float(self.options.get("sp_en_mix", 0.0))
        if mixing == 0.0:
            return
        if self.gamma_only:
            raise ValueError("Wannier90 v4 space-energy localization does not support gamma_only")
        incompatible = {
            "site_symmetry": True,
            "selective_loc": True,
            "use_ss_functional": True,
            "precond": True,
        }
        enabled = [key for key, value in incompatible.items() if bool(self.options.get(key, False)) == value]
        if enabled:
            raise ValueError(
                "Wannier90 v4 space-energy localization is incompatible with: "
                + ", ".join(enabled)
            )

    def setup(self) -> None:
        if self.library is not None:
            raise RuntimeError("This W90 instance has already been set up")
        library = Wannier90V4Library(self.library_path, mpi_comm=self.mpi_comm)
        try:
            self.is_mpi = library.is_mpi
            self.mpi_comm = library.mpi_comm
            self.mpi_rank = library.mpi_rank
            self.mpi_size = library.mpi_size
            if self.requested_kpoint_distribution is None:
                distribution = _default_kpoint_distribution(
                    self.num_kpts, self.mpi_size
                )
            else:
                distribution = self.requested_kpoint_distribution
                if distribution.shape != (self.num_kpts,):
                    raise ValueError(
                        "kpoint_distribution must contain one rank for each "
                        f"k-point; expected {self.num_kpts}, got {distribution.size}"
                    )
                if np.any(distribution < 0) or np.any(distribution >= self.mpi_size):
                    raise ValueError(
                        "kpoint_distribution contains a rank outside the active "
                        f"communicator of size {self.mpi_size}"
                    )
                distribution = np.array(distribution, dtype=np.int32, copy=True)
            self.dist_kpoints = distribution
            self.local_kpoint_indices = np.flatnonzero(
                distribution == self.mpi_rank
            ).astype(np.int32)

            library.set_option("kpoints", self.kpt_latt)
            library.set_option("mp_grid", self.mp_grid)
            library.set_option("distk", self.dist_kpoints)
            library.set_option("total_bands", self.num_bands_total)
            library.set_option("num_wann", self.num_wann)
            library.set_option("unit_cell_cart", self.real_lattice)
            # The v4 library API takes atomic symbols and coordinates as two
            # separate vectors.  This differs from the atoms_cart text block
            # accepted by a .win file.
            library.set_option("symbols", self.atom_symbols)
            library.set_option("atoms_cart", self.atoms_cart)
            library.set_option("gamma_only", self.gamma_only)
            library.set_option("spinors", self.spinors)
            library.set_option("use_bloch_phases", self.use_bloch_phases)
            if self.requested_exclude_bands.size:
                library.set_option("exclude_bands", self.requested_exclude_bands)
            if self.projections:
                library.set_option("projections", self.projections)
            elif self.use_bloch_phases:
                # input_setopt still validates that an initial projection set
                # exists, even though project_overlap will receive the identity
                # A matrix in Bloch-phase mode.
                library.set_option("projections", ["random"])
            for key, value in self.options.items():
                library.set_option(key, value)
            Path(self.seedname).parent.mkdir(parents=True, exist_ok=True)
            library.input_setopt(self.seedname)

            returned_nk = library.get_count("w90_get_nk")
            returned_nw = library.get_count("w90_get_nw")
            if returned_nk != self.num_kpts or returned_nw != self.num_wann:
                raise Wannier90Error(
                    "Wannier90 returned inconsistent dimensions: "
                    f"nk={returned_nk}, nw={returned_nw}"
                )
            self.nnkp, self.gkpb = library.get_neighbours(self.num_kpts)
            self.nntot = self.nnkp.shape[1]
            self.exclude_bands = library.get_excluded_bands()
            excluded = set(int(index) for index in self.exclude_bands)
            self.band_included_list = [
                index for index in range(self.num_bands_total) if index + 1 not in excluded
            ]
            self.num_bands = len(self.band_included_list)
            if self.num_bands < self.num_wann:
                raise Wannier90Error("Fewer included bands than requested Wannier functions")
            if not self.use_bloch_phases:
                self.projection_data = library.get_projections()
                if self.projection_data["l"].size != self.num_wann:
                    raise NotImplementedError(
                        "The current PySCF projection builder requires exactly num_wann projections"
                    )
            self.library = library
        except Exception:
            library.close()
            raise

    def get_M_mat(self) -> np.ndarray:
        if self.nnkp is None or self.gkpb is None or self.num_bands is None:
            raise RuntimeError("Call setup() before constructing the M matrix")
        from pyscf.pbc.df import ft_ao

        result = np.empty(
            (
                self.num_bands,
                self.num_bands,
                self.nntot,
                self.local_kpoint_indices.size,
            ),
            dtype=np.complex128,
            order="F",
        )
        for local_index, k_index_value in enumerate(self.local_kpoint_indices):
            k_index = int(k_index_value)
            k1 = self.cell.get_abs_kpts(self.kpt_latt[k_index])
            coeff1 = self.mo_coeff[k_index][:, self.band_included_list]
            for neighbour in range(self.nntot):
                k2_index = int(self.nnkp[k_index, neighbour]) - 1
                k2_scaled = self.kpt_latt[k2_index] + self.gkpb[:, k_index, neighbour]
                k2 = self.cell.get_abs_kpts(k2_scaled)
                overlap_ao = ft_ao.ft_aopair(
                    self.cell,
                    Gv=np.asarray([k2 - k1]),
                    kpti_kptj=[k1, k2],
                    q=np.zeros(3),
                )[0]
                coeff2 = self.mo_coeff[k2_index][:, self.band_included_list]
                result[:, :, neighbour, local_index] = (
                    coeff1.conj().T @ overlap_ao @ coeff2
                )
        return result

    def get_A_mat(self) -> np.ndarray:
        if self.num_bands is None:
            raise RuntimeError("Call setup() before constructing the A matrix")
        result = np.zeros(
            (self.num_bands, self.num_wann, self.num_kpts),
            dtype=np.complex128,
            order="F",
        )
        if self.use_bloch_phases:
            for index in range(min(self.num_bands, self.num_wann)):
                result[index, index, :] = 1.0
            return result
        if self.projection_data is None:
            raise RuntimeError("Wannier90 projection data are unavailable")

        from pyscf.dft import gen_grid, numint

        grid = gen_grid.Grids(self.cell).build()
        coordinates = grid.coords
        weights = grid.weights
        ao_values = numint.eval_ao(self.cell, coordinates)
        lattice_bohr = np.asarray(self.cell.lattice_vectors(), dtype=np.float64)

        for wannier_index in range(self.num_wann):
            fractional_site = self.projection_data["site"][:, wannier_index]
            absolute_site = fractional_site @ lattice_bohr
            projector = projection_function(
                coordinates,
                absolute_site,
                int(self.projection_data["l"][wannier_index]),
                int(self.projection_data["m"][wannier_index]),
                int(self.projection_data["radial"][wannier_index]),
                float(self.projection_data["zona"][wannier_index]),
                self.projection_data["x_axis"][:, wannier_index],
                self.projection_data["z_axis"][:, wannier_index],
            )
            ao_projector = np.einsum("g,g,gv->v", weights, projector, ao_values)
            for k_index, kpoint in enumerate(self.kpts):
                overlap = self.cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpoint)
                coefficients = self.mo_coeff[k_index][:, self.band_included_list]
                result[:, wannier_index, k_index] = np.einsum(
                    "v,vu,um->m", ao_projector, overlap, coefficients, optimize=True
                ).conj()
        return result

    def get_epsilon_mat(self) -> np.ndarray:
        if not self.band_included_list:
            raise RuntimeError("Call setup() before constructing the eigenvalue matrix")
        values = self.mo_energy[:, self.band_included_list] * self.hartree_to_ev
        return np.asfortranarray(values.T, dtype=np.float64)

    def run(
        self,
        m_matrix: np.ndarray | None = None,
        a_matrix: np.ndarray | None = None,
        eigenvalues: np.ndarray | None = None,
    ) -> "W90":
        if self.library is None or self.num_bands is None or self.nntot is None:
            raise RuntimeError("Call setup() before run()")
        m_matrix = self.get_M_mat() if m_matrix is None else m_matrix
        a_matrix = self.get_A_mat() if a_matrix is None else a_matrix
        eigenvalues = self.get_epsilon_mat() if eigenvalues is None else eigenvalues

        expected_m = (
            self.num_bands,
            self.num_bands,
            self.nntot,
            self.local_kpoint_indices.size,
        )
        global_m = (self.num_bands, self.num_bands, self.nntot, self.num_kpts)
        expected_a = (self.num_bands, self.num_wann, self.num_kpts)
        expected_e = (self.num_bands, self.num_kpts)
        supplied_m_shape = np.shape(m_matrix)
        if supplied_m_shape == global_m and global_m != expected_m:
            m_matrix = np.take(m_matrix, self.local_kpoint_indices, axis=3)
        elif supplied_m_shape != expected_m:
            allowed = f"{expected_m} (rank-local)"
            if global_m != expected_m:
                allowed += f" or {global_m} (global)"
            raise ValueError(f"M matrix shape must be {allowed}, got {supplied_m_shape}")
        if np.shape(a_matrix) != expected_a:
            raise ValueError(f"A matrix shape must be {expected_a}, got {np.shape(a_matrix)}")
        if np.shape(eigenvalues) != expected_e:
            raise ValueError(
                f"Eigenvalue matrix shape must be {expected_e}, got {np.shape(eigenvalues)}"
            )

        self.M_matrix = np.asfortranarray(m_matrix, dtype=np.complex128)
        self.M_matrix_local = self.M_matrix
        self.A_matrix = np.asfortranarray(a_matrix, dtype=np.complex128)
        self.eigenvalues = np.asfortranarray(eigenvalues, dtype=np.float64)
        self.u_matrix = np.zeros(
            (self.num_wann, self.num_wann, self.num_kpts),
            dtype=np.complex128,
            order="F",
        )
        self.u_matrix_opt = self.A_matrix.copy(order="F")
        self.library.attach_matrices(
            self.eigenvalues, self.M_matrix, self.u_matrix, self.u_matrix_opt
        )
        self.library.disentangle()
        self.library.project_overlap()
        status = self.library.wannierise()
        self.converged = status == 0
        centres, spreads = self.library.get_centres_and_spreads(self.num_wann)
        self.wann_centres = centres.T.copy()
        self.wann_spreads = spreads.copy()

        # Backward-compatible, k-point-major views used by original pyWannier90.
        self.U_matrix = np.transpose(self.u_matrix, (2, 0, 1)).copy()
        self.U_matrix_opt = np.transpose(self.u_matrix_opt, (2, 1, 0)).copy()
        return self

    def kernel(self) -> "W90":
        self.setup()
        return self.run()

    def get_hamiltonian_kpts(self) -> np.ndarray:
        """Return the Wannier-gauge Hamiltonian at the sampled k-points in Hartree."""

        if self.u_matrix is None or self.u_matrix_opt is None:
            raise RuntimeError("Run wannierization before constructing the Hamiltonian")
        result = np.empty(
            (self.num_kpts, self.num_wann, self.num_wann), dtype=np.complex128
        )
        for k_index in range(self.num_kpts):
            energies = self.mo_energy[k_index, self.band_included_list]
            u_opt = self.u_matrix_opt[:, :, k_index]
            u = self.u_matrix[:, :, k_index]
            subspace_hamiltonian = u_opt.conj().T @ (energies[:, None] * u_opt)
            result[k_index] = u.conj().T @ subspace_hamiltonian @ u
        return result

    def close(self) -> None:
        if self.library is not None:
            self.library.close()
            self.library = None

    def __enter__(self) -> "W90":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "W90",
    "Wannier90Error",
    "Wannier90V4Library",
    "projection_function",
]
