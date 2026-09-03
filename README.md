[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
![GitHub last commit](https://img.shields.io/github/last-commit/hungpham2017/pyWannier90.svg?color=green)
![GitHub issues](https://img.shields.io/github/issues-raw/hungpham2017/pyWannier90.svg?color=crimson)

# pyWannier90: A Python interface for wannier90
[wannier90](http://www.wannier.org/) is a well-established package to construct maximally-localied Wannier functions (MLWFs) as well as to perform MLWF-based analysis.
pyWannier90 uses the library-mode of wannier90 to perform the wannierization on the wave function obtained by PySCF or VASP.

<img src="https://github.com/hungpham2017/pyWannier90/blob/master/doc/Si_sp3.png" width="500" align="middle">

## PySCF 2.9 and Wannier90 v4 interface

`src/pywannier90_v4.py` is a new adapter for the official Wannier90 v4 C
library API. It keeps the legacy `src/pywannier90.py` implementation unchanged
for reproducibility. The v4 adapter has been exercised with PySCF 2.9.0 using
both ordinary maximally-localized Wannier functions and the v4 space-energy
(DLWF) localizer.

The current scope covers the core PySCF periodic workflow: restricted and
unrestricted k-point MO extraction, band exclusion, neighbor and overlap
construction, atomic projections or Bloch-phase initialization,
disentanglement, Wannierization, centers, spreads, and Wannier-gauge
Hamiltonians. Spinor wavefunctions are not yet supported. Wannier90 v4 itself
does not currently allow `gamma_only` with space-energy localization.

### SCDM-k initialization

The v4 adapter can build projection matrices directly from PySCF Bloch
orbitals with selected columns of the density matrix (SCDM-k). This removes
the need to choose atomic trial orbitals:

```python
with W90(
    kmf,
    cell,
    mp_grid=[4, 4, 4],
    num_wann=8,
    projections=None,
    use_bloch_phases=False,
    scdm_proj=True,
    scdm_entanglement="erfc",
    scdm_mu=33.0782,       # eV
    scdm_sigma=1.2366,     # eV, strictly positive
    scdm_grid=cell.mesh,   # optional; this is the default
    exclude_bands=exclude_bands,
    options={"dis_num_iter": 2000, "num_iter": 1000},
) as w90:
    w90.kernel()
```

The k mesh must include Gamma. At Gamma the adapter evaluates the included
PySCF orbitals on the uniform real-space grid and uses pivoted QR to select
`num_wann` representative grid points. It then evaluates the same points at
every k-point, applies the requested energy filter, and uses the SVD polar
factor to give the projection matrix orthonormal columns. This follows the
`pw2wannier90` convention for the A matrix. SCDM construction is performed on
MPI rank 0 and the comparatively small result is broadcast to other ranks;
Wannier90's later k-point work remains distributed normally.

Use `scdm_entanglement="isolated"` only when the included manifold already
has exactly `num_wann` bands. For an entangled outer space, `"erfc"` applies
`0.5 * erfc((energy - mu) / sigma)` and `"gaussian"` applies
`exp(-((energy - mu) / sigma)**2)`. All three SCDM energy arguments are in
electron-volts, matching Quantum ESPRESSO's interface. `scdm_mu`,
`scdm_sigma`, and the other `scdm_*` controls are constructor arguments, not
entries in the Wannier90 `options` dictionary.

### Recommended serial installation for the PySCF interface

When pyWannier90 is checked out as the submodule of the
[`Yichen-Fan/wannier90`](https://github.com/Yichen-Fan/wannier90/tree/dlwf)
`dlwf` branch, the parent repository provides a tested setup script. Start in
the parent Wannier90 directory, not inside this submodule:

```bash
git switch dlwf
git pull origin dlwf
git submodule update --init --recursive

./scripts/install_pywannier90.sh \
  --python "$(command -v python)" \
  --jobs "${SLURM_CPUS_PER_TASK:-8}"
```

The script performs the following steps:

1. Configures a Release build with the v4 C API and shared library enabled.
2. Builds and installs the serial Wannier90 library and executables.
3. Installs this Python adapter into the selected Python environment without
   upgrading NumPy or PySCF.
4. Generates an activation script containing `PYWANNIER90_LIB`, `PATH`, and
   the required library and package search paths.
5. Loads the installed library and runs the Python and native PySCF/DLWF tests.

The default installation prefix is
`$HOME/.local/wannier90-dlwf-4.0`. Activate it in each interactive or batch
shell with:

```bash
source "$HOME/.local/wannier90-dlwf-4.0/activate-pywannier90.sh"
```

Useful customizations include:

```bash
# Select a different installation and build location.
./scripts/install_pywannier90.sh \
  --prefix "$HOME/software/wannier90-dlwf-4.0" \
  --build-dir "$PWD/build/build-pywannier90" \
  --python "$(command -v python)" \
  --jobs 16

# Show all options.
./scripts/install_pywannier90.sh --help

# Build/install W90 without modifying the active Python environment.
./scripts/install_pywannier90.sh --skip-python --skip-tests
```

This installer deliberately builds the serial library with
`WANNIER90_MPI=OFF`; use the MPI installer below when rank-level distribution
is wanted. The setting does not disable threaded BLAS or OpenMP used by PySCF.
Compiler selection may be supplied in the normal CMake manner, for example:

```bash
CC=gcc FC=gfortran ./scripts/install_pywannier90.sh
```

The installer itself is maintained at
[`scripts/install_pywannier90.sh`](https://github.com/Yichen-Fan/wannier90/blob/dlwf/scripts/install_pywannier90.sh).

### MPI-enabled PySCF/Wannier90 installation

The parent repository provides a separate installer for MPI-enabled
Wannier90 and the Python adapter. Load a mutually compatible compiler and MPI
module stack first. The selected Python environment must contain `mpi4py`
built against that same MPI implementation, as well as PySCF 2.9 or newer.
Then run from the parent Wannier90 directory:

```bash
CC=gcc FC=mpifort ./scripts/install_pywannier90_mpi.sh \
  --python "$(command -v python)" \
  --jobs "${SLURM_CPUS_PER_TASK:-8}"

source "$HOME/.local/wannier90-dlwf-4.0-mpi/activate-wannier90-mpi.sh"

# Run the bundled PySCF/DLWF example with the MPI library.
mpiexec -n 4 python pyWannier90/examples/PySCF/h2_v4.py

# Or run the standalone Wannier90 executable.
mpiexec -n 4 wannier90.x seedname
```

After obtaining a compute-node allocation, the native two-rank regression can
be run from the parent repository with:

```bash
PYWANNIER90_TEST_LIB="$PYWANNIER90_LIB" \
  mpiexec -n 2 python -m unittest discover -s pyWannier90/tests -v
```

On a cluster, replace `gcc`, `mpifort`, and `mpiexec` with the compiler and MPI
wrappers supplied by the same module stack. Mixing compiler families can cause
errors such as a missing `ISO_Fortran_binding.h` while compiling the v4 C API.
Load the same MPI/compiler modules again in every shell or batch job before
sourcing the activation file and running `mpiexec`.
The installation and build locations can be customized independently of the
serial installation:

```bash
./scripts/install_pywannier90_mpi.sh \
  --prefix "$HOME/software/wannier90-dlwf-4.0-mpi" \
  --build-dir "$PWD/build/build-pywannier90-mpi" \
  --python "$(command -v python)" \
  --jobs 16

./scripts/install_pywannier90_mpi.sh --help

# Install only the MPI executable/library, without Python checks.
./scripts/install_pywannier90_mpi.sh --skip-python --skip-tests
```

The MPI build is kept separate because Wannier90 names its shared library
`libwannier90_mpi`, rather than `libwannier90`. Its activation file exports
both `WANNIER90_MPI_LIB` and `PYWANNIER90_LIB`, so the Python adapter discovers
the installed MPI library automatically.

When it loads `libwannier90_mpi`, `pywannier90_v4` initializes the library with
`mpi4py.MPI.COMM_WORLD`, supplies Wannier90's `distk` map, and constructs only
the rank-local part of the M matrix. The eigenvalues, projection matrix, and
final U matrices remain replicated, as required by the Wannier90 v4 library
API. A different communicator or distribution can be supplied explicitly:

```python
from mpi4py import MPI
from pywannier90_v4 import W90

w90 = W90(
    kmf,
    cell,
    mp_grid=[4, 4, 4],
    num_wann=8,
    mpi_comm=MPI.COMM_WORLD,
    kpoint_distribution=None,  # balanced contiguous distribution
)
w90.kernel()
```

Every MPI rank must enter the `W90` calls with an equivalent PySCF cell,
k-point mean-field result, and options. PySCF's SCF calculation is not made
MPI-parallel by this interface; for large jobs, load or reconstruct the same
converged mean-field data on every rank before calling `W90`.

The silicon valence-plus-conduction example demonstrates a staged hybrid job.
Its `scf` stage runs one threaded PySCF process and writes a reusable orbital
file; its `localize` stage loads that file on rank 0, broadcasts the orbitals,
and runs Wannier90 with one thread per MPI rank. The bundled Slurm script
reuses the same CPU allocation for both stages:

```bash
sbatch examples/PySCF/si_444_mpi.slurm

# The two stages can also be launched separately.
PYWANNIER90_STAGE=scf PYSCF_SCF_THREADS=4 \
  python examples/PySCF/si_444_valence_conduction_v4.py
PYWANNIER90_STAGE=localize PYWANNIER90_EXPECT_MPI_RANKS=4 \
  mpiexec -n 4 python examples/PySCF/si_444_valence_conduction_v4.py
```

The serial and MPI installations can coexist under their different default
prefixes. Source only the activation file for the variant needed by the job.

The MPI installer is maintained at
[`scripts/install_pywannier90_mpi.sh`](https://github.com/Yichen-Fan/wannier90/blob/dlwf/scripts/install_pywannier90_mpi.sh).

### Manual installation

Build Wannier90 v4 as a shared library with its C interface enabled:

```bash
cmake -S /path/to/wannier90 -B /path/to/w90-build \
  -DWANNIER90_WITH_C=ON \
  -DWANNIER90_SHARED_LIBS=ON \
  -DWANNIER90_MPI=OFF
cmake --build /path/to/w90-build
```

The C interface requires a Fortran compiler with the Fortran 2018 C-descriptor
support described in the Wannier90 v4 installation guide. Install this Python
adapter and point it to the resulting library:

```bash
python -m pip install -e .
export PYWANNIER90_LIB=/path/to/w90-build/libwannier90.so
# macOS: /path/to/w90-build/libwannier90.dylib
```

Then pass a converged periodic PySCF mean-field object:

```python
from pywannier90_v4 import W90

with W90(
    kmf,
    cell,
    mp_grid=[2, 2, 2],
    num_wann=4,
    seedname="output/seed",
    projections=["C:sp3"],
    options={
        "num_iter": 100,
        "sp_en_mix": 0.5,
        "sp_en_scale": 0.3781820678533007,
        "sp_en_power": 2.0,
    },
) as w90:
    w90.kernel()
    print(w90.wann_centres, w90.wann_spreads)
    w90.plot_wf(
        outfile="output/MLWF",
        supercell=[3, 3, 3],
        grid=[20, 20, 20],
    )
```

`plot_wf` reconstructs the final Wannier functions from the PySCF Bloch AOs
and the Wannier90 `U_opt U` transformation, then writes signed real amplitudes
as `output/MLWF-0.xsf`, `output/MLWF-1.xsf`, and so on. Wannier-function
indices are zero-based. In an MPI run only rank 0 performs plotting and file
output.

`other_keywords` remains available for the common legacy syntax, but new code
should use the typed `options`, `projections`, and `exclude_bands` arguments.
See `examples/PySCF/h2_v4.py` for a complete PySCF 2.9 DLWF calculation.

Run the Python-only tests with:

```bash
python -m unittest discover -s tests
```

Enable the end-to-end PySCF/Wannier90-v4 integration test by setting:

```bash
PYWANNIER90_TEST_LIB=/path/to/libwannier90.so \
  python -m unittest discover -s tests
```

## Attribution and license

This remains a BSD-3-Clause adaptation of pyWannier90, originally written by
Hung Q. Pham. The original copyright notice and license are retained in
`LICENSE`, and the v4 source and example carry explicit attribution. Wannier90
and PySCF are separate dependencies under their own licenses; see
`THIRD_PARTY_NOTICES.md` for credits and details.

## News:
- pyWannier90 is now available for wannier90 community, check it out [here](http://www.wannier.org/download/).
- The legacy interface supports Wannier90 v3-era library mode; the new module targets the Wannier90 v4 C API.

## pyWannier90 can use the wave function obtained by:
- PySCF 1.5 through the legacy interface, and PySCF 2.9 through `pywannier90_v4`
- VASP via the [MCU package](https://hungpham2017.github.io/mcu/)

## Why pyWannier90?
- If you would like to construct MLWFs for the wave function obtained by [PySCF](https://github.com/pyscf/pyscf)
- pyWannier90 for VASP can construct MLWFs using only the WAVECAR and vasprun.xml obtained by VASP.
That means you won't need to rerun the VASP calculation everytime you change some wannierization parameters with the wannier90 build-in library (pw2wannier90.x).
One should note that the MLWFs constructed by pyWannier90 may be not identical to that by VASP since only the pseudo wave function (WAVECAR) is used.
However, I have not experienced any significant discrepancy between the two approaches. 

## How to install libwannier90
libwannier90 needs to be installed first before one can call pyWannier90 from the code of choice (PySCF or MCU).
- Replace /wannier90-xxx/src/wannier_lib.F90 with /pyWannier90/src/wannier_lib.F90
- Go to /wannier90-xxx, modify make.inc with your favorite compiler and important adding this line to it:
	```
	FCOPTS = -O3 -fPIC -g
	```
- Compile wannier90-xxx:
	```
	make & make lib
	```
- Go to /pyWannier90/src, modify Makefile and compile libwannier90:
	```
	make
	```
- Modify the path of libwannier90 in pyWannier90.py (in /pyscf/pbc/tools or in /mcu/wannier90)
	
- Test libwannier90 library:
	```
	python -c "import libwannier90"		#Should return nothing if the compilation was successful
	```
	
## How to cite?
Please cite this paper when you use pyWannier90 code in your research:
- pyWannier90 for PySCF: Q. Sun et al.,Recent developments in the PySCF program package, [**J. Chem. Phys**](https://doi.org/10.1063/5.0006074), **2020**, Just Accepted
- pyWannier90 for VASP: H. Q. Pham, MCU: a Multipurpose Python Library to Analyze the Electronic Wave Function of Solid-State Materials, Manuscript under preparation, [MCU package](https://hungpham2017.github.io/mcu/)
