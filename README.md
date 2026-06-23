# GalacticStochastic

GalacticStochastic simulates realizations of the **galactic stochastic gravitational-wave
background** as seen by the [LISA](https://www.lisamission.org/) observatory, and runs an
iterative fitting procedure to separate resolvable binaries from the unresolved confusion
foreground.

It is an improved, modular reimplementation of the code described in
[*LISA Gravitational Wave Sources in A Time-Varying Galactic Stochastic Background*](https://arxiv.org/abs/2206.14813)
(arXiv:2206.14813), designed to make it easy to re-run with different realizations of the
galaxy, different instrument-noise models, or a modified fitting procedure.

The pipeline takes a **galaxy** (an HDF5 catalog of binaries) plus a **`.toml` configuration
file**, and produces compressed HDF5 output. The fit state can be reloaded from that output,
unless options are chosen to shrink the output file.

---

## How the iterative fit works

Because a galaxy typically contains a very large number of binaries that individually have no
chance of detection but collectively must be coadded into the foreground, the **first
iteration is by far the most expensive**. The code therefore treats it specially:

- **Preliminary iteration** — a single, conservative first pass that only removes binaries
  virtually guaranteed to be undetectable. It considers instrument noise only and can use a
  lower SNR cutoff and a longer observation window than later iterations. Its result is saved
  to disk so it can be reused. *Example:* run the preliminary pass at SNR > 5 over 10 years
  while later iterations use SNR > 7 over 4 years — there is then essentially zero chance of
  incorrectly suppressing a detectable binary. If the cutoff turns out to be *too*
  conservative, a second preliminary pass can be run, reusing the first to go faster.
- **Main iterations** — once a satisfactory preliminary result exists, the main loop can be
  re-run many times with different fitting parameters or time windows, reusing the saved
  preliminary background each time.

---

## Installation

Requires Python ≥ 3.12. From the repository root:

```bash
pip install .
```

Optional extras:

```bash
pip install ".[dev]"        # pytest + type stubs for the test/type-check suite
pip install ".[plots]"      # matplotlib, for the make_gb_*_compare_plot.py scripts
pip install ".[cosmic]"     # pandas + astropy, for loading COSMIC population catalogs
pip install ".[imrphenomd]" # PyIMRPhenomD (not on PyPI; install from source first)
```

---

## Running the pipeline

### Command line

After installation, the full procedure is available as a console script:

```bash
run-galactic-stochastic-iterative GALAXY_DIR CONFIG_FILE
```

- `GALAXY_DIR` — directory containing the galaxy HDF5 file.
- `CONFIG_FILE` — path to a `.toml` configuration file.

Optional flags map directly onto `fetch_or_run_iterative_loop` (see the modes below):
`--cyclo-mode` (default `stationary`), `--fetch-mode` (default `run_all`),
`--output-mode` (default `store_always`), `--preprocess-mode` (default `final`).

> **Tip:** copy a dedicated config file into each galaxy's directory so the exact run can be
> reproduced later. Use a **separate directory per galaxy** — otherwise outputs can overwrite
> one another.

### From Python

The pipeline is driven primarily by one function,
[`fetch_or_run_iterative_loop`](GalacticStochastic/iterative_fit.py), imported from
`GalacticStochastic.iterative_fit`:

```python
from GalacticStochastic.config_helper import get_config_objects
from GalacticStochastic.iterative_fit import fetch_or_run_iterative_loop

config, wc, lc, ic, seed = get_config_objects("default_parameters.toml")
config["files"]["galaxy_dir"] = "Galaxies/Galaxy1/"

ifm = fetch_or_run_iterative_loop(
    config,
    cyclo_mode="stationary",
    fetch_mode="fetch_or_run_all",   # reuse results on disk if present, else compute
    output_mode="store_if_new",
    preprocess_mode="final",
)
```

The returned `IterativeFitManager` holds the converged fit state, noise model, and binary
inclusion state. See [`run_gb_iterative.py`](run_gb_iterative.py) for a worked example
including plotting.

#### Modes

`fetch_or_run_iterative_loop` accepts both integer and string forms for each mode.

| `cyclo_mode` | meaning |
| --- | --- |
| `0` / `cyclostationary` | fit a cyclostationary (time-varying) foreground spectrum |
| `1` / `stationary` | fit a stationary foreground spectrum |

| `fetch_mode` | meaning |
| --- | --- |
| `0` / `run_reprocess_only` | run from scratch; stop if the preliminary file is missing |
| `1` / `run_or_fetch_reprocess_only` | fetch if available, else run; stop if preliminary missing |
| `2` / `fetch_or_fail_reprocess_only` | fetch if available, else abort; stop if preliminary missing |
| `3` / `fetch_or_run_all` | fetch if available, else run; create the preliminary file if missing |
| `4` / `run_all` | run from scratch; do not look for a preliminary file |

| `output_mode` | meaning |
| --- | --- |
| `0` / `no_store` | do not write output |
| `1` / `store_if_new` | store results unless they were fetched from disk |
| `2` / `store_always` | store results even when fetched |

| `preprocess_mode` | meaning |
| --- | --- |
| `0` / `final` | run the main iterations (requires a preliminary background) |
| `1` / `initial` | run the preliminary iteration only |
| `2` / `repeat_initial` | re-process an existing preliminary result |

---

## Configuration files

Configuration is a TOML file (see [`default_parameters.toml`](default_parameters.toml)) with
sections for file I/O, the noise realization, iterative-fit constants, wavelet-transform
constants, and LISA-instrument constants. File names default to:

```toml
[files]
galaxy_file         = 'galaxy_binaries.hdf5'
processed_prefix    = 'processed_iterations'     # final (main-loop) output
preprocessed_prefix = 'preprocessed_background'  # preliminary-iteration dump
```

The preliminary dump (`preprocessed_prefix`) stores the fit state after the first iteration;
`processed_prefix` stores the results of the remaining iterations. How much detail is written
is controlled by storage-mode options in the config. As a rough guide, for ≳100 million
binaries the preliminary file approaches ~half the size of the input galaxy file, while the
final processed files are smaller (exact sizes depend on the sampling rate and assumed LISA
mission length).

---

## Repository layout

| Path | Description |
| --- | --- |
| `GalacticStochastic/` | The pipeline: iterative fit, noise modeling, background decomposition, config, and HDF5 I/O. |
| `GalacticStochastic/iterative_fit.py` | `fetch_or_run_iterative_loop` — the top-level entry point. |
| `GalacticStochastic/iterative_fit_manager.py` | `IterativeFitManager` — orchestrates the loop over component state managers. |
| `GalacticStochastic/cli/` | Console-script entry point (`run-galactic-stochastic-iterative`). |
| `LisaWaveformTools/` | LISA waveform & response models, instrument noise, source parameters. |
| `WaveletWaveforms/` | WDM wavelet-transform tooling (sparse wavelet waveforms, Taylor coefficients). |
| `Galaxies/` | Example galaxy catalogs and their per-galaxy `.toml` configs. |
| `tests/` | Pytest suite and accompanying test configs. |

### Architecture

`fetch_or_run_iterative_loop` builds configuration objects from the TOML
(`WDMWaveletConstants`, `LISAConstants`, `IterationConfig`), loads the galaxy parameters, and
— when running the main loop — first recurses with `preprocess_mode='initial'` to obtain the
preliminary background. It then assembles an `IterativeFitManager` from three component
**state managers**, each implementing the `StateManager` interface
(`advance_state` / `log_state` / `state_check` / `loop_finalize` / `print_report` /
`store_hdf5` / `load_hdf5`):

- `IterativeFitState` — convergence state machine (bright/faint convergence decisions),
- `NoiseModelManager` — stationary / cyclostationary noise model,
- `BinaryInclusionState` — tracks which binaries belong to each background component.

`IterativeFitManager.do_loop()` advances all three each iteration until both bright and faint
components converge (or `max_iterations` is reached), then finalizes and reports.

---

## Development

```bash
pip install ".[dev]"
pytest                 # run the test suite
```

Continuous integration runs build, lint (ruff/pylint), type-checking, and docstring
(pydoclint) checks; see `.github/workflows/`. Pre-commit hooks are configured in
`.pre-commit-config.yaml`.
