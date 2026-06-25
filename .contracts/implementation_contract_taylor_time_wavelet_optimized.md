# Implementation Contract: Dense-Stripe Wavelet Coaddition Alternative To `wavemaket`

## Intended Capability

Add a new numba-compiled alternative to `wavemaket` that coadds one waveform directly into a dense wavelet-domain stripe array in place.

The implementation only needs to support:

- `force_nulls=0` equivalent behavior.
- `amplitude_order=0`.
- Dense stripe accumulation.
- Linearly evolving intrinsic source frequency over the supported range.

It does not need full `wavemaket` feature parity.

## Motivation

The current `wavemaket` path uses a sparse-format wavelet-domain representation. Future higher-level parallel coaddition will operate on independent dense frequency stripes and later sum those coadded stripe results into a final wavelet-domain result.

The full wavelet transform matrix is too large to store many complete instances simultaneously, so each worker must operate on a narrow frequency range rather than the full frequency grid.

## Files And Public API

Required new implementation file:

- `WaveletWaveforms/taylor_time_wavelet_optimized.py`

Required new test file:

- `tests/test_taylor_time_wavelet_optimized.py`

Required speed benchmark/demo:

- `speed_tests/taylor_time_wavelet_optimized_speed_demo.py`

Required functions:

```python
wavemaket_stripe_sparse(
    wavelet_stripe: NDArray[np.floating],
    waveform: StationaryWaveformTime,
    nf_min: int,
    stripe_height: int,
    nt_lim_waveform: PixelGenericRange,
    wc: WDMWaveletConstants,
    taylor_table: WaveletTaylorTimeCoeffs,
)
```

```python
wavemaket_stripe_dense(
    wavelet_stripe: NDArray[np.floating],
    waveform: StationaryWaveformTime,
    nf_min: int,
    stripe_height: int,
    nt_lim_waveform: PixelGenericRange,
    wc: WDMWaveletConstants,
    taylor_table: WaveletTaylorTimeCoeffs,
)
```

If the fixed-`k` implementation requires a distorted or realigned interpolation-table representation, `taylor_table` may use a new table type for that implementation. Any such table is also treated as a compile-time constant.

Both core required functions are equally supported APIs. Neither is merely an internal benchmark variant.

## Dense Stripe Shape

`wavelet_stripe` has shape:

```text
(Nt, Nf, Nc)
```

where:

- `Nt` is the number of time pixels.
- `Nf` is the number of frequency pixels in the stripe.
- `Nc` is the number of TDI channels.

Shape and layout constraints:

- `Nt` is always even.
- Typical `Nt` is 2048.
- `Nf` is between 2 and 5 inclusive.
- `Nc` is nearly always 3.
- The stripe is a contiguous copy.
- Either C-contiguous or Fortran-contiguous layout may be used, guided by benchmark results.
- Shape may be treated as a compile-time constant, including recompilation for different shapes.

`nf_min` identifies the first frequency pixel represented by the stripe.

`stripe_height` is the number of frequency pixels in the stripe and must correspond to `wavelet_stripe.shape[1]`.

`nt_lim_waveform` gives the relevant time-pixel range for the waveform.

## Scientific And Mathematical Basis

The numerical oracle is the existing `wavemaket` function called with:

- `force_nulls=0`
- `amplitude_order=0`

The new dense-stripe functions must match this oracle for the supported restricted domain.

Supported waveform domain:

- Intrinsic source frequency evolves linearly over time.
- Total intrinsic frequency drift is no more than two frequency pixels over the observation period.
- Slope may be positive, negative, zero, or nearly zero.
- Initial source frequency is within `+/- 0.5` frequency pixel of the stripe center.
- Doppler shifting, polarization rotation, and amplitude modulation come from existing TDI functions.

TDI functions must not be modified and are not part of the performance benchmark.

## Numerical Tolerances

For implementations that do not distort the interpolation table:

- `atol = 1e-10 * amplitude_source`
- `rtol = 1e-9`

`amplitude_source` is the maximum value of the amplitude.

For benchmark-only experimental implementations that distort or realign the interpolation table:

- Looser behavior may be considered.
- Proposed threshold: approximately `rtol = 1e-4`.
- The benchmark and tests must clearly identify that this is a distorted-table experimental path, not the primary faithful tolerance regime unless explicitly accepted later.

## Implementation Variants

### `wavemaket_stripe_sparse`

This function must:

- Preserve the current variable-`k` inner-loop structure as closely as practical.
- Write into `wavelet_stripe` in place.
- Produce dense stripe output rather than sparse-format output.
- Match the existing `wavemaket(force_nulls=0, amplitude_order=0)` oracle within the faithful numerical tolerance.

### `wavemaket_stripe_dense`

This function must:

- Use a fixed-`k` inner loop.
- Add zero for `k` values that would otherwise be outside the valid variable range.
- Prioritize minimizing conditional branches in the hot loop.
- Include a literal zero-conditional hot-loop implementation if achievable.
- Write into `wavelet_stripe` in place.
- Match the existing `wavemaket(force_nulls=0, amplitude_order=0)` oracle within the faithful numerical tolerance unless an explicitly benchmark-only distorted-table variant is being evaluated separately.

If a branch-containing fixed-`k` version benchmarks faster than a literal zero-branch version, benchmark experiments may include both.

Additional benchmark-only variants are allowed if clearly named and isolated from the two required supported APIs.

## Distorted-Table Experiments

Distorted or realigned interpolation-table experiments are optional, not required.

They are not required if the implementation provides a fixed-`k` dense variant that:

- Meets the faithful numerical tolerance against `wavemaket(force_nulls=0, amplitude_order=0)`.
- Achieves comparable or superior speed in the benchmark.
- Has zero branching conditionals in the targeted hot inner loop.

If those conditions are not met, distorted-table experiments may be included as benchmark-only alternatives. Any distorted-table variant must be clearly separated from the two required supported APIs unless later promoted by explicit review decision.

## Data And Shape Expectations

The stripe frequency-pixel range must be represented by a `PixelGenericRange`.

Input sources are selected so their initial frequency is within `+/- 0.5` frequency pixel of the stripe center.

Required frequency-position coverage:

- Waveforms beginning in odd-index frequency pixels.
- Waveforms beginning in even-index frequency pixels.
- Frequencies exactly on pixel boundaries.
- Frequencies exactly halfway between pixel boundaries.
- Frequencies at arbitrary positions on either side of halfway between pixel boundaries.

## Boundary Cases

Required supported cases:

- `Nf` values from 2 through 5 inclusive.
- Even `Nt`, including representative `Nt = 2048` in benchmark or test coverage.
- Linear frequency slopes in `[-2, +2]` frequency pixels over the observation.
- Initial frequencies on boundaries.
- Initial frequencies halfway between boundaries.
- Initial frequencies below halfway.
- Initial frequencies above halfway.
- Initial frequencies in odd and even frequency-pixel index regions.

Unsupported behavior:

- Non-ignored null handling.
- Higher-order amplitude handling.
- Arbitrary nonlinear intrinsic frequency evolution.
- Performance guarantees outside the restricted linear-frequency domain.
- Any changes to existing TDI routines.

Validation behavior:

- Runtime input validation must use assertions only.
- Assertions may check shape, contiguity, supported stripe width, even time length, dtype, and range consistency.
- Production behavior with assertions disabled is not required to branch defensively for invalid inputs.

## Compatibility Constraints

No existing tracked Python, configuration, or coefficient files may be modified except strict additive edits to existing `__init__.py` files if required for imports.

No existing `wavemaket` behavior may change.

A reviewer must be able to verify that diffs are limited to:

- `WaveletWaveforms/taylor_time_wavelet_optimized.py`
- `tests/test_taylor_time_wavelet_optimized.py`
- `speed_tests/taylor_time_wavelet_optimized_speed_demo.py`
- Optional strictly additive `__init__.py` export edits.

## QA Constraints

Code must:

- Use numba `njit`.
- Include specific type annotations for all inputs required by repository style.
- Include concise NumPy-style docstrings.
- Prioritize correctness and precise input/output semantics over long explanations.
- Use assertions only for runtime validation.
- Avoid production defensive branches for invalid inputs in hot paths.

Tests must:

- Compare both required functions against existing `wavemaket(force_nulls=0, amplitude_order=0)`.
- Compare `wavemaket_stripe_sparse` and `wavemaket_stripe_dense` against each other.
- Verify in-place mutation of the input dense stripe.
- Verify the required slope, parity, boundary, half-pixel, arbitrary-position, and stripe-width cases.
- Avoid shape-only or placeholder-friendly assertions.
- Clearly distinguish the faithful tolerance regime from any optional distorted-table benchmark experiment.

Existing test files do not need to be modified.

Existing test files do not need to be run for this contract.

## Performance Requirements

Add:

- `speed_tests/taylor_time_wavelet_optimized_speed_demo.py`

The speed demo must compare:

- Existing `wavemaket`.
- `wavemaket_stripe_sparse`.
- `wavemaket_stripe_dense`.
- Any optional benchmark-only variants.

The benchmark must:

- Exclude TDI computation from timed sections.
- Include representative `(Nt, Nf, Nc)` shapes.
- Include `Nt = 2048`, `Nf = 2..5`, and `Nc = 3`.
- Compare C-contiguous and Fortran-contiguous layouts if both are plausible implementation paths.
- Document empirical results clearly enough to justify implementation choices.

No explicit parallelism should be used inside these functions. Higher-level parallel coaddition is future work.

## Known Risks

- Matching sparse `wavemaket` behavior against a dense stripe may be sensitive to indexing conventions.
- Frequency-pixel boundary and half-pixel cases are likely to expose off-by-one or rounding differences.
- Odd and even frequency-pixel starting indices may interact with wavelet coefficient parity assumptions.
- Fixed-`k` zero-contribution behavior could accidentally differ from variable valid-range behavior near stripe boundaries.
- Reworking interpolation-table layout could create numerically subtle differences.
- C-contiguous versus Fortran-contiguous layout may materially affect numba compilation and vectorization.
- Adding only new files may make normal package import/export inconvenient unless strict additive `__init__.py` edits are used.

## Explicitly Out Of Scope

- Full `wavemaket` compatibility.
- Null handling modes other than ignored/nulls disabled.
- Higher-order amplitude interpolation.
- TDI modification or optimization.
- Internal parallelism.
- General sparse-format replacement.
- Broad waveform models beyond the specified linear-frequency case.
- Existing coefficient file modifications.
- Existing test-file modifications.
- Required distorted-table experiments when the faithful zero-branch dense implementation satisfies accuracy and speed requirements.

## Decisions Already Made

- New implementation path: `WaveletWaveforms/taylor_time_wavelet_optimized.py`
- New test path: `tests/test_taylor_time_wavelet_optimized.py`
- New benchmark path: `speed_tests/taylor_time_wavelet_optimized_speed_demo.py`
- Required function names:
  - `wavemaket_stripe_sparse`
  - `wavemaket_stripe_dense`
- Dense stripe shape: `(Nt, Nf, Nc)`
- `Nt` is even.
- Typical `Nt` is 2048.
- `Nf` is between 2 and 5.
- `Nc` is nearly always 3.
- C or Fortran contiguity may be selected empirically.
- Faithful interpolation tolerance: `atol = 1e-10 * amplitude_source`, `rtol = 1e-9`.
- `amplitude_source` is the maximum value of the amplitude.
- Distorted-table variants may be considered around `rtol = 1e-4`.
- Distorted-table experiments are optional and benchmark-only unless explicitly promoted later.
- Existing `__init__.py` files may receive strict additive export edits if necessary.
- Existing `wavemaket` with `force_nulls=0` and `amplitude_order=0` is the oracle.
- Both core required functions are equally supported APIs.

## Reviewer Acceptance Checklist

An implementation satisfies this contract only if a reviewer can verify that:

- The required implementation, test, and speed-demo files exist at the specified paths.
- The required function names and signatures are present.
- Both required functions mutate the provided dense stripe in place.
- Both required functions are numba `njit` compiled.
- Tests compare against existing `wavemaket(force_nulls=0, amplitude_order=0)`.
- Tests cover the required frequency-slope, parity, boundary, half-pixel, arbitrary-position, and stripe-width cases.
- The faithful implementations meet `atol = 1e-10 * amplitude_source` and `rtol = 1e-9`.
- Any distorted-table variant is clearly benchmark-only unless separately approved.
- The speed demo excludes TDI computation from timed sections.
- The speed demo includes representative `Nt = 2048`, `Nf = 2..5`, and `Nc = 3` cases.
- Git diffs are limited to the allowed new files and optional strict additive `__init__.py` edits.
