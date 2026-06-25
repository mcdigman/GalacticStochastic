# Implementation Contract v4: Dense-Stripe Taylor-Time Wavelet Coaddition

## Status And Scope

This is a revised implementation contract for adding numba-compiled dense-stripe alternatives to `wavemaket`.

This contract is not an implementation approval. It resolves the review findings in `adversarial_review_taylor_time_wavelet_optimized.md`, `adversarial_review_taylor_time_wavelet_optimized_v2.md`, and `adversarial_review_taylor_time_wavelet_optimized_v3.md` using:

- The original contract in `implementation_contract_taylor_time_wavelet_optimized.md`.
- Existing repository behavior of `wavemaket`, `wavelet_sparse_to_dense`, `SparseWaveletWaveform`, `PixelGenericRange`, `StationaryWaveformTime`, `WaveletTaylorTimeCoeffs`, and `WDMWaveletConstants`.
- The adversarial re-review in `adversarial_review_taylor_time_wavelet_optimized_v2.md`.
- The adversarial re-review in `adversarial_review_taylor_time_wavelet_optimized_v3.md`.
- The human decisions supplied for v2, v3, and v4.

The implementation must add direct dense-stripe coaddition for one waveform over a narrow frequency stripe. It only needs to support:

- `force_nulls=0` equivalent behavior.
- `amplitude_order=0`.
- Dense stripe accumulation into an existing output array.
- Linearly evolving intrinsic source frequency over the supported test domain.

The implementation does not need full `wavemaket` feature parity.

## Motivation

The current `wavemaket` path produces a sparse-format wavelet-domain representation over the full wavelet band. Future higher-level parallel coaddition will operate on independent dense frequency stripes and later sum those stripe results into a final wavelet-domain result.

The full wavelet transform matrix is too large to store many complete instances simultaneously, so each worker must operate on a narrow frequency range rather than materializing full-band intermediate results.

## Required Files

Required new implementation file:

- `WaveletWaveforms/taylor_time_wavelet_optimized.py`

Required new test file:

- `tests/test_taylor_time_wavelet_optimized.py`

Required speed benchmark/demo:

- `speed_tests/taylor_time_wavelet_optimized_speed_demo.py`

Optional strictly additive import/export edits are allowed only in existing `__init__.py` files if required for imports. No other existing tracked Python, configuration, coefficient, test, workflow, or documentation file may be modified by this implementation contract.

## Public API

The implementation must provide these three supported functions:

```python
wavemaket_stripe_sparse(
    wavelet_stripe: NDArray[np.floating],
    waveform: StationaryWaveformTime,
    nf_start: int,
    stripe_height: int,
    nt_lim_waveform: PixelGenericRange,
    wc: WDMWaveletConstants,
    taylor_table: WaveletTaylorTimeCoeffs,
) -> None
```

```python
wavemaket_stripe_dense(
    wavelet_stripe: NDArray[np.floating],
    waveform: StationaryWaveformTime,
    nf_start: int,
    stripe_height: int,
    nt_lim_waveform: PixelGenericRange,
    wc: WDMWaveletConstants,
    taylor_table: WaveletTaylorTimeCoeffs,
) -> None
```

```python
wavemaket_stripe_dense_aligned(
    wavelet_stripe: NDArray[np.floating],
    waveform: StationaryWaveformTime,
    nf_start: int,
    stripe_height: int,
    nt_lim_waveform: PixelGenericRange,
    wc: WDMWaveletConstants,
    taylor_table_aligned: WaveletTaylorTimeCoeffsAligned,
) -> None
```

All three functions are supported APIs, not throwaway benchmark-only variants. The name `nf_start` replaces the original contract's `nf_min` parameter to avoid collision with `wavemaket`'s local full-band lower-edge variable.

The implementation must define `WaveletTaylorTimeCoeffsAligned` as a numba-compatible table type. It may be a NamedTuple, dataclass-like numba-compatible object, or other numba-compatible representation, but it must implement the zero-padded, integer-aligned representation defined in "Aligned Table Requirements." The implementation may add a helper to construct this table from `WaveletTaylorTimeCoeffs`; if the helper is public, it must have narrow annotations and a NumPy-style docstring.

## Dense Stripe Shape And Indexing

`wavelet_stripe` has shape:

```text
(Nt, stripe_height, Nc)
```

where:

- `Nt == wc.Nt`.
- `stripe_height == wavelet_stripe.shape[1]`.
- `stripe_height` is in `{2, 3, 4, 5}`.
- `Nc == wavelet_stripe.shape[2] == waveform.AT.shape[0]`.
- `wavelet_stripe.dtype` is a NumPy floating dtype.
- `wavelet_stripe` is contiguous, either C-contiguous or Fortran-contiguous.

`wc.Nf` is the full number of wavelet frequency layers, for example 256 in existing test configurations. It is distinct from `stripe_height`. The stripe is a contiguous global frequency-layer window:

```text
[nf_start, nf_start + stripe_height)
```

inside the full band:

```text
0 <= nf_start
nf_start + stripe_height <= wc.Nf
```

The time axis uses absolute `wavemaket` time-pixel coordinates. A contribution for full-band time pixel `j` and global frequency layer `k` maps to:

```text
wavelet_stripe[j, k - nf_start, c]
```

No function may write outside columns `[0, stripe_height)`. Contributions to global frequency layers outside `[nf_start, nf_start + stripe_height)` are dropped, not wrapped, clamped, or accumulated into edge columns.

The implementation must use `wc.Nf`, not `stripe_height`, wherever full-band wavelet math requires the full frequency-layer count, including pixel-index encoding, layer center calculations, and oracle parity expressions that depend on `j + k`.

Shape may be treated as a compile-time constant, including numba recompilation for different shapes and layouts.

## Runtime Validation

Runtime input validation must use assertions only. With assertions disabled, invalid-input behavior is undefined and is not required to branch defensively.

Each public function must assert at least:

- `wavelet_stripe.ndim == 3`.
- `wavelet_stripe.shape[0] == wc.Nt`.
- `wavelet_stripe.shape[1] == stripe_height`.
- `wavelet_stripe.shape[2] == waveform.AT.shape[0]`.
- `stripe_height in (2, 3, 4, 5)`.
- `wc.Nt % 2 == 0`.
- `0 <= nf_start`.
- `nf_start + stripe_height <= wc.Nf`.
- `wavelet_stripe` is C-contiguous or F-contiguous.
- `nt_lim_waveform.nx_min >= 0`.
- `nt_lim_waveform.nx_max <= wc.Nt`.
- `nt_lim_waveform.nx_min <= nt_lim_waveform.nx_max`.

`wavemaket_stripe_dense_aligned` must additionally assert:

- `2 * wc.Nsf % 3 == 0`.
- With `R = 2 * wc.Nsf // 3`, `abs((wc.DF / wc.df_bw) - R) <= 1e-12 * max(1.0, abs(R))`.
- The aligned table's stored `R`, valid-region metadata, row count, and channel-independent coefficient-array dimensions are consistent with `wc`, `taylor_table_aligned`, and the supported stripe-height domain.

Assertions may include messages, but message wording is not part of the contract.

The numba-compiled function bodies are not required to assert the exact dtype of `wavelet_stripe`, and tests must not require a non-float64 floating dtype to fail solely because of dtype. Float64 is the required dtype for faithful numerical acceptance tests unless an additional tolerance for another floating dtype is explicitly authorized later.

Runtime validation assertions must be evaluated before the per-pixel hot loop or at a per-call setup boundary. Numerical tolerance or closeness assertions against the oracle must not appear inside the required numba-compiled functions; those comparisons belong only in tests and benchmark/demo correctness checks.

## Scientific And Mathematical Basis

The numerical oracle is the existing `wavemaket` function called with:

- `force_nulls=0`
- `amplitude_order=0`

The oracle must be run with the same `waveform`, `nt_lim_waveform`, `wc`, and original `WaveletTaylorTimeCoeffs` table. The oracle is converted from sparse to dense by:

1. Running `wavemaket` into a `SparseWaveletWaveform`.
2. Calling `wavelet_sparse_to_dense(oracle_sparse, wc)`, producing shape `(wc.Nt, wc.Nf, Nc)`.
3. Taking `oracle_dense[:, nf_start:nf_start + stripe_height, :]`.

All correctness comparisons must cover the entire stripe slice, including cells expected to be exactly zero. Comparisons restricted to `oracle != 0` or any other nonzero mask are prohibited.

All three new functions must reproduce the oracle's `force_nulls=0`, `amplitude_order=0` drop behavior in the stripe. For each time pixel `j`, global frequency layer `k`, and channel `c`, the stripe cell may receive a nonzero contribution only when `wavemaket` would write `(j, k, c)` under all of these conditions:

- The frequency-derivative table index condition holds: `0 <= ny + wc.Nfd_negative < wc.Nfd - 1`, where `ny = floor(waveform.FTd[c, j] / wc.dfd)`.
- `k` is inside the oracle bandwidth interval for that time sample and channel: `ceil((fa - half_bandwidth) / wc.DF) <= k <= floor((fa + half_bandwidth) / wc.DF)`, where `fa = waveform.FT[c, j]` and `half_bandwidth` is computed from the two adjacent `taylor_table.Nfsam` values exactly as in `wavemaket`.
- `k` is inside the stripe window `[nf_start, nf_start + stripe_height)`.
- The interpolation-table overflow guard holds: `0 <= jj1 < Nfsam1_loc - 1` and `0 <= jj2 < Nfsam2_loc - 1`, using the same `jj1` and `jj2` definitions as `wavemaket`.

Where any of these conditions fail, the new functions must contribute `0.0` to the stripe cell.

The required functions must compute `wavelet_stripe[...] += contribution`; they must not overwrite existing contents. Pre-existing values in `wavelet_stripe` must be preserved additively.

## Supported Waveform Domain

Supported waveform inputs are selected so that:

- Intrinsic source frequency evolves linearly over time.
- Total intrinsic frequency drift is no more than two full-band frequency pixels over the observation period.
- Slope may be positive, negative, zero, nearly zero, or very nearly zero.
- Initial source frequency is within `+/- 0.5` full-band frequency pixel of the stripe center.
- Doppler shifting, polarization rotation, and amplitude modulation come from existing TDI functions.

TDI functions must not be modified and are not part of the performance benchmark.

Unsupported behavior:

- Non-ignored null handling other than `force_nulls=0` equivalence.
- Higher-order amplitude handling.
- Arbitrary nonlinear intrinsic frequency evolution.
- Performance guarantees outside the restricted linear-frequency domain.
- Any changes to existing TDI routines.

## Numerical Tolerances

For float64 `wavelet_stripe` inputs, all three required functions must match the oracle stripe slice with:

```text
atol = 1e-10 * amplitude_source
rtol = 1e-9
```

where:

```python
amplitude_source = float(np.max(np.abs(waveform.AT[:, nt_lim_waveform.nx_min:nt_lim_waveform.nx_max])))
```

Numerical-correctness cases must use a non-empty selected time range and `amplitude_source > 0`.

The previous v1 allowance for a distorted or realigned table with looser tolerance is removed. No required or optional path in this contract may use a looser tolerance for float64 correctness acceptance.

For other NumPy floating dtypes, this contract requires that the public functions do not reject the input solely because of dtype. Exact numerical tolerances for non-float64 output are unresolved and are not part of this contract's required acceptance tests.

## Implementation Variants

### `wavemaket_stripe_sparse`

This function must:

- Preserve the current variable-`k` inner-loop structure as closely as practical while writing directly to `wavelet_stripe`.
- Use the full-band `wc.Nf` for full-band wavelet calculations.
- Restrict writes to global layers in `[nf_start, nf_start + stripe_height)`.
- Accumulate into `wavelet_stripe` in place.
- Match the oracle stripe slice within the faithful float64 numerical tolerance.

### `wavemaket_stripe_dense`

This function must:

- Use a fixed-`k` inner loop over the stripe window.
- Add zero for `k` values that would fail any oracle drop guard.
- Use the original `WaveletTaylorTimeCoeffs` representation.
- Prioritize minimizing conditional branches in the hot loop without changing observable behavior.
- Accumulate into `wavelet_stripe` in place.
- Match the oracle stripe slice within the faithful float64 numerical tolerance.

### `wavemaket_stripe_dense_aligned`

This function must:

- Use a fixed-`k` inner loop over the stripe window.
- Use the zero-padded, integer-aligned table representation defined in this contract.
- Assert the integer-alignment input preconditions before using the aligned-table path.
- Add zero for `k` values that would fail any oracle drop guard. Bandwidth and table-overflow drops must be reproduced exactly by per-time-pixel loop bounds, per-regime loop bounds, a branchless `0.0`/`1.0` validity factor derived from the oracle guards, or an observably equivalent mechanism.
- Prioritize minimizing conditional branches in the hot loop without changing observable behavior.
- Accumulate into `wavelet_stripe` in place.
- Match the oracle stripe slice within the faithful float64 numerical tolerance.

This third function is required by human decision. It is not an experimental lower-accuracy distorted-table path.

## Aligned Table Requirements

The aligned-table path is supported only when the original input configuration is already integer-aligned. It must use the original coefficient grid and must not numerically alter the original `WaveletTaylorTimeCoeffs` coefficient values.

The required input precondition is:

```text
2 * wc.Nsf % 3 == 0
R = 2 * wc.Nsf // 3
abs((wc.DF / wc.df_bw) - R) <= 1e-12 * max(1.0, abs(R))
```

When this precondition holds, the original `df_bw` grid is already aligned with full-band frequency-layer spacing. The useful aligned-path property is not a new coefficient grid; it is that the fixed-`k` loop can use integer stride `R` and compute the interpolation floor and `dx` values at a per-time-pixel or per-regime boundary instead of recomputing those quantities independently for every `k`.

The aligned table representation used by `wavemaket_stripe_dense_aligned` must differ structurally from a plain identity wrapper around `WaveletTaylorTimeCoeffs`. The structural difference is layout and metadata, not coefficient values.

The aligned representation must:

- Reuse coefficient values from the original `WaveletTaylorTimeCoeffs` on the original `wc.df_bw` grid.
- Preserve `Nfd`, `Nfd_negative`, `Nfsam`, `wc.df_bw`, `wc.DF`, `wc.Nf`, `wc.Nt`, `wc.dfd`, and the original frequency-derivative grid.
- Store `R` or equivalent metadata proving the integer stride used by the aligned path.
- Provide readable coefficient rows with zero padding on both sides of each row's original valid coefficient range.
- Store explicit valid-region metadata or an equivalent observable convention that distinguishes original valid coefficients from left and right padding.
- Make the padded readable row width large enough that all table indices generated by `wavemaket_stripe_dense_aligned` for the supported stripe heights and supported waveform domain are in bounds.
- Support constant-stride reads by `R` in the aligned fixed-`k` loop.
- Match the oracle stripe slice within the faithful float64 tolerance.

The aligned representation may reuse the original coefficient values exactly, including bitwise-equal values, after moving them into a different padded layout. It may also use copied arrays whose valid-region values are exactly equal to the originals. An identity wrapper that merely exposes the original table without the required input-alignment metadata, two-sided padding convention, valid-region metadata, and constant-stride layout support is noncompliant.

The aligned implementation must not use zero padding as the sole mechanism for reproducing bandwidth or table-overflow drops at valid-range edges. Edge coefficients can be non-negligible, and the oracle drops those cells exactly. Bandwidth and overflow drops must therefore come from one of:

- Per-time-pixel loop bounds that admit only `k` values passing all oracle guards.
- Per-regime loop bounds, for example below and above the reflection center `k_star = fa / wc.DF`, that admit only `k` values passing all oracle guards.
- A branchless `0.0`/`1.0` validity factor derived from the same bandwidth and `jj1`/`jj2` conditions as `wavemaket`.
- An observably equivalent mechanism that gives exactly zero contribution for cells the oracle drops.

Zero padding may be relied on to keep aligned-table reads in bounds and to return `0.0` only for reads whose contribution has already been excluded by loop bounds or validity factors, or for cases where the dropped contribution is proven within `atol` of zero. It must not replace the oracle's table-overflow guard at the lower or upper valid-range edge.

Required verification for the aligned table:

- A structural test must verify the aligned-path precondition: `2 * wc.Nsf % 3 == 0`, `R == 2 * wc.Nsf // 3`, and `abs((wc.DF / wc.df_bw) - R) <= 1e-12 * max(1.0, abs(R))`.
- A structural test must verify that aligned tests and benchmarks use a configuration satisfying the precondition; the repository `Nsf = 150` configuration qualifies with `R = 100`.
- A structural test must verify that the aligned representation exposes two-sided padding and valid-region metadata or an equivalent observable convention.
- A structural test must verify that an identity wrapper around `WaveletTaylorTimeCoeffs` without aligned metadata and two-sided padding fails the aligned-table structural tests.
- A padding test must verify that representative reads in left and right padding return `0.0`.
- A drop-behavior test must verify that table-overflow edge cells dropped by the oracle remain zero in `wavemaket_stripe_dense_aligned`; the test must not pass merely because padded reads blend nonzero edge coefficients with zero padding.
- A differential correctness test must compare `wavemaket_stripe_dense_aligned` against the oracle over the full stripe slice for the same float64 cases required of the other two functions.

## Forbidden Shortcuts And Memory Constraints

None of the required functions may:

- Call `wavemaket`.
- Call `wavemaket_direct`.
- Allocate or materialize a full dense `(wc.Nt, wc.Nf, Nc)` array.
- Allocate or materialize a full-band `SparseWaveletWaveform`.
- Allocate any per-call intermediate whose frequency extent is `O(wc.Nf)`.
- Use a wrapper over the sparse oracle as the implementation strategy.
- Hide full-band materialization behind helper functions, aliases, closures, dynamically imported functions, generated-code labels, or equivalent indirection.

Per-call scratch storage in the frequency dimension is limited to `O(stripe_height)`.

Tests and benchmark code may construct the full-band oracle for verification and timing comparison. This exception applies only outside the three required implementation functions and their private helpers.

## Required Coverage Cases

Correctness tests must compare each of the three required functions against the full-stripe oracle for:

- `stripe_height` values `2`, `3`, `4`, and `5`.
- At least one even `nf_start`.
- At least one odd `nf_start`.
- At least one stripe whose top edge is near, but not beyond, `wc.Nf`.
- Even `wc.Nt`, including a representative `wc.Nt == 2048` case in either correctness tests or benchmark/demo setup.
- `Nc == 3`.
- Initial-frequency offsets from the stripe center, in full-band frequency-pixel units: `-0.5`, `-0.3`, `-0.25`, `0.0`, `0.25`, `0.3`, and `0.5`.
- Linear slopes in full-band frequency pixels per observation period: `-2.0`, `-1.99`, `-1.0`, `-1e-3`, `-1e-10`, `0.0`, `1e-10`, `1e-3`, `1.0`, `1.99`, and `2.0`.
- Frequencies exactly on pixel boundaries.
- Frequencies exactly halfway between pixel boundaries.
- Frequencies at arbitrary positions on each side of halfway, represented by offsets `-0.3`, `-0.25`, `0.25`, and `0.3`.

The required tests may use parametrization rather than the full Cartesian product if the selected cases still exercise each listed requirement for each function and each `stripe_height`. At least one test must exercise the bandwidth drop and table-overflow drop. The derivative-index drop may be verified either by a targeted out-of-supported-domain input whose required behavior is zero contribution for affected time pixels, matching the oracle, or by source inspection confirming that the `0 <= n_ind < wc.Nfd - 1` guard is present in all three required functions.

## Test Requirements

Tests must:

- Compare all three required functions against the oracle stripe slice over the entire stripe.
- Compare `wavemaket_stripe_sparse`, `wavemaket_stripe_dense`, and `wavemaket_stripe_dense_aligned` against each other after independent calls.
- Verify in-place mutation by checking that the same array object and data pointer are retained.
- Verify additive coaddition by pre-filling `wavelet_stripe` with known nonzero values `B`, calling each function, and asserting the result equals `B + oracle_slice`, not merely `oracle_slice`.
- Verify that cells dropped by the oracle remain unchanged except for the pre-existing `B` value in additive tests.
- Verify required assertions for invalid shape, invalid `stripe_height`, invalid `nf_start`, mismatched `Nc`, and invalid time range.
- Verify the additional aligned-path assertions for a configuration that does not satisfy `2 * wc.Nsf % 3 == 0` or the `wc.DF / wc.df_bw` ratio requirement.
- Avoid shape-only, nonzero-only, smoke-only, or placeholder-friendly assertions.
- Use an independent oracle construction based on existing `wavemaket` plus `wavelet_sparse_to_dense`, not on the new implementation's helper logic.
- Keep all new tests active. No `@pytest.mark.skip` or `xfail` may be added to the new test file unless specifically authorized outside this contract.

Because repository-wide tests are known to include slow or currently failing cases, this contract does not require the implementation agent to run the full test suite or add CI execution. The enforceable repository-scope QA requirement for this implementation is the allowed-diff/file-scope restriction plus unskipped targeted tests in the new test file.

## Benchmark/Demo Requirements

The speed benchmark/demo is required and must be committed at:

- `speed_tests/taylor_time_wavelet_optimized_speed_demo.py`

It must:

- Benchmark `wavemaket`, `wavemaket_stripe_sparse`, `wavemaket_stripe_dense`, and `wavemaket_stripe_dense_aligned`.
- Exclude TDI waveform construction and other source-generation work from timed sections.
- Warm up numba compilation before recording timings.
- Report at least median per-call wall time.
- Include `wc.Nt == 2048`, `stripe_height` values `2`, `3`, `4`, and `5`, and `Nc == 3`.
- Include both C-contiguous and F-contiguous stripe layouts, or explicitly report that one layout is unsupported because of a measured correctness or compilation failure.
- Verify within the benchmark script that benchmarked float64 stripe outputs match the oracle tolerance before reporting speed.

The benchmark has no required speed threshold in this contract. It is a required evidence-producing artifact, not an acceptance gate for performance. This preserves the original motivation while avoiding an invented performance target.

## QA And Repository Policy

Implementation code must:

- Use numba `njit` for the three required functions.
- Stay in nopython-compatible code for the required functions.
- Include specific type annotations for public function inputs and return `-> None`.
- Include concise NumPy-style docstrings for public functions and public helpers.
- Use assertions only for runtime validation.
- Avoid production defensive branches for invalid inputs in hot paths.
- Preserve existing `wavemaket` behavior.

Unless separately approved, the implementation may not introduce or modify:

- Inline linter suppressions such as `# noqa`.
- Type-checker suppressions such as `# type: ignore`, `# pyright: ignore`, or casts that hide broad types.
- Checker configuration.
- File exclusions.
- Test skips or expected failures.
- Warning filters.
- Coverage exclusions such as `# pragma: no cover`.
- CI, pre-commit, ruff, pyrefly, mypy, pytest, or coverage configuration.
- Broad `Any`-typed public interfaces, broad protocols, dynamic attribute access, wrapper aliases, or generated-code designations that obscure the contract.

If a future approval permits one of these changes, the implementation must disclose it explicitly and explain why the contract could not be met without it.

The required `speed_tests` file may inherit existing repository per-file lint policy for `speed_tests/*`, but this does not relax the public API annotations in the implementation file.

## Verification Methods

The implementation must be reviewable by:

- `git diff --name-only`, confirming diffs are limited to the required new files and optional strictly additive `__init__.py` edits.
- Source inspection, confirming the required functions do not call `wavemaket`, call `wavemaket_direct`, or allocate full-band intermediates.
- Test inspection, confirming the target tests are active, full-stripe, oracle-based, and include the required coverage cases.
- Benchmark inspection or execution, confirming it includes all required variants and validates correctness before timing.

Running the full repository test suite is explicitly outside the required acceptance procedure for this contract revision.

## Finding Disposition Ledger

| Finding | Disposition | Contract section affected | Exact revision made | Reasoning or authority | Remaining uncertainty |
|---|---|---|---|---|---|
| B1 | Accepted and resolved | Dense Stripe Shape And Indexing | Replaced `(Nt, Nf, Nc)` with `(Nt, stripe_height, Nc)` and separated `stripe_height` from full-band `wc.Nf`; required `Nt == wc.Nt`. | Human decision 2 and codebase fact that `wc.Nf` is full band. | None for naming and shape. |
| B2 | Accepted and resolved | Public API; Dense Stripe Shape And Indexing; Runtime Validation | Renamed `nf_min` to `nf_start`; specified `k -> k - nf_start`; required dropping outside stripe and assertions for bounds. | Human decision 2; review A2. | None. |
| B3 | Accepted and resolved | Scientific And Mathematical Basis; Test Requirements | Enumerated derivative-index, bandwidth, stripe-window, and table-overflow guards; required full-stripe oracle comparison. | Human decision 3 agreeing with A3 and A6. | Exact test data that reaches each guard is implementation responsibility. |
| B4 | Accepted and resolved | Forbidden Shortcuts And Memory Constraints | Forbid calls to `wavemaket`/`wavemaket_direct` and full-band dense/sparse intermediates inside required functions. | Human decision 3 agreeing with A4; original memory motivation. | Source inspection required. |
| R1 | Accepted and resolved | Public API; QA And Repository Policy | Added `-> None`; made all three APIs explicit. | Repository annotation policy and human decision for third version. | Exact aligned table internal layout may vary. |
| R2 | Accepted and resolved | Numerical Tolerances; Scientific And Mathematical Basis | Defined oracle construction, full-stripe comparison, and `amplitude_source`. | Review matrix and A6. | None. |
| R3 | Accepted and resolved | Test Requirements | Required pairwise comparison among all three variants after independent calls. | Original QA requirement extended to third function by human decision. | Pairwise comparison remains secondary to oracle. |
| R4 | Accepted and resolved | Test Requirements | Required object and data-pointer in-place verification. | Original in-place requirement. | None. |
| R5 / MC3 | Accepted and resolved | Scientific And Mathematical Basis; Test Requirements | Defined coadd as `+=`; required pre-filled-stripe test. | Review A5; original "coadds" intent. | None. |
| R6 / MC6 | Accepted and resolved | Required Coverage Cases; Test Requirements | Converted vague arbitrary positions to concrete offsets and required oracle assertion for listed cases. | Human decision agreeing with A9. | Parametrization may select a reduced non-Cartesian matrix. |
| R7 | Accepted and resolved | Required Coverage Cases | Enumerated slopes including endpoints, near endpoints, small slopes, zero, and `+/-1e-10`. | Human decision adding smaller nonzero slope. | Scientific validity outside oracle remains out of scope. |
| R8 | Accepted and resolved | Dense Stripe Shape And Indexing; Required Coverage Cases | Replaced stripe `Nf` wording with `stripe_height`; retained 2 through 5 as stripe heights. | Human decision 2. | None. |
| R9 / Type gaps 3-4, 9 | Accepted and resolved | Runtime Validation | Changed optional assertions to required assertions; stated undefined behavior under stripped assertions. | Review type/interface gaps. | No defensive behavior under `-O`. |
| R10 / MC7 / A7 | Accepted and partially resolved | Implementation Variants; Benchmark/Demo Requirements | Kept branch-minimization as implementation guidance, prohibited adding conditionals that change behavior, and required benchmark evidence. Did not add objective speed threshold. | Human decision: for A7 none of the proposed labels/comments/threshold criteria may be added. | Performance acceptance remains a human review matter after benchmark evidence. |
| R11 | Accepted and partially resolved | Benchmark/Demo Requirements | Required committed benchmark/demo and correctness validation before timing, but no pass/fail speed threshold. | Human decision 1 requires benchmark test; human decision rejects A7-style acceptance labels. | Actual speed target unresolved by authority. |
| R12 | Accepted and resolved | Required Files; QA And Repository Policy; Verification Methods | Reaffirmed allowed-diff restriction and no config edits. | Original compatibility constraint and user CI decision. | None. |
| Type gap 1 | Accepted and resolved in v2, superseded in v3 | Dense Stripe Shape And Indexing; Runtime Validation; Numerical Tolerances | v2 required `np.float64`; v3 changes the public type to `NDArray[np.floating]`, removes dtype assertion and invalid-dtype tests, and limits faithful tolerance acceptance to float64 cases. | Human v3 decision that float64 is not required and float32 is acceptable as a floating dtype. | Exact non-float64 numerical tolerance remains unresolved. |
| Type gap 2 | Accepted and resolved | Dense Stripe Shape And Indexing | Required absolute time axis and `shape[0] == wc.Nt`. | Existing sparse decode uses `j = pixel_index // wc.Nf`. | None. |
| Type gap 5 | Accepted and resolved | Dense Stripe Shape And Indexing; Runtime Validation | Required `Nc == waveform.AT.shape[0]`. | Prevents silent channel truncation. | None. |
| Type gap 6 | Accepted and resolved | Numerical Tolerances | Defined `amplitude_source` as max absolute waveform amplitude over all channels and selected time samples. | Review A6. | Empty ranges excluded from numerical tests. |
| Type gap 7 | Accepted and resolved | Public API | Added return annotation `-> None`. | Repository annotation policy. | None. |
| Type gap 8 | Accepted and resolved | Public API; Aligned Table Requirements | Defined semantic requirements for aligned table and required integrity tests. | Human decision 1. | Exact storage layout remains implementation freedom. |
| QA risk 1 | Accepted and partially resolved | Test Requirements; Verification Methods | Stated full suite/CI execution is out of required acceptance; required allowed-diff review and active targeted tests. | Human decision that CI test running is later and existing tests fail/are slow. | Durable CI coverage remains unresolved. |
| QA risk 2 / MC5 | Accepted and resolved | Test Requirements; QA And Repository Policy | Prohibited skip/xfail in new tests unless separately authorized. | Review A8, adjusted for human CI decision. | None. |
| QA risk 3 | Accepted and resolved | QA And Repository Policy | Clarified speed-test per-file policy does not relax implementation API annotations. | Existing ruff policy. | Speed demo annotations remain governed by existing repo config. |
| QA risk 4 | Accepted and resolved | QA And Repository Policy | Prohibited suppressions unless separately approved. | Review A8, strengthened per user decision A7 no inline comments for those additions because they may not be added at all. | None. |
| QA risk 5 | Accepted and resolved | Required Files; QA And Repository Policy | Prohibited QA config edits, exclusions, warning filters, and `conftest.py`-style collection changes. | Original allowed-diff policy and review A8. | None. |
| QA risk 6 | Deferred as explicitly out of scope | QA And Repository Policy | Did not require a `fastmath` setting; correctness tolerance is observable requirement. | Review marked as minor and not correctness risk. | Reproducibility across numba versions remains residual. |
| QA risk 7 | Accepted and resolved, revised again in v4 | Numerical Tolerances; Aligned Table Requirements | Removed looser distorted-table tolerance and banned relaxed float64 tolerance. v4 removes the v3 alternate-grid requirement and instead requires integer-aligned input assertions plus oracle-matching layout changes. | Human v2 decision removed loose tolerance; v3 review showed the alternate-grid approach was not scientifically supportable. | Non-float64 tolerance unresolved. |
| MC1 | Accepted and resolved | Forbidden Shortcuts And Memory Constraints | Explicitly forbids wrapper over `wavemaket` and full-band materialization. | Original memory motivation; B4. | Source inspection required. |
| MC2 | Accepted and resolved | Dense Stripe Shape And Indexing; Required Coverage Cases | Requires realistic full-band `wc.Nf` distinct from stripe height and odd/even stripe heights as window widths. | B1 and human decision 2. | Exact fixture values left to implementation. |
| MC4 | Accepted and resolved | Scientific And Mathematical Basis; Test Requirements | Banned nonzero-mask oracle comparisons. | B3 and A6. | None. |
| A1 | Accepted and resolved | Dense Stripe Shape And Indexing | Incorporated with revised `stripe_height` terminology. | Human decision 2. | None. |
| A2 | Accepted and resolved | Dense Stripe Shape And Indexing; Runtime Validation | Incorporated with `nf_start` terminology. | Human decision 2. | None. |
| A3 | Accepted and resolved | Scientific And Mathematical Basis | Incorporated. | Human decision 3. | None. |
| A4 | Accepted and resolved | Forbidden Shortcuts And Memory Constraints | Incorporated. | Human decision 3. | None. |
| A5 | Accepted and resolved | Scientific And Mathematical Basis; Test Requirements | Incorporated. | Human decision 3. | None. |
| A6 | Accepted and resolved | Scientific And Mathematical Basis; Numerical Tolerances; Test Requirements | Incorporated. | Human decision 3. | None. |
| A7 | Rejected with evidence | Benchmark/Demo Requirements | Did not add dense-experimental labeling, inline comments, or objective performance threshold; instead required benchmark evidence only. | Human decision: for A7 none of those may be added at all. | Performance threshold requires future human decision if desired. |
| A8 | Accepted and partially resolved | QA And Repository Policy; Test Requirements | Incorporated no skips, no suppressions, no config drift. Did not require captured pytest output or CI run. | Human decision says CI test running will be added later and existing tests fail/are slow. | CI execution remains unresolved. |
| A9 | Accepted and resolved | Required Coverage Cases | Incorporated offsets and slopes, adding `+/-1e-10`. | Human decision. | None. |
| V2-1 | Accepted and resolved by human decision | Public API; Dense Stripe Shape And Indexing; Runtime Validation; Test Requirements | Removed required float64 dtype assertion from the numba body, changed `wavelet_stripe` annotation to `NDArray[np.floating]`, removed invalid non-float64 dtype test requirement, and limited faithful numerical acceptance tests to float64. | Human v3 decision: no dtype assertion is required, dtype need not be float64, and `np.floating` is the intended type. | Exact numerical acceptance tolerance for float32 or other floating dtypes is unresolved. |
| V2-2 | Accepted and resolved with revised authority, superseded in v4 | Aligned Table Requirements; Implementation Variants; Test Requirements | Replaced the v2 hollow aligned-table requirement with required two-sided padding, valid-region metadata, constant-stride support, and integer-alignment input assertions. Removed the v3 alternate-grid requirement. | Human v4 instruction that the alternate-grid wording is dropped and replaced with assertions about inputs; v3 review V3-1. | Exact aligned layout remains implementation freedom subject to oracle tolerance. |
| V2-3 | Accepted and partially resolved | Required Coverage Cases; Finding Disposition Ledger | Derivative-index drop coverage may use an out-of-supported-domain targeted test with zero-contribution expectation, or source inspection confirming the guard exists in all three functions. | Review showed the guard may be unreachable in the supported linear drift domain; oracle behavior remains authority. | If source inspection is used, behavioral exercise of that guard remains absent. |
| V2-4 | Accepted and resolved | Runtime Validation | Added that validation assertions occur before the per-pixel hot loop and tolerance/closeness assertions must not appear inside required numba functions. | Review confirmation and existing `wavemaket` behavior. | None. |
| V2-5 | Accepted and resolved | Numerical Tolerances | Required numerical-correctness cases to use non-empty ranges and `amplitude_source > 0`. | Review showed all-zero amplitudes make the test trivial. | None. |
| V2-6 | Accepted and resolved | Test Requirements | Replaced counterfactual overwrite-test wording with observable requirement: prefill with nonzero `B` and assert `result == B + oracle_slice`, not merely `oracle_slice`. | Review V2-A5. | None. |
| V2-7 | Deferred as informational | Benchmark/Demo Requirements; Test Requirements | No additional contract language added for layout opt-out or pairwise-comparison tautology beyond existing oracle requirements. | Review marked these as minor/carried-over and acceptable. | Pairwise comparison remains secondary to oracle. |
| MC-v2-A | Accepted and resolved | Aligned Table Requirements | Explicitly forbids identity wrapper that lacks aligned metadata, two-sided padding, valid-region metadata, and constant-stride layout support. | Human v4 instruction and v3 review V3-A1. | None. |
| MC-v2-B | Accepted and partially resolved | Required Coverage Cases | Derivative-index guard must be present by source inspection or exercised with a targeted out-of-domain zero-contribution test. | Review V2-3. | Behavioral in-domain tests may not exercise the guard. |
| MC-v2-C | Accepted and resolved by human decision | Runtime Validation; Test Requirements | Removed dtype assertion and invalid dtype failure test requirement rather than prescribing a fragile numba dtype-check spelling. | Human v3 dtype decision. | Non-floating dtype behavior is not required beyond type annotations. |
| V2-A1 | Rejected with evidence from human decision | Runtime Validation | Did not require the suggested `isinstance(wavelet_stripe[0,0,0], np.float64)` assertion or Python validation wrapper. | Human v3 decision says to drop the dtype assertion requirement and non-float64 failure test. | None. |
| V2-A2 | Accepted with revised wording, superseded in v4 | Aligned Table Requirements | Required structural distinction by two-sided padding, valid-region metadata, constant-stride support, and input assertions rather than a changed coefficient grid. | Human v4 instruction; v3 review rejected the changed-grid approach. | Exact aligned layout remains implementation freedom. |
| V2-A3 | Accepted and resolved | Required Coverage Cases | Added out-of-domain guard test option or source-inspection option for derivative-index drop. | Review V2-3. | See V2-3. |
| V2-A4 | Accepted and resolved | Runtime Validation | Added no tolerance/closeness assertions inside required numba functions. | Review V2-4. | None. |
| V2-A5 | Accepted and resolved | Numerical Tolerances; Test Requirements | Added `amplitude_source > 0` and reworded pre-filled test around observable `B + oracle_slice`. | Review V2-5 and V2-6. | None. |
| V3-1 | Accepted and resolved | Runtime Validation; Aligned Table Requirements; Requirement Traceability Table | Removed the v3 alternate-grid requirement; required aligned input assertions `2 * wc.Nsf % 3 == 0` and `abs((wc.DF / wc.df_bw) - R) <= 1e-12 * max(1.0, abs(R))`; aligned table reuses original `wc.df_bw` grid coefficient values. | Adversarial review v3 showed the alternate-grid approach is unnecessary when alignment holds and oracle-breaking when it does not; user instructed that it be replaced with assertions about inputs. | Aligned path is unsupported for non-integer-aligned configurations until a future authority defines behavior. |
| V3-2 | Accepted and resolved | Implementation Variants; Aligned Table Requirements; Test Requirements | Removed requirement that padding alone reproduce table-overflow drops. Required exact drop reproduction through loop bounds, per-regime bounds, branchless validity factor, or equivalent mechanism; padding is only for bounded reads or already-excluded/proven-negligible contributions. | Adversarial review v3 showed edge coefficients are non-negligible and padding-only drops cannot meet tolerance. | Exact mechanism remains implementation freedom subject to full-stripe oracle tests. |
| V3-A1 | Accepted and resolved with user modification | Runtime Validation; Aligned Table Requirements | Adopted integer-alignment assertions and original-grid coefficient reuse; removed all alternate-grid requirements. | User instruction and review amendment V3-A1. | None for supported aligned inputs. |
| V3-A2 | Accepted and resolved | Implementation Variants; Aligned Table Requirements | Required bandwidth and table-overflow drops to come from loop bounds, branchless validity factors, or equivalent exact mechanisms rather than padding-only reads. | Review amendment V3-A2. | None. |
| V3-A3 | Accepted and partially resolved | Aligned Table Requirements; Benchmark/Demo Requirements; Unresolved Blockers | Stated the observable distinction of the aligned API as two-sided padded layout, valid-region metadata, and constant-stride support; retained no performance threshold. | Review amendment V3-A3 and standing human decision rejecting performance thresholds. | Whether aligned layout is measurably faster remains a human review decision after benchmark data. |
| V3 minor/carried-over | Deferred as informational | Test Requirements; Benchmark/Demo Requirements | No further changes for pairwise-comparison tautology or measured layout opt-out. | Review marked these as informational/acceptable. | Pairwise comparison remains secondary to oracle. |
| V3 residual risk 1 | Requires human or repository-owner resolution | Unresolved Blockers | Listed aligned performance benefit as unresolved because no speed threshold or branch metric is authorized. | Review v3 residual risk and prior human decision. | Needs future human judgment after benchmark data. |
| V3 residual risk 2 | Requires human or repository-owner resolution | Unresolved Blockers | CI execution remains unresolved. | Review v3 residual risk and prior human decision. | Needs future CI policy. |
| V3 residual risk 3 | Deferred as explicitly out of scope | Numerical Tolerances; Unresolved Blockers | Retained float64 oracle tolerance and did not specify numba/LLVM fastmath reproducibility policy. | No authority supplied for environment-pinned tolerance policy. | Empirical adequacy remains implementation-dependent. |
| Residual risk 1 | Requires human or repository-owner resolution | Unresolved Blockers | Listed as unresolved CI decision. | Human decision defers CI. | Needs future CI policy. |
| Residual risk 2 | Accepted and partially resolved | Required Coverage Cases; Unresolved Blockers | Required bandwidth and table-overflow guard tests; derivative-index guard may be tested out of domain or confirmed by source inspection. | Review V2-3 and oracle authority. | Behavioral in-domain exercise of derivative-index drop remains unresolved. |
| Residual risk 3 | Deferred as explicitly out of scope | Runtime Validation; Unresolved Blockers | Stated stripped-assert invalid behavior is undefined; did not require bounds-checked numba run. | Original validation policy uses assertions only. | Debug bounds-check policy unresolved. |
| Residual risk 4 | Deferred as explicitly out of scope | Benchmark/Demo Requirements; Unresolved Blockers | Required layout reporting; did not require cross-machine reproducibility. | Benchmark environment policy not supplied. | Needs future human judgment. |
| Residual risk 5 | Deferred as explicitly out of scope | Numerical Tolerances; Unresolved Blockers | Retained faithful tolerance from original contract. | Original numerical tolerance authority. | Empirical adequacy depends on implementation. |

## Requirement Traceability Table

| Requirement | Source of authority | Contract section | Verification method | Dependencies or unresolved questions |
|---|---|---|---|---|
| TR1: Add new dense-stripe alternative to `wavemaket`. | Original intended capability. | Status And Scope; Public API | Import required functions. | None. |
| TR2: Support only `force_nulls=0` and `amplitude_order=0`. | Original scope. | Status And Scope; Scientific And Mathematical Basis | Oracle tests call `wavemaket` with those options. | None. |
| TR3: Provide three supported functions including aligned-table version. | Original two APIs plus human decision 1. | Public API; Implementation Variants | Signature inspection and tests import each function. | Exact aligned table storage layout is implementation freedom. |
| TR4: Use `(Nt, stripe_height, Nc)` and `nf_start`. | Human decision 2. | Public API; Dense Stripe Shape And Indexing | Signature and shape assertion tests. | None. |
| TR5: Treat `stripe_height` as window into full `wc.Nf`. | Original motivation; B1 accepted. | Dense Stripe Shape And Indexing | Tests use `wc.Nf > stripe_height` and odd/even stripe heights. | None. |
| TR6: Map global `k` to stripe column `k - nf_start` and drop outside stripe. | B2 accepted; necessary consequence of dense stripe window. | Dense Stripe Shape And Indexing | Boundary tests near stripe lower/top edge. | None. |
| TR7: Use oracle construction through `wavemaket` and `wavelet_sparse_to_dense`. | Original oracle requirement; B3 accepted. | Scientific And Mathematical Basis | Test source inspection and oracle-based assertions. | None. |
| TR8: Reproduce all oracle drop guards. | Existing `wavemaket` behavior; B3 accepted. | Scientific And Mathematical Basis | Tests hitting derivative, bandwidth, and table-overflow drops. | Exact fixture construction is implementation work. |
| TR9: Compare over entire stripe including zero cells. | B3/A6 accepted. | Scientific And Mathematical Basis; Test Requirements | Tests must use full-array `assert_allclose`. | None. |
| TR10: Match faithful tolerance for float64 acceptance cases only. | Original tolerance, human v2 decision removing loose tolerance, human v3 dtype decision. | Numerical Tolerances | Full-stripe oracle comparisons using float64 stripes. | Non-float64 tolerance unresolved. |
| TR11: Define `amplitude_source` as max abs amplitude over selected channels/time and require it to be positive in numerical tests. | A6 accepted; V2-5 accepted. | Numerical Tolerances | Test helper/source inspection. | None. |
| TR12: Accumulate with `+=`, not overwrite. | Original "coadd" intent; A5 accepted. | Scientific And Mathematical Basis; Test Requirements | Pre-filled-stripe test. | None. |
| TR13: Forbid `wavemaket` wrapper and full-band intermediates in required functions. | Original memory motivation; B4 accepted. | Forbidden Shortcuts And Memory Constraints | Source inspection and possible monkeypatch/adversarial tests. | Source inspection required for helper indirection. |
| TR14: Require aligned table with integer-alignment input assertions, two-sided zero padding, valid-region metadata, and constant-stride layout support. | Human v4 instruction; adversarial review v3 V3-1/V3-2. | Runtime Validation; Aligned Table Requirements | Input-assertion tests, structural layout tests, padding tests, identity-wrapper rejection, drop-behavior tests, and oracle differential tests. | Exact aligned storage layout remains implementation freedom; non-integer-aligned inputs are unsupported for the aligned path. |
| TR15: Require concrete slope and frequency-offset coverage including `+/-1e-10`. | Original boundary coverage; human decision A9. | Required Coverage Cases | Parametrized tests. | Reduced matrix allowed if each requirement covered. |
| TR16: Require `stripe_height` 2 through 5, odd/even `nf_start`, `Nc == 3`, representative `wc.Nt == 2048`. | Original boundary cases; B1 fix. | Required Coverage Cases; Benchmark/Demo Requirements | Tests or benchmark/demo setup. | None. |
| TR17: Runtime validation by required assertions only, excluding dtype assertion. | Original validation policy; human v3 dtype decision. | Runtime Validation | Invalid-input assertion tests for shape/range/channel/time constraints. | Invalid behavior under stripped assertions undefined; non-floating runtime behavior not specified. |
| TR18: Use numba `njit` and nopython-compatible required functions. | Original QA constraints. | QA And Repository Policy | Source inspection and numba compilation. | Exact `fastmath` setting not specified. |
| TR19: Do not modify existing behavior or files outside allowed set. | Original compatibility policy; human CI decision. | Required Files; Verification Methods | `git diff --name-only`. | None. |
| TR20: No new suppressions, config exclusions, skips, warning filters, or broad dynamic public interfaces without separate approval. | Standing QA requirement from prompt; review A8. | QA And Repository Policy | Source/config/test diff review. | Future approvals must be explicit. |
| TR21: Required benchmark/demo with correctness validation before timing and all four variants. | Original benchmark requirement plus human decision 1. | Benchmark/Demo Requirements | Inspect or run benchmark script. | No speed threshold authorized. |
| TR22: Full repository test suite and CI changes are not required by this contract. | Human decision. | Test Requirements; Verification Methods | Confirm no CI/config diffs required. | Future CI policy unresolved. |
| TR23: Public stripe dtype is `NDArray[np.floating]`; implementation must not reject floating dtypes solely because they are not float64. | Human v3 decision. | Public API; Dense Stripe Shape And Indexing; Runtime Validation | Signature inspection and absence of non-float64 failure tests. | Exact non-float64 numerical tolerance unresolved. |
| TR24: Tolerance/closeness checks must stay out of required numba functions. | V2-4 accepted; existing `wavemaket` behavior. | Runtime Validation; QA And Repository Policy | Source inspection. | None. |
| TR25: Aligned path must use the original coefficient grid and values. | Human v4 instruction; adversarial review v3 V3-1. | Aligned Table Requirements | Source and structural test inspection confirming original `wc.df_bw` grid use and exact original coefficient-value reuse in valid regions. | None for supported aligned inputs. |
| TR26: Aligned path must not use zero padding as the sole mechanism for table-overflow or bandwidth drops at valid-range edges. | Adversarial review v3 V3-2. | Implementation Variants; Aligned Table Requirements; Test Requirements | Drop-behavior test for overflow edge cells plus source inspection for loop bounds, branchless validity factor, or equivalent exact mechanism. | Exact mechanism is implementation freedom. |

## Unresolved Blockers

- CI execution for the new tests is unresolved. Human decision says CI test running will be added later, so this v4 does not require CI or full-suite execution.
- No objective performance threshold is authorized. The benchmark/demo must provide evidence, but acceptance of speed results requires future human review.
- Exact aligned table storage type, constructor name, and layout algorithm are not fixed. The contract defines observable input assertions, two-sided zero padding, valid-region metadata, constant-stride support, identity-wrapper rejection, and oracle-matching behavior, leaving storage layout and construction algorithm to implementation.
- Non-integer-aligned configurations for `wavemaket_stripe_dense_aligned` are unsupported until a future authority defines behavior; the function must assert and reject them.
- Exact numerical tolerances for non-float64 floating outputs are unresolved. Float64 remains the required dtype for faithful numerical acceptance tests.
- Exact waveform fixture parameters that hit bandwidth and table-overflow drop guards are not specified. The implementation must provide tests demonstrating those guards are exercised.
- Behavioral in-domain exercise of the derivative-index drop is unresolved; this contract allows either an out-of-supported-domain guard test or source inspection.
- Whether the aligned layout is measurably faster than `wavemaket_stripe_dense` is unresolved because no performance threshold or branch metric is authorized.
- Numba bounds-check or debug-build policy for catching out-of-bounds writes is unresolved and outside this contract.
- Cross-machine benchmark reproducibility policy is unresolved.

## Change Summary

### Clarifications That Preserve Intent

- Replaced ambiguous stripe shape `(Nt, Nf, Nc)` with `(Nt, stripe_height, Nc)`.
- Replaced `nf_min` with `nf_start`.
- Defined the stripe as a window into full-band `wc.Nf`.
- Specified global-to-stripe index mapping.
- Defined full oracle construction and full-stripe comparison.
- Enumerated the oracle drop guards that must be reproduced as zeros.
- Defined `amplitude_source`.
- Required numerical-correctness tests to use `amplitude_source > 0`.
- Required additive coaddition semantics and a pre-filled-stripe test.
- Reworded the pre-filled-stripe test as the observable assertion `result == B + oracle_slice`.
- Required concrete coverage for frequency offsets, slopes, stripe heights, parity, and edge windows.
- Clarified that derivative-index drop coverage may be out-of-domain or source-inspected.
- Clarified that tolerance comparisons belong in tests and benchmark/demo checks, not inside required numba functions.
- Clarified that the aligned path is supported only for integer-aligned input configurations satisfying `2 * wc.Nsf % 3 == 0` and `wc.DF / wc.df_bw == 2 * wc.Nsf // 3` within the stated tolerance.
- Clarified that zero padding is not a substitute for exact oracle bandwidth or table-overflow drops at valid-range edges.

### Newly Authorized Requirements

- Added `wavemaket_stripe_dense_aligned` as a third supported function version.
- Required an aligned table representation with two-sided zero padding for readable row regions.
- Required aligned-path input assertions, constant-stride metadata/support, structural tests, padding tests, drop-behavior tests, and identity-wrapper rejection.
- Required the benchmark/demo to include the aligned-table function and to verify correctness before timing.

### Removed Or Narrowed Requirements

- Removed the v1 looser tolerance allowance for distorted or realigned interpolation-table experiments.
- Removed any benchmark-only distorted-table acceptance path.
- Removed the v2 requirement that `wavelet_stripe` be float64 or that the numba function body assert float64 dtype.
- Removed the v2 requirement that tests verify failure for non-float64 dtype.
- Removed the v2 requirement that aligned-table coefficient values be lossless or bitwise equal to original table entries, then removed v3's alternate-grid authorization in v4; v4 requires original-grid coefficient reuse for supported aligned inputs.
- Removed the v3 alternate-grid requirement; v4 reuses the original coefficient grid and asserts the aligned-input precondition instead.
- Removed the v3 requirement that table-overflow drops be obtained from padding alone.
- Did not add A7's proposed performance threshold or experimental labeling requirement because the supplied human decision rejected those additions.
- Did not require full repository test-suite execution or CI changes.

### QA-Enforcement Changes

- Required assertions for key shape, range, channel, and time constraints rather than merely allowing them.
- Prohibited new skips, xfails, warning filters, coverage exclusions, linter/type suppressions, and checker configuration changes unless separately approved.
- Reaffirmed allowed diffs as the repository-scope enforcement mechanism for this contract.
- Required tests to be active and oracle-based over the full stripe.

### Out-Of-Scope Recommendations

- Add a future CI job for `tests/test_taylor_time_wavelet_optimized.py`.
- Define a repository-owner-approved performance threshold after benchmark data exists.
- Add a bounds-checked numba debug run for this module if maintainers want stronger out-of-bounds detection.
- Pin a benchmark environment if cross-machine performance comparisons become approval-critical.
