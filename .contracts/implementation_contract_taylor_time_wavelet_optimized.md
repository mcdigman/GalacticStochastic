# Implementation Contract v6 (Consolidated): Dense-Stripe Taylor-Time Wavelet Coaddition

> **Revision r2 (PR #42 review + human decision HD-1).** r1 resolved findings
> F001–F008 from the PR #42 contract-adversary review (dispositions in "PR #42
> contract-adversary dispositions"; new requirements TR37–TR44). **r2 applies
> authorized human decision HD-1:** LLVM IR inspection is now a **required**
> divergence-free verification (TR45), superseding the LLM-drafted "no fragile
> compiled-code inspection" phrasing that the cross-family adversary correctly
> flagged as F002; F002 is now fully resolved, not routed to the human. See
> "Authoritative Human Decisions." This revision does not change any frozen Design
> Intent. **Not yet re-approved — a design-adversary and the human approver still
> own the freeze decision.**

## Status And Scope

This contract **supersedes and replaces** both of the following as the single
authoritative specification for this work:

- `implementation_contract_taylor_time_wavelet_optimized_final.md`
- `implementation_contract_taylor_time_wavelet_optimized_amendment_1.md`

Those two documents are now historical. Where this document is silent, the
superseded documents do **not** apply by default; this file is intended to be
read standalone. The Finding Disposition Ledger records which prior dispositions
are carried forward, which are superseded, and why.

This consolidation exists because review of PR #40 surfaced two contract defects
plus a recurrence of the same defect class inside Amendment 1:

- **CD-A (under-specification, original `final`):** the correctness gate ("match
  the fastmath oracle exactly, including dropped cells") combined with no
  measurable structural/performance gate let an implementation defeat the
  optimization purpose while passing every test.
- **CD-B (over-specification, original `final`):** the aligned-table section
  required the aligned object to *store* `wc` configuration fields that are
  passed separately at call time, producing a perpetual false-positive
  noncompliance finding.
- **CD-A′ (recurrence, Amendment 1):** Amendment 1 patched CD-A by *prescribing a
  control-flow recipe* (regime-splitting + per-parity sub-loops + per-regime
  loop bounds). That recipe makes the *inner loop body* branch-free by
  **relocating** branches into data-dependent loop bounds and duplicated
  sub-loops — which does not serve the actual purpose (see Design Intent) and is
  itself a divergence source on the eventual GPU target.

The implementation must add direct dense-stripe coaddition for one waveform over
a narrow frequency stripe, supporting only:

- `force_nulls=0` equivalent behavior.
- `amplitude_order=0`.
- Dense stripe accumulation into an existing output array.
- Linearly evolving intrinsic source frequency over the supported test domain.

It does not need full `wavemaket` feature parity.

---

## Authoritative Human Decisions

These are explicit decisions made by the human maintainer. They are the
**highest-authority** inputs (above Design Intent and above any LLM-drafted
contract wording); where any other text in this document conflicts with a
decision here, the decision here governs. Each is dated and traceable.

| ID | Date | Decision | Supersedes / resolves | Sections |
|---|---|---|---|---|
| HD-1 | 2026-06-26 | The aligned function's divergence-free property **must** be verified at the **LLVM IR** level (via numba's `inspect_llvm()` or equivalent), in addition to source/AST inspection, at the repository-pinned python/numba/LLVM versions. LLVM IR is machine- and language-independent and is harder to adversarially evade than Python source/AST. The genuinely fragile, **excluded** target is the **final machine-dependent compiled output** (native assembly / PTX / object bytecode), which is machine-specific and is **not** required. | Supersedes the LLM-drafted TR35 phrasing "no reliance on fragile compiled-code inspection," which was a same-model-family drafting blind spot (correctly flagged by the cross-family adversary as F002), not a human decision. Resolves F002. | Test Requirements (TR45); Operative Definitions; Verification Methods; Unresolved Blockers |
| HD-2 | Step-0 | Derivative-domain drop is handled **in-kernel** via clamp-and-mask folded into the amplitude prefactor; no control-flow guard. | — | Implementation Variants — aligned (TR33) |
| HD-3 | Step-0 | The CPU performance floor is a **soft, flagged** aligned-vs-dense median-ratio trigger (intended ≤1.10×), independently re-measured, escalated to human — not an auto-fail. | — | Benchmark/Demo Requirements (TR34/DI-3); strengthened by F003/TR40 (a ratio above the floor blocks auto-approval) |

(The PR #40 maintainer findings CD-A and CD-B that motivated the v6 consolidation
also originate from human review; they are recorded in the Finding Disposition
Ledger below.)

---

## Design Intent

This section states *why* each function exists. It is normative: where the
literal requirement text below conflicts with this intent, the implementer must
**not** silently resolve the conflict in favor of the letter — see
"Spirit-Over-Letter And Conflict Escalation."

- **DI-1 — Parallel stripe coaddition (core deliverable).** Future higher-level
  parallel coaddition operates on independent dense frequency stripes and sums
  them later. The full wavelet transform is too large to materialize many full
  instances, so each worker must operate on a narrow frequency range without
  materializing full-band intermediates. `wavemaket_stripe_sparse` and
  `wavemaket_stripe_dense` are the production CPU deliverables for this.

- **DI-2 — `wavemaket_stripe_dense_aligned` is a GPU-readiness staging
  reference.** Eventually (pending a larger rewrite of the callers, which today
  make a GPU port useless because of device-transfer overhead) this computation
  will run on GPU. A GPU/SIMT implementation requires **divergence-free**
  execution — essentially all data-dependent control flow must be eliminated, or
  it suffers an unacceptable performance or clarity penalty. The aligned CPU
  variant exists to work out and demonstrate that divergence-free structure on
  CPU **now**, so the future GPU kernel is a near-mechanical translation rather
  than a redesign. Its value is the *structure*, not raw CPU speed.

- **DI-3 — The aligned variant must be an adequate CPU drop-in.** Because the
  hard performance gate lives on the *future GPU* implementation, this contract
  does not impose a hard CPU speed requirement. But the aligned variant should
  not make CPU performance meaningfully worse than `wavemaket_stripe_dense`
  (so it can replace existing behavior without regressing), and a correctly
  predicated implementation is expected to be at least competitive. Hence a
  **soft, flagged** CPU performance floor (DI-3 maps to the perf-floor
  requirement; it is a review-escalation trigger, not an auto-fail).

- **DI-4 — Faithful numerics.** All three functions must reproduce the
  `wavemaket(force_nulls=0, amplitude_order=0)` oracle over the entire stripe
  slice within the faithful float64 tolerance, except for the rare
  integer-stride ULP-boundary cells explicitly carved out below (DI-2's
  divergence-free integer-stride arithmetic legitimately disagrees with the
  oracle's fastmath arithmetic at those cells).

---

## Non-Goals And Deferred Future Work

- This is **not** a GPU implementation. It is a CPU reference whose *structure*
  is GPU-portable. No CUDA/GPU code is in scope.
- **Caller-side derivative-domain handling is deferred.** One future option for
  the GPU rewrite is to filter/compact out-of-domain time pixels at the caller
  so the kernel can assume an in-domain derivative index. The current callers do
  **not** provide that guarantee, so this contract requires the derivative-domain
  drop to be handled **in-kernel** via clamp-and-mask (see TR33). The caller-side
  option is recorded here as future work and must not be assumed by this
  implementation.
- No hard CPU performance threshold (see DI-3); only a soft flagged floor.
- No full `wavemaket` feature parity, non-`force_nulls=0` null handling,
  higher-order amplitude, or arbitrary nonlinear frequency evolution.
- No changes to existing TDI routines or other tracked files beyond the allowed
  diff.

---

## Operative Definitions

These definitions pin terms that were previously left as prose proxies. They are
normative.

- **Branch / data-dependent control flow.** A control-flow construct whose taken
  path or iteration count depends on per-element runtime data (waveform values,
  per-pixel frequency, per-`k` keep/drop). On SIMT hardware these cause warp
  divergence. This includes: `if`/`else` that selects whether or what to write;
  `break`/`continue`/early-return governed by data; and **loops whose trip
  count, bounds, or stride depend on runtime data** (e.g. per-time-pixel or
  per-regime `k` ranges).

- **Divergence-free predication (the required structure for the aligned
  variant).** All of:
  1. **Uniform iteration domain.** For every channel `itrc` and time pixel `j`,
     the set of frequency layers iterated is exactly the stripe window
     `[nf_start, nf_start + stripe_height)` — a fixed, data-independent trip
     count of `stripe_height`. Keep/drop is **not** expressed by restricting
     which `k` are visited.
  2. **Decisions as masks/selects.** Every keep/drop (bandwidth + table
     overflow), reflection, parity-sign, and derivative-domain decision is
     reduced to data-independent control flow plus arithmetic: a multiplicative
     `0.0`/`1.0` mask, or an arithmetic select (e.g. `a*(1-m) + b*m`).
  3. **No data-dependent loop bounds, strides, trip counts, or per-element
     early-exit.** Regime-splitting into contiguous data-dependent `k`
     sub-ranges, per-parity sub-loops with data-dependent start/step, and
     per-regime loop bounds are **prohibited** (this is the CD-A′ relocation
     anti-pattern).
  4. **Address safety is separate from drop semantics.** Reads are kept in
     bounds by table padding (bandwidth axis) and index clamping (derivative
     axis). The actual *drop* always comes from the oracle-derived mask, never
     from "the padded value happens to be ≈0."

- **Allowed, not a branch:** fixed-trip loops whose bounds are compile-time or
  shape constants (over `Nc`, over `stripe_height`); `min`/`max`/clamp used to
  keep a *read index* in bounds (these lower to branchless selects and do not
  change trip counts); arithmetic ternaries the compiler lowers to a select that
  do not alter control flow or trip count. Under HD-1, "lowers to a branchless
  select" is not merely asserted — it is **verified at the LLVM IR level** (TR45):
  such constructs must appear in the IR as `select`/arithmetic, and only the
  data-independent fixed-trip loop control may appear as a conditional branch.

- **Oracle.** `wavemaket(force_nulls=0, amplitude_order=0)` decoded to dense via
  `wavelet_sparse_to_dense` and sliced to the stripe window (see Scientific And
  Mathematical Basis).

---

## Required Files

Required new implementation file:

- `WaveletWaveforms/taylor_time_wavelet_optimized.py`

Required new test file:

- `tests/test_taylor_time_wavelet_optimized.py`

Required speed benchmark/demo:

- `speed_tests/taylor_time_wavelet_optimized_speed_demo.py`

Strictly additive import/export edits are allowed only in existing `__init__.py`
files if required for imports. No other existing tracked Python, configuration,
coefficient, test, workflow, or documentation file may be modified.

---

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

All three are supported APIs, not throwaway benchmark-only variants. `nf_start`
replaces the original `nf_min` parameter to avoid colliding with `wavemaket`'s
full-band lower-edge local variable.

The implementation must define `WaveletTaylorTimeCoeffsAligned` as a
numba-compatible table type (NamedTuple, numba-compatible object, or equivalent)
implementing the zero-padded, integer-aligned representation in "Aligned Table
Requirements." A public constructor helper from `WaveletTaylorTimeCoeffs` is
allowed; if public it must have narrow annotations and a NumPy-style docstring.

---

## Dense Stripe Shape And Indexing

`wavelet_stripe` has shape `(Nt, stripe_height, Nc)` where:

- `Nt == wc.Nt`.
- `stripe_height == wavelet_stripe.shape[1]`, in `{2, 3, 4, 5}`.
- `Nc == wavelet_stripe.shape[2] == waveform.AT.shape[0]`.
- `wavelet_stripe.dtype` is a NumPy floating dtype.
- `wavelet_stripe` is C-contiguous or F-contiguous.
- `wavelet_stripe` is a **writable** NumPy `ndarray` usable in numba nopython
  mode (F008). Passing a read-only array, or a non-`ndarray` array-like, is
  undefined behavior; public docstrings must state this. The functions must not
  copy `wavelet_stripe` internally — accumulation is into the caller's array
  (verified by the in-place mutation test). Consistent with the assertions-only,
  no-defensive-branch policy, tests are not required to cover read-only or
  array-like inputs (a read-only negative case is permitted but optional).

`wc.Nf` is the full number of wavelet frequency layers (e.g. 256), distinct from
`stripe_height`. The stripe is the contiguous global frequency-layer window
`[nf_start, nf_start + stripe_height)` inside the full band, with
`0 <= nf_start` and `nf_start + stripe_height <= wc.Nf`.

The time axis uses absolute `wavemaket` time-pixel coordinates. A contribution
for full-band time pixel `j` and global frequency layer `k` maps to
`wavelet_stripe[j, k - nf_start, c]`. No function may write outside columns
`[0, stripe_height)`. Contributions to global layers outside the stripe window
are dropped (not wrapped, clamped, or accumulated into edge columns).

The implementation must use `wc.Nf`, not `stripe_height`, wherever full-band
wavelet math requires the full frequency-layer count (pixel-index encoding,
layer-center calculations, parity expressions depending on `j + k`).

Shape may be treated as a compile-time constant, including numba recompilation
for different shapes/layouts.

---

## Runtime Validation

Runtime input validation uses assertions only. With assertions disabled,
invalid-input behavior is undefined and need not branch defensively.

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
- With `R = 2 * wc.Nsf // 3`, the integer-alignment ratio in the
  **cross-multiplied form** `abs(wc.DF - R * wc.df_bw) <= 1e-15 * max(1.0, abs(R)) * wc.df_bw`.
  The division form `abs(wc.DF / wc.df_bw - R) <= 1e-15 * max(1.0, abs(R))` must
  **not** appear in this `@njit(fastmath=True)` body: re-evaluating
  `wc.DF / wc.df_bw` is subject to fastmath common-subexpression elimination that
  can perturb the integer-stride arithmetic. (Carried from Amendment 1 AM1-R7.)
- Aligned-table consistency with `wc` and the supported stripe-height domain,
  limited to what the aligned representation actually needs:
  `taylor_table_aligned.R == R`; `taylor_table_aligned.Nfd_negative == wc.Nfd_negative`;
  `Nfsam` size equals `wc.Nfd`; coefficient arrays have row count `wc.Nfd` and
  equal shapes; padding is large enough for predicated reads (see Aligned Table
  Requirements). The aligned table is **not** required to store `wc.df_bw`,
  `wc.DF`, `wc.Nf`, `wc.Nt`, `wc.dfd`, or the derivative grid — `wc` is supplied
  at call time, and no consistency claim may be based on the table re-storing
  those fields. (Supersedes CD-B.)

Numerical tolerance or closeness assertions must not appear inside the required
numba functions; those comparisons belong only in tests and the benchmark.

The numba bodies are not required to assert the dtype of `wavelet_stripe`, and
tests must not require a non-float64 floating dtype to fail solely because of
dtype. Float64 is the required dtype for faithful numerical acceptance tests.

---

## Scientific And Mathematical Basis

The oracle is `wavemaket(force_nulls=0, amplitude_order=0)` run with the same
`waveform`, `nt_lim_waveform`, `wc`, and original `WaveletTaylorTimeCoeffs`
table, decoded by: (1) run into a `SparseWaveletWaveform`; (2)
`wavelet_sparse_to_dense(oracle_sparse, wc)` → shape `(wc.Nt, wc.Nf, Nc)`; (3)
slice `[:, nf_start:nf_start + stripe_height, :]`.

All correctness comparisons must cover the entire stripe slice, including cells
expected to be exactly zero. Nonzero-mask comparisons (`oracle != 0` etc.) are
prohibited.

All three functions must reproduce the oracle's drop behavior. A stripe cell
`(j, k, c)` may receive a nonzero contribution only when `wavemaket` would write
`(j, k, c)` under all of:

- Derivative-table index condition: `0 <= ny + wc.Nfd_negative < wc.Nfd - 1`,
  with `ny = floor(waveform.FTd[c, j] / wc.dfd)`.
- Bandwidth: `ceil((fa - half_bandwidth) / wc.DF) <= k <= floor((fa + half_bandwidth) / wc.DF)`,
  `fa = waveform.FT[c, j]`, `half_bandwidth` from the two adjacent
  `taylor_table.Nfsam` values exactly as in `wavemaket`.
- Stripe window: `k` in `[nf_start, nf_start + stripe_height)`.
- Table-overflow guard: `0 <= jj1 < Nfsam1_loc - 1` and `0 <= jj2 < Nfsam2_loc - 1`.

Where any condition fails, the new functions must contribute `0.0`. The required
functions must accumulate (`+=`); they must not overwrite existing contents.

---

## Supported Waveform Domain

Supported inputs are selected so that intrinsic source frequency evolves
linearly over time; total drift is no more than two full-band pixels over the
observation; slope may be positive, negative, zero, nearly zero, or very nearly
zero; initial source frequency is within `±0.5` full-band pixel of the stripe
center; Doppler/polarization/amplitude modulation come from existing TDI
functions (which must not be modified and are not benchmarked).

Unsupported: non-`force_nulls=0` null handling, higher-order amplitude,
arbitrary nonlinear frequency evolution, performance guarantees outside the
linear-frequency domain, and any TDI changes.

**Behavior outside the supported waveform domain (F005).** For inputs outside the
supported waveform domain (nonlinear `waveform.FT`, total drift exceeding two
full-band pixels, or any other unsupported regime listed above), numerical
behavior is **undefined**, and required acceptance tests need not cover it.
Consistent with the assertions-only, no-defensive-branch policy, the functions are
**not** required to detect, assert, or reject nonlinearity or drift bounds. Public
docstrings must state that results outside the supported waveform domain are
undefined. This is distinct from the **in-domain** derivative-table drop, which is
a specified contribute-`0.0` behavior handled by clamp-and-mask (TR33), not
undefined behavior.

---

## Numerical Tolerances

For float64 `wavelet_stripe`, all three functions must match the oracle stripe
slice with:

```text
atol = 1e-10 * amplitude_source
rtol = 1e-9
amplitude_source = float(np.max(np.abs(waveform.AT[:, nt_lim_waveform.nx_min:nt_lim_waveform.nx_max])))
```

Numerical-correctness cases must use a non-empty selected time range and
`amplitude_source > 0`. No path may use a looser float64 tolerance. For non-
float64 floating dtypes, the functions must not reject solely because of dtype;
exact non-float64 tolerances are unresolved and not part of required acceptance.

**Aligned-path ULP-boundary exception (carried from Amendment 1 AM1-1, restated
for predication).** `wavemaket_stripe_dense_aligned` computes the interpolation
midpoint using the integer stride `R * k` (or an equivalent integer-stride
expression), never the oracle's fastmath product `(wc.DF / wc.df_bw) * k`. A
cell `(j, k, c)` is a **ULP-boundary cell** when

```text
int(floor(za - R * k - 0.5)) != int(floor(za - (wc.DF / wc.df_bw) * k - 0.5))
```

with `za = waveform.FT[itrc, j] / wc.df_bw`. At ULP-boundary cells the oracle's
fastmath and the integer-stride arithmetic can make different table-overflow
keep/drop decisions; these cells are **exempt** from the `atol`/`rtol`
comparison. At such a cell the aligned output may be either `0.0` or the
integer-stride interpolated value; tests must not fail solely because a
ULP-boundary cell differs from the oracle. All non-boundary cells must satisfy
the standard tolerance in full.

This exemption is what makes divergence-free integer-stride arithmetic jointly
satisfiable with the oracle-match requirement; without it, an implementation
would be forced to mirror the oracle's fastmath expression and could not be made
divergence-free (the CD-A failure).

---

## Implementation Variants

### `wavemaket_stripe_sparse`

- Write directly into `wavelet_stripe` with no full-band
  (`O(wc.Nf)`-frequency-extent) materialization. A variable-`k` inner loop is
  permitted for this variant (unlike the aligned variant); no specific prior loop
  structure is required to be preserved. (F007: the unobservable "preserve the
  current variable-`k` inner-loop structure as closely as practical" mechanism
  prescription was removed in favor of the observable outcomes below.)
- Use full-band `wc.Nf` for full-band wavelet calculations.
- Restrict writes to global layers in `[nf_start, nf_start + stripe_height)`.
- Accumulate in place; match the oracle within the faithful float64 tolerance.

### `wavemaket_stripe_dense`

This is the **production CPU baseline** and the reference against which the
aligned variant's soft performance floor is measured. It is **not** required to
be divergence-free.

- Use a fixed-`k` inner loop over the stripe window.
- Contribute `0.0` for `k` failing any oracle drop guard.
- Use the original `WaveletTaylorTimeCoeffs` representation.
- Ordinary control flow is permitted; clarity and correctness take priority.
- Accumulate in place; match the oracle within the faithful float64 tolerance.

### `wavemaket_stripe_dense_aligned`

This is the **GPU-readiness staging reference (DI-2)**. It must satisfy
**divergence-free predication** as defined in "Operative Definitions." The
requirements below state *outcomes*; the exact arithmetic is implementation
freedom. (Supersedes the Amendment 1 regime/parity recipe — see CD-A′.)

- **Uniform iteration domain.** For each `(itrc, j)`, iterate `k` over exactly
  `[nf_start, nf_start + stripe_height)` with a data-independent trip count.
- **No data-dependent loop bounds, strides, regime sub-ranges, per-parity
  sub-loops, or per-element early-exit.** Reflection (`R*k` vs `za`), parity
  (`(j + k) % 2` sign), bandwidth, and table-overflow decisions must each be an
  arithmetic mask or select within the uniform-domain body, not a partition of
  `k` or a branch.
- **Drops come from a multiplicative `0.0`/`1.0` mask** derived from the same
  bandwidth and `jj1`/`jj2` conditions as `wavemaket`. Zero padding and index
  clamping exist only to keep reads in bounds; they must **not** be the mechanism
  that produces a drop (edge coefficients are non-negligible — V3-2).
- **Integer stride.** Compute the interpolation midpoint with `R * k` (the
  expensive per-pixel `floor`/`dx` quantities for each reflection side may be
  lifted to a per-time-pixel boundary and reused across `k`, but `k`-selection
  between sides must be arithmetic, not a sub-loop). Never form
  `(wc.DF / wc.df_bw) * k` inside the function.
- **Derivative-domain handling: clamp-and-mask in-kernel (TR33).** Do not guard
  the derivative index with control flow. Compute
  `n_safe = min(max(n_ind, 0), wc.Nfd - 2)` for all table-row reads (so reads are
  always in bounds with `boundscheck` disabled and on GPU), and an arithmetic
  mask `mask_d = f64(0 <= n_ind) * f64(n_ind <= wc.Nfd - 2)`. Fold `mask_d` into
  the per-time-pixel amplitude prefactor so an out-of-domain pixel contributes
  exactly `0.0` everywhere in one multiply. The masked intermediates must be
  finite (no `inf`/`NaN`) before masking, so that `mask_d * value == 0.0`
  cleanly; this is guaranteed here because `dy ∈ [0, 1)` is bounded and clamped
  reads are valid finite table entries, and a test must assert it.
- **Precondition assertion** uses the cross-multiplied form (Runtime Validation).
- Accumulate in place; match the oracle within the faithful float64 tolerance,
  ULP-boundary cells exempted.

---

## Aligned Table Requirements

The aligned-table path is supported only when the original configuration is
already integer-aligned. It uses the original coefficient grid and must not
numerically alter the original `WaveletTaylorTimeCoeffs` coefficient values.

Precondition: `2 * wc.Nsf % 3 == 0`; `R = 2 * wc.Nsf // 3`; the cross-multiplied
ratio check in Runtime Validation holds. When it holds, the original `df_bw`
grid is already aligned with full-band layer spacing; the useful property is that
the fixed-`k` loop can read at integer stride `R` with `floor`/`dx` lifted to a
per-time-pixel boundary. The repository `Nsf = 150` configuration satisfies this
exactly (`wc.DF / wc.df_bw == 100.0`, `R = 100`).

The ratio tolerance is deliberately ULP-scale (`1e-15`): the integer-stride
lift-out replaces `(wc.DF / wc.df_bw) * k` with `R * k`, so its deviation scales
as `abs((wc.DF / wc.df_bw) - R) * k * (local slope)` and must stay within the
faithful tolerance for `k` up to `wc.Nf` (apart from the carved-out ULP-boundary
cells).

The aligned representation must:

- Reuse the original `WaveletTaylorTimeCoeffs` coefficient values on the original
  `wc.df_bw` grid (bitwise-equal reuse, or copies whose valid-region values are
  exactly equal, are both fine).
- Store `R` (or equivalent metadata proving the integer stride).
- Provide readable coefficient rows with **two-sided zero padding** around each
  row's original valid coefficient range.
- Store explicit valid-region metadata (or an equivalent observable convention)
  distinguishing original valid coefficients from left/right padding.
- Size the padded row width so that **every table index generated by the
  predicated uniform iteration domain** — all `k` in the stripe window, both
  reflection sides — is in bounds. (Padding serves address safety for
  predication; it is not the drop mechanism.)
- Support constant-stride reads by `R`.

**Anti-identity-wrapper requirement (replaces the CD-B field list).** The
required structural difference from `WaveletTaylorTimeCoeffs` is *layout*
(two-sided padding + valid-region metadata + `R` + predication-safe width), not
re-storage of `wc` fields. Expressed as a prohibited mechanism plus a must-fail
test: a representation that is a ragged/identity wrapper around
`WaveletTaylorTimeCoeffs` — lacking two-sided padding, valid-region metadata,
`R`, or predication-safe read width — **must fail** the aligned-table structural
test (see Test Requirements). The aligned table neither needs nor is prohibited
from incidentally storing other fields, but **no requirement, assertion, or test
may demand that it re-store `wc.df_bw`, `wc.DF`, `wc.Nf`, `wc.Nt`, `wc.dfd`, or
the derivative grid.**

Required verification for the aligned table:

- A structural test verifies the precondition (`2 * wc.Nsf % 3 == 0`, `R`, and
  the cross-multiplied ratio) and that aligned tests/benchmarks use a satisfying
  configuration (`Nsf = 150`, `R = 100`).
- A structural test verifies two-sided padding and valid-region metadata.
- A structural test verifies that an identity/ragged wrapper without the required
  layout fails the structural test.
- A padding test verifies representative left/right padding reads return `0.0`.
- A drop-behavior test verifies that table-overflow edge cells dropped by the
  oracle are exactly `0.0` in the aligned output — and that this holds because of
  the mask, not because padded reads blended a nonzero edge coefficient with
  zero padding.
- A differential correctness test compares the aligned function against the
  oracle over the full stripe slice for the same float64 cases as the other two
  functions (ULP-boundary cells exempted).

---

## Forbidden Shortcuts And Memory Constraints

None of the required functions may:

- Call `wavemaket` or `wavemaket_direct`.
- Allocate or materialize a full dense `(wc.Nt, wc.Nf, Nc)` array, a full-band
  `SparseWaveletWaveform`, or any per-call intermediate with frequency extent
  `O(wc.Nf)`.
- Use a wrapper over the sparse oracle as the implementation strategy.
- Hide full-band materialization behind helpers, aliases, closures, dynamic
  imports, generated-code labels, or equivalent indirection.

Per-call frequency-dimension scratch is limited to `O(stripe_height)`.

**Bounded aligned-table construction / no waveform-dependent full-band precompute
(F006).** No component introduced for this work — including the
`WaveletTaylorTimeCoeffsAligned` constructor and any cache or lookup structure,
whether or not it is allocated inside the three required functions — may
materialize a waveform-dependent full-band or `O(wc.Nf)`-frequency-extent
structure, nor a per-`nf_start` / per-frequency-layer lookup keyed across the full
band, as a way to avoid doing the intended stripe arithmetic at call time. The
aligned **coefficient table** itself is explicitly permitted: it is
waveform-independent (derived once from `WaveletTaylorTimeCoeffs` / `wc`), and its
size is bounded by the coefficient grid (`wc.Nfd` rows) plus the two-sided zero
padding required for predication-safe reads (see Aligned Table Requirements) — it
is **not** an `O(wc.Nf)` full-band structure and is not waveform-dependent.

**Anti-relocation clause (CD-A′).** `wavemaket_stripe_dense_aligned` may not
satisfy the divergence-free requirement by *relocating* data-dependent control
flow — e.g. converting a per-`k` branch into a data-dependent loop bound,
per-regime contiguous sub-range, per-parity sub-loop with data-dependent
start/step, or per-element early-exit. Such relocation is a divergence source on
the GPU target and is noncompliant regardless of whether the innermost loop body
contains a literal `if`.

Tests and benchmark code may construct the full-band oracle for verification and
timing comparison; this exception applies only outside the three required
functions and their private helpers.

---

## Required Coverage Cases

Correctness tests must compare each function against the full-stripe oracle for:

- `stripe_height` `2, 3, 4, 5`.
- At least one even and one odd `nf_start`.
- At least one stripe whose top edge is near, but not beyond, `wc.Nf`.
- Even `wc.Nt`, including a representative `wc.Nt == 2048` case in tests or
  benchmark setup.
- `Nc == 3`.
- Initial-frequency offsets from stripe center (full-band pixel units):
  `-0.5, -0.3, -0.25, 0.0, 0.25, 0.3, 0.5`.
- Linear slopes (full-band pixels per observation):
  `-2.0, -1.99, -1.0, -1e-3, -1e-10, 0.0, 1e-10, 1e-3, 1.0, 1.99, 2.0`.
- Frequencies exactly on pixel boundaries and exactly halfway between them.

Parametrization may replace the full Cartesian product if each listed
requirement is still exercised for each function and `stripe_height`. At least
one test must exercise the bandwidth drop and the table-overflow drop.

**Derivative-domain drop (revised for clamp-and-mask, TR33).** Because the
aligned function no longer contains a `0 <= n_ind < wc.Nfd - 1` control-flow
guard, the source-inspection-for-the-guard option is removed. Instead, a
targeted out-of-supported-domain derivative input must verify that affected time
pixels contribute exactly `0.0` (matching the oracle), produce no `NaN`/`inf`,
and — with numba bounds-checking disabled — do not fault.

---

## Test Requirements

Tests must:

- Compare all three functions against the oracle over the entire stripe.
- Compare the three functions against each other after independent calls
  (ULP-boundary cells exempted for the aligned vs others comparison).
- Verify in-place mutation (same array object and data pointer retained).
- Verify additive coaddition: prefill with known nonzero `B`, call each function,
  assert result equals `B + oracle_slice` (not merely `oracle_slice`); and that
  oracle-dropped cells are unchanged except for `B`.
- Verify required assertions for invalid shape, invalid `stripe_height`, invalid
  `nf_start`, mismatched `Nc`, and invalid time range — **parametrized across all
  three functions** (each has its own validation block; carried F002 repair).
- Verify the aligned-path assertions for a configuration violating
  `2 * wc.Nsf % 3 == 0` or the cross-multiplied ratio.
- Use an independent oracle construction (`wavemaket` + `wavelet_sparse_to_dense`),
  not the new implementation's helper logic.
- Avoid shape-only, nonzero-only, smoke-only, or placeholder-friendly assertions.
- Keep all new tests active; no `@pytest.mark.skip`/`xfail` without separate
  authorization.

Aligned-specific tests must additionally:

- **Divergence-free structural test (TR35).** By source/AST inspection, verify
  that `wavemaket_stripe_dense_aligned` **and every project-defined helper it
  calls, transitively, that executes inside the per-`(itrc, j, k)` accumulation
  path** are free of data-dependent control flow: the per-`k` loop bound is the
  data-independent stripe window; there is no data-dependent loop bound, regime
  sub-range, per-parity sub-loop, or per-element `if`/`break`/`continue` anywhere
  in that call graph; and reflection, parity, keep/drop, and the
  derivative-domain decision are arithmetic masks/selects. (F001: inspecting only
  the named public function would let a fixed-loop wrapper delegate divergent
  control flow to a private `_aligned_inner(...)`-style helper and still pass; the
  test now covers the whole hot-path call graph.) Helpers excluded from this
  inspection are limited to pure validation and table-construction helpers **not**
  executed inside the per-`(itrc, j, k)` accumulation path; any such exclusion
  must be explicitly disclosed in the implementation handoff, naming the helper
  and why it is outside the accumulation path.
- **LLVM IR divergence-free inspection (TR45; authorized human decision HD-1).**
  In addition to the source/AST test above, the aligned function's divergence-free
  property **must** be verified at the **LLVM IR** level. Using numba's stable
  IR-extraction API (`wavemaket_stripe_dense_aligned.inspect_llvm()`, or an
  equivalent documented numba mechanism) for the signature(s) the tests exercise,
  compiled with bounds-checking disabled at the repository-pinned
  python/numba/LLVM versions, the test must verify that the LLVM IR of the aligned
  accumulation path contains **no conditional branch whose predicate depends on
  per-element runtime data** — waveform values, `k` keep/drop (bandwidth or
  table-overflow), reflection side, parity, or derivative-domain membership.
  Equivalently, those decisions must appear in the IR as **data flow**
  (`select`/arithmetic), not as control flow. The only conditional branches
  permitted in that IR are those implementing the **data-independent fixed-trip
  loop control** (uniform bounds over `Nc` and `stripe_height`, whose trip counts
  do not depend on runtime data). This is required precisely because it is harder
  to adversarially evade than source/AST inspection: a helper whose name resembles
  arithmetic but whose body branches on data will surface as a data-dependent
  branch in the IR.
  - **Scope of the "compiled-code" exclusion (supersedes the prior TR35 phrasing
    under HD-1).** LLVM IR is machine- and language-independent and is a required
    inspection target at frozen toolchain versions. What remains excluded as
    genuinely fragile is inspection of the **final machine-dependent compiled
    output** (native assembly, PTX, or object bytecode), which is machine-specific
    and is **not** required. The earlier wording that the structural test "is not
    required to inspect numba/LLVM/PTX lowered IR" is **superseded for LLVM IR** by
    HD-1; PTX/native machine code remain excluded.
  - **Toolchain pinning and unavailability.** The criterion is evaluated at the
    repository-pinned python/numba/LLVM versions. If a future toolchain change
    prevents IR extraction, or alters the IR so the test cannot run, the
    implementation must be marked **pending human structural review** rather than
    silently passing, and the toolchain dependency re-confirmed. This is the
    bounded, acknowledged fragility of IR inspection (a version dependency); it does
    not reach the machine-dependent fragility of compiled-bytecode inspection.
  - The contract therefore **does** assert divergence-free structure at the LLVM IR
    level for the pinned toolchain. Whether the eventual hand-written GPU/PTX kernel
    is divergence-free is separate future-work review (Non-Goals), since no GPU
    kernel is in scope here.
- **ULP-boundary detection test** (Amendment 1 AM1-4; strengthened per F004): the
  ULP-boundary mask must be computed **directly from the contract formula** in
  "Aligned-path ULP-boundary exception" (using `R`, `wc.DF`, `wc.df_bw`, and
  `waveform.FT`), **without calling any implementation helper of the aligned
  path**, so the exemption set is an independent oracle rather than implementation
  output. The test must: confirm all non-boundary cells satisfy the standard
  tolerance; confirm ULP-boundary cells contain either `0.0` or the integer-stride
  interpolated value; **assert that the exempt set is not the entire comparison
  domain** (not all cells are exempt) and that **at least one non-boundary cell is
  exercised** for each tested case; and **report the exempt-cell count**. (F004:
  prevents an implementation-coupled mask from marking all differing — or all —
  cells as ULP-boundary, which would make the oracle comparison vacuous.)
- **Clamp-and-mask test (TR33/TR36):** out-of-domain derivative input → exactly
  `0.0`, no `NaN`/`inf`, and no fault with bounds-checking disabled; in-domain
  results unchanged within tolerance versus a control-flow-guarded reference.

Repository-wide test/CI execution is not required; the enforceable repository
QA requirement is the allowed-diff/file-scope restriction plus unskipped targeted
tests.

---

## Benchmark/Demo Requirements

Required at `speed_tests/taylor_time_wavelet_optimized_speed_demo.py`. It must:

- Benchmark `wavemaket`, `wavemaket_stripe_sparse`, `wavemaket_stripe_dense`, and
  `wavemaket_stripe_dense_aligned`.
- Exclude TDI/source-generation work from timed sections; warm up numba
  compilation before timing; report at least median per-call wall time.
- Include `wc.Nt == 2048`, `stripe_height` `2, 3, 4, 5`, `Nc == 3`.
- Include C- and F-contiguous stripe layouts, or report a layout unsupported on a
  measured correctness/compilation failure.
- Validate that benchmarked float64 outputs match the oracle tolerance
  (ULP-boundary cells exempted for the aligned variant) before reporting speed.

**Soft, flagged CPU performance floor (DI-3, TR34).** The benchmark must report,
per `stripe_height ∈ {2,3,4,5}` at the `Nsf = 150` config, the ratio of
`wavemaket_stripe_dense_aligned` median time to `wavemaket_stripe_dense` median
time. The intended floor is **≤ 1.10×** (aligned no more than ~10% slower than
dense). This is **not** an auto-fail and **not** auto-acceptance either (F003): a
measured ratio above the floor does not by itself fail acceptance, but it also may
not be self-accepted by reporting the number alone. If the independently
re-measured aligned/dense median ratio exceeds **1.10×** for any required
`stripe_height`, the implementation is **not automatically acceptable**: the
review handoff must set status `blocked_pending_human` and obtain **explicit human
acceptance of the measured regression** before proceeding. The benchmark must emit
the ratio regardless. There is no hard speed threshold — the hard performance gate
lives on the future GPU implementation (DI-2/DI-3). The reviewer must re-measure
this ratio independently rather than restate the implementer's claim.

---

## QA And Repository Policy

Implementation code must: use numba `njit` for the three functions; stay
nopython-compatible; include specific input annotations and `-> None`; include
concise NumPy-style docstrings for public functions/helpers; use assertions only
for runtime validation; avoid defensive branches for invalid inputs in hot
paths; preserve existing `wavemaket` behavior.

Unless separately approved, the implementation may not introduce or modify:
inline linter suppressions (`# noqa`); type-checker suppressions (`# type: ignore`,
`# pyright: ignore`, broad-type-hiding casts); checker configuration; file
exclusions; test skips/xfails; warning filters; coverage exclusions
(`# pragma: no cover`); CI/pre-commit/ruff/pyrefly/mypy/pytest/coverage config;
broad `Any`-typed public interfaces, broad protocols, dynamic attribute access,
wrapper aliases, or generated-code designations that obscure the contract. Any
future-approved exception must be disclosed explicitly with justification.

The `speed_tests` file may inherit existing repository per-file lint policy for
`speed_tests/*`; this does not relax the implementation file's public
annotations.

---

## Spirit-Over-Letter And Conflict Escalation

- **Tie-breaker.** Where the literal requirement text and the Design Intent
  conflict, the implementer must **not** resolve it silently. An implementation
  that satisfies the letter while defeating a stated Design Intent item is
  **noncompliant by definition**.
- **Escape hatch (mandatory stop).** If satisfying a requirement appears to
  require defeating an intent item, or two requirements appear jointly
  unsatisfiable, the implementer must **stop and emit a Contract Defect Report**
  (what conflicts, with which intent/requirement, and the minimal decision
  needed) instead of choosing a letter-compliant resolution. Reverse-engineering
  an oracle's incidental arithmetic, or relocating control flow to pass a
  structural proxy, are the canonical things this clause forbids.
- **Implementer deliverables.** Alongside the diff, the implementer must produce:
  (1) an **Intent Compliance Note** stating, per Design-Intent item, how the
  implementation advances it (not "tests pass"); and (2) a **Contract Defect
  Report**, or an explicit statement that no conflicts were found.

---

## Joint-Satisfiability Witness

The following single implementation sketch satisfies all requirements
simultaneously, demonstrating the requirement set is not internally
contradictory (the gate that CD-A failed):

> For each `(itrc, j)`: compute `n_ind`, `n_safe = clamp(n_ind, 0, Nfd-2)`,
> `mask_d`, and `mult1 = AT[itrc, j] * mask_d`; compute both reflection-side
> `floor`/`dx` once. Loop `k` over `[nf_start, nf_top]` (fixed trip count): form
> `R*k`, arithmetically select the reflection side, compute padded/clamped read
> indices, interpolate, compute the bandwidth+overflow `mask_k` arithmetically,
> and do `wavelet_stripe[j, k-nf_start, itrc] += mask_k * (coef_y*y1 + coef_z*z1)`
> with parity sign selected arithmetically. ULP-boundary cells (where `R*k` and
> the fastmath product disagree on overflow) are exempt from oracle comparison;
> all reads are in bounds via padding (bandwidth axis) and clamp (derivative
> axis); no data-dependent control flow exists; drops come from `mask_k`/`mask_d`.

This witness is illustrative, not prescriptive: the exact arithmetic is
implementation freedom, subject to the Operative Definitions and Implementation
Variants.

---

## Malicious-Compliance Scenarios (carried into implementation review)

The implementation reviewer must check the implementation against each scenario
and report whether it landed on any path. (Extends the prior contract's MC list;
MC8/MC9 are the failures observed on PR #40.)

- **MC1** — wrapper over `wavemaket`/full-band materialization (forbidden by
  Forbidden Shortcuts).
- **MC2** — `wc.Nf` set equal to stripe width so windowing is never exercised.
- **MC3** — overwrite (`=`) instead of coadd (`+=`).
- **MC4** — nonzero-mask oracle comparison hiding spurious writes.
- **MC5** — skipped/xfail tests satisfying "comparison present" without running.
- **MC6** — coverage parametrization that asserts shape/nonzero only.
- **MC7** — aligned variant keeps ordinary branches and is no faster than dense,
  with the purpose declared "not achievable."
- **MC8 (CD-A′ relocation)** — aligned variant made "branchless" by moving
  branches into data-dependent loop bounds / regime sub-ranges / per-parity
  sub-loops, leaving the inner body `if`-free but still divergent. **Forbidden
  by the anti-relocation clause and the divergence-free structural test.**
- **MC9 (CD-A mirroring)** — aligned variant reverse-engineers the oracle's
  fastmath `(DF/df_bw)*k` to pass the tolerance, defeating divergence-free
  integer-stride arithmetic. **Obviated by the ULP-boundary exception**, which
  removes the incentive; reviewer confirms integer-stride `R*k` is used and the
  fastmath product is absent from the function body.

Any "landed on / uncertain" answer is escalated to the human maintainer and may
not be auto-resolved. Any entry the reviewer lists under "Possible Contract
Defects" blocks auto-approval and routes to the human maintainer.

---

## Verification Methods

- `git diff --name-only` confirms diffs limited to the required new files and
  optional strictly additive `__init__.py` edits.
- Source inspection confirms no `wavemaket`/`wavemaket_direct` calls and no
  full-band intermediates; and confirms the aligned function's divergence-free
  structure (uniform `k`-loop bound; no data-dependent bounds/sub-loops/branches;
  arithmetic masks/selects; clamp-and-mask derivative handling; cross-multiplied
  precondition; integer-stride `R*k`).
- Test inspection confirms tests are active, full-stripe, oracle-based, include
  required coverage, and include the source/AST divergence-free structural test,
  the **LLVM IR divergence-free inspection (TR45/HD-1)**, the ULP-boundary
  detection test, and the clamp-and-mask test.
- LLVM IR inspection (via numba `inspect_llvm()` at the pinned toolchain, bounds-
  checking disabled) confirms the aligned accumulation path has no data-dependent
  conditional branch (keep/drop, reflection, parity, derivative-domain decisions
  appear as `select`/arithmetic, not control flow); final machine-dependent
  compiled output (native/PTX) is out of scope.
- Benchmark inspection or execution confirms all four variants, correctness
  validation before timing, and the reported aligned-vs-dense median ratio;
  reviewer re-measures the ratio independently.
- Review confirms the Intent Compliance Note and Contract Defect Report are
  present.

Running the full repository test suite is outside required acceptance.

---

## Finding Disposition Ledger (consolidation deltas)

Prior dispositions in the superseded `final` ledger and Amendment 1 ledger are
carried forward **except** as changed below.

| Finding | Disposition | Section affected | Revision | Authority | Remaining uncertainty |
|---|---|---|---|---|---|
| CD-A | Resolved | Numerical Tolerances; Implementation Variants; Operative Definitions | Replaced "match oracle exactly + minimize branches" prose with the divergence-free predication definition plus the ULP-boundary exemption that makes it satisfiable. | PR #40 maintainer finding; this consolidation. | Exact integer-stride arithmetic is implementation freedom. |
| CD-B | Resolved | Runtime Validation; Aligned Table Requirements; TR14/TR25 | Deleted the "preserve `wc.df_bw`/`wc.DF`/`wc.Nf`/`wc.Nt`/`wc.dfd`/derivative-grid" storage list; re-expressed the anti-wrapper intent as a prohibited mechanism + must-fail test; prohibited any consistency claim based on re-storing `wc` fields. | PR #40 maintainer note (object has more metadata than the original; `wc` is passed at call time). | None. |
| CD-A′ | Resolved | Implementation Variants — aligned; Forbidden Shortcuts; Operative Definitions | Superseded the Amendment 1 regime-split / per-parity-sub-loop / per-regime-loop-bound recipe with an outcome requirement (divergence-free predication) plus an explicit anti-relocation clause and a divergence-free structural test. | This consolidation; GPU-readiness intent DI-2. | Whether a predicated CPU form beats dense is measured, not assumed (DI-3). |
| TR27–TR31 | **Superseded** | Implementation Variants — aligned | The Amendment 1 mechanism-prescription requirements (no per-k `(DF/df_bw)*k`; two-regime split; per-regime loop bounds; reflection sub-loops) are replaced. TR27 (no per-k `(DF/df_bw)*k`) and TR31 (cross-multiplied precondition) are retained in spirit (folded into the aligned requirements and Runtime Validation). TR28/TR29 (regime split, per-regime loop bounds) are **removed** because they mandate divergence on the GPU target and conflict with DI-2. TR30 (ULP exemption) is retained. | This consolidation. | None. |
| DI-3 floor | Resolved | Benchmark/Demo Requirements; TR34 | Added the soft, flagged aligned-vs-dense median-ratio floor (≤1.10×), independently re-measured, escalated not auto-failed. | Maintainer Step-0 clarification (soft flagged floor). | Exact ratio threshold is a review heuristic. |
| TR33 | Resolved | Implementation Variants — aligned; Required Coverage; Test Requirements | Derivative-domain drop handled by in-kernel clamp-and-mask folded into the amplitude prefactor; control-flow guard removed; no-NaN/no-fault tests required. | Maintainer Step-0 decision (clamp-and-mask). | Caller-side compaction deferred (Non-Goals). |

### PR #42 contract-adversary dispositions (revision r1)

These resolve the eight findings from the PR #42 contract-adversary review of this
file (contract sha256 `de2a2a3f…`). They do not change any frozen Design Intent;
they close loopholes and pin interface details.

| Finding | Class | Disposition | Section affected | Authority |
|---|---|---|---|---|
| F001 | Compliance loophole | Accepted, resolved | Test Requirements (TR35/TR37) | DI-2; PR #42 review |
| F002 | Missing acceptance criterion | Accepted, **resolved** (authorized human decision HD-1) | Test Requirements (TR45); Operative Definitions; Verification Methods | Authorized human decision HD-1 (LLVM IR inspection now required); DI-2 |
| F003 | Blocking ambiguity | Accepted, resolved | Benchmark/Demo (TR34/TR40) | Maintainer Step-0; handoff human-authority |
| F004 | Missing acceptance criterion | Accepted, resolved | Test Requirements (TR39) | DI-4; formula already in contract |
| F005 | Missing interface detail | Accepted, resolved | Supported Waveform Domain (TR41) | Non-Goals; assertions-only policy |
| F006 | Compliance loophole | Accepted, resolved | Forbidden Shortcuts (TR42) | DI-1 |
| F007 | Phantom requirement | Accepted, resolved | Implementation Variants — sparse (TR43) | DI-1; removes unauthorized mechanism |
| F008 | Missing interface detail | Accepted, resolved | Dense Stripe Shape And Indexing (TR44) | numba nopython QA policy; in-place mutation |

F002 is now **fully resolved by authorized human decision HD-1**. The helper-
internal-branch hole it shares with F001 is closed by TR37; source/AST inspection
(TR35) is retained; and HD-1 **adds required LLVM IR inspection** (TR45) as a
harder-to-evade verification of the same divergence-free property. The reviewer's
proposed lowered-IR mandate was correct: the contrary phrasing "no reliance on
fragile compiled-code inspection" was LLM-drafted in the v6 consolidation (a
same-model-family blind spot the cross-family adversary correctly flagged), not a
human decision, and HD-1 supersedes it. The genuinely fragile, still-excluded target
is the final machine-dependent compiled output (native/PTX), not LLVM IR. The
earlier "route the depth question to the human" disposition is closed: the human
decided.

---

## Requirement Traceability Table (changed/added requirements)

Unchanged requirements TR1–TR13, TR15–TR24 retain their meaning from the
superseded `final` contract. Changes and additions:

| Requirement | Authority | Section | Verification | Notes |
|---|---|---|---|---|
| TR14 (revised) | This consolidation; CD-B | Runtime Validation; Aligned Table Requirements | Structural layout tests; identity-wrapper rejection; padding/drop/differential tests. | Aligned table requires layout (padding + valid-region metadata + `R` + predication-safe width); **does not** re-store `wc` fields. |
| TR25 (revised) | This consolidation; CD-B | Aligned Table Requirements | Source/structural inspection of original-grid value reuse. | Original coefficient grid/values reused; metadata-storage clause removed. |
| TR26 (revised) | V3-2; this consolidation | Aligned Table Requirements; Implementation Variants | Drop-behavior test. | Drops come from the multiplicative mask; padding/clamp only for address safety. |
| TR27 (retained, narrowed) | Amendment 1 AM1-2 | Implementation Variants — aligned | Source inspection: no `(DF/df_bw)*k` in the function body. | Integer-stride `R*k` only. |
| TR28, TR29 | — | — | — | **Removed/superseded** (regime split + per-regime loop bounds mandate divergence; conflict with DI-2). |
| TR30 (retained) | Amendment 1 AM1-1 | Numerical Tolerances | ULP-boundary detection test. | ULP-boundary cells exempt. |
| TR31 (retained) | Amendment 1 AM1-3 | Runtime Validation | Source inspection of cross-multiplied assertion. | No second `wc.DF/wc.df_bw` in the njit body. |
| TR32 (new) | This consolidation; DI-2 | Operative Definitions; Implementation Variants — aligned | Divergence-free structural test (TR35). | Uniform iteration domain; masks/selects, not data-dependent control flow. |
| TR33 (new) | Maintainer Step-0 | Implementation Variants — aligned; Required Coverage; Test Requirements | Out-of-domain → 0, no NaN/inf, no fault; in-domain unchanged. | Clamp-and-mask derivative handling, in-kernel. |
| TR34 (new) | Maintainer Step-0; DI-3 | Benchmark/Demo Requirements | Reported and independently re-measured aligned/dense median ratio. | Soft, flagged floor ≤1.10×; escalated, not auto-failed. |
| TR35 (new) | This consolidation | Test Requirements; Verification Methods | Python-source/AST inspection. | The divergence-free structural test itself. |
| TR36 (new) | This consolidation | Test Requirements | No-NaN/no-inf and bounds-check-disabled no-fault assertions on out-of-domain input. | Finiteness invariant for mask-after-compute. |
| TR37 (new) | PR #42 review F001; DI-2 | Test Requirements (TR35); Verification Methods | Source/AST inspection of the public aligned function **plus all hot-path helpers** transitively reachable from the per-`(itrc,j,k)` path; disclosed exclusions limited to validation/table-construction helpers off that path. | Closes the wrapper-delegates-to-branching-helper loophole. |
| TR38 (superseded by TR45/HD-1) | PR #42 review F002; DI-2 | Test Requirements | Was: "source-level only; IR not required." **Superseded** — the human (HD-1) decided LLVM IR inspection is required. | Retained only as the source/AST half of the now-two-level check (TR35 + TR45). |
| TR45 (new) | **Authorized human decision HD-1**; PR #42 review F002; DI-2 | Test Requirements; Operative Definitions; Verification Methods | LLVM IR inspection via numba `inspect_llvm()` at pinned toolchain, bounds-check disabled: no data-dependent conditional branch in the aligned accumulation path; keep/drop, reflection, parity, derivative-domain decisions appear as `select`/arithmetic. Final machine-dependent compiled output (native/PTX) excluded; IR-extraction unavailability ⇒ pending human review. | Adopts the F002 reviewer proposal under human authority; supersedes the LLM-drafted "no fragile compiled-code inspection" wording. |
| TR39 (new) | PR #42 review F004; DI-4 | Test Requirements | ULP-boundary mask computed from the contract formula without implementation helpers; assert not-all-exempt, ≥1 non-boundary cell exercised, report exempt count. | Prevents a vacuous exemption set. |
| TR40 (new) | PR #42 review F003; maintainer Step-0; handoff human-authority | Benchmark/Demo Requirements (TR34) | Re-measured aligned/dense ratio > 1.10× ⇒ status `blocked_pending_human` + explicit human acceptance; reporting the number alone is not acceptance. | Soft floor stays soft (human decides) but is enforceable. |
| TR41 (new) | PR #42 review F005; Supported Waveform Domain; Non-Goals | Supported Waveform Domain | Out-of-supported-domain behavior undefined and untested; no detection/assertion required; docstring discloses it. | Distinct from the in-domain derivative-table drop (TR33). |
| TR42 (new) | PR #42 review F006; DI-1 | Forbidden Shortcuts | No waveform-dependent full-band / `O(wc.Nf)` precompute or per-layer lookup in any component incl. the aligned-table constructor; waveform-independent aligned coefficient table permitted. | Extends narrow-stripe memory intent to constructors/caches. |
| TR43 (new) | PR #42 review F007; DI-1 | Implementation Variants — sparse | Removed the "preserve variable-`k` structure as practical" mechanism prescription; replaced with stripe-only writes, in-place accumulate, no full-band materialization, oracle tolerance. | Phantom/mechanism requirement removed; variable-`k` loop still permitted. |
| TR44 (new) | PR #42 review F008; numba nopython QA policy; in-place mutation requirement | Dense Stripe Shape And Indexing | `wavelet_stripe` must be a writable `ndarray` usable in nopython mode; read-only/array-like undefined and docstring-disclosed; no internal copy. | Pins the array interface contract. |

---

## Unresolved Blockers

- Hard performance acceptance lives on the future GPU implementation, not this
  CPU reference; only the soft flagged floor (TR34) applies here.
- Caller-side derivative-domain compaction is deferred to the GPU caller rewrite
  (Non-Goals); this contract mandates in-kernel clamp-and-mask.
- Non-integer-aligned configurations for the aligned path remain unsupported; the
  function must assert and reject them.
- Exact non-float64 floating tolerances remain unresolved; float64 is required
  for faithful acceptance.
- CI execution for the new tests remains deferred to a future CI policy.
- Cross-machine benchmark reproducibility remains unresolved; the aligned/dense
  ratio is interpreted within a single benchmark environment.
- **Verification depth of the divergence-free criterion (F002) — RESOLVED by
  HD-1.** The human decided (HD-1) that **LLVM IR inspection is required** in
  addition to source/AST inspection (TR45). This is no longer an open blocker. The
  only acknowledged residuals are bounded: (a) IR inspection is pinned to the
  repository toolchain versions and must be re-confirmed if python/numba/LLVM
  change (IR-extraction unavailability ⇒ pending human review, not silent pass);
  and (b) the eventual hand-written GPU/PTX kernel's divergence-freedom is separate
  future-work review, since no GPU kernel is in scope here (Non-Goals).
