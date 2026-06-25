# Contract Amendment 1: Aligned-Path R-Stride Arithmetic and Structural Requirements

## Status and Scope

This document amends
`implementation_contract_taylor_time_wavelet_optimized_final.md` (the final
approved contract) with respect to two interrelated issues discovered during
review of PR #40:

1. The numerical-tolerance section did not carve out cells where wavemaket's
   fastmath arithmetic and integer-stride arithmetic produce different drop
   decisions. This silently forced any compliant implementation to mirror the
   exact fastmath expression in the loop body, making it impossible to lift
   computations out of the loop while still passing tests.
2. The implementation-variants requirement for `wavemaket_stripe_dense_aligned`
   used vague "prioritize minimizing conditional branches" language that could be
   satisfied by keeping all original branches while adding a single dense outer
   loop, defeating the design intent.

This amendment does not change any other section of the final contract. All
finding-disposition and traceability entries from the final contract remain
authoritative except where explicitly superseded below.

---

## Root-Cause Analysis

`wavemaket` is compiled with `@njit(fastmath=True)`. Under fastmath, LLVM is
permitted to reassociate floating-point operations. The inner-loop expression
`(wc.DF / wc.df_bw) * k` is subject to such reassociation: for the repository's
`Nsf = 150` configuration where `wc.DF / wc.df_bw == 100.0` exactly, the
product `(wc.DF / wc.df_bw) * 254` can evaluate to `25399.999999999996` rather
than `25400.0`. The integer-stride expression `R * k = 100 * 254 = 25400` gives
a different floating-point value.

At a table-overflow boundary, this 1-ULP difference shifts `kk =
floor(za - zmid - 0.5)` by 1, flipping the keep-or-drop decision for a cell
whose coefficient contribution is of order `amplitude_source` — far outside the
`atol = 1e-10 * amplitude_source` tolerance. Because the final contract's
tolerance section defines the oracle as wavemaket's fastmath output with no
exception for such cells, any implementation that used integer-stride arithmetic
`R * k` instead of the floating-point expression would fail the oracle
comparison. The only way to pass the tests was to reproduce the fastmath
expression-for-expression, which in turn prevented lifting `zmid`, `kk`, and
`dx` out of the per-k loop body.

A secondary issue: computing `wc.DF / wc.df_bw` a second time within the same
`@njit(fastmath=True)` function body (e.g. in a precondition assertion) creates
a common-subexpression-elimination opportunity that can perturb the loop result.
This forced the cross-multiplied assertion form already present in the
implementation and codified below.

---

## Amended Section: Numerical Tolerances

The existing tolerance paragraph is retained unchanged. The following paragraph
is added immediately after it.

> **Aligned-path ULP-boundary exception.** `wavemaket_stripe_dense_aligned` is
> explicitly permitted to compute `zmid` using integer-stride arithmetic
> `R * k` (or its incrementally advanced form `zmid_0 + R * (k - nf_start)`)
> rather than the floating-point expression `(wc.DF / wc.df_bw) * k`.
>
> When integer-stride arithmetic is used, a cell `(j, k, c)` is a
> **ULP-boundary cell** if
>
> ```text
> int(floor(za - R * k - 0.5)) != int(floor(za - (wc.DF / wc.df_bw) * k - 0.5))
> ```
>
> where `za = waveform.FT[itrc, j] / wc.df_bw` and `R =
> taylor_table_aligned.R`. ULP-boundary cells are exempt from the `atol`/`rtol`
> oracle comparison. For a ULP-boundary cell, the implementation may contribute
> either `0.0` or the interpolated value based on the R-stride arithmetic;
> tests must not fail solely because ULP-boundary cells differ from the wavemaket
> output. For all non-ULP-boundary cells the standard
> `atol = 1e-10 * amplitude_source, rtol = 1e-9` requirement applies in full.
>
> An implementation that instead uses `(wc.DF / wc.df_bw) * k` to reproduce
> wavemaket's fastmath drop decisions at ULP boundaries is also permitted, but is
> not required.

---

## Amended Section: Implementation Variants — `wavemaket_stripe_dense_aligned`

The existing bullet list for this function is replaced in its entirety by the
following.

`wavemaket_stripe_dense_aligned` must:

- Use a fixed-`k` inner loop over the stripe window.
- Use the zero-padded, integer-aligned table representation defined in this
  contract.
- Assert the integer-alignment input preconditions before using the aligned-table
  path. The precondition assertion for the ratio `wc.DF / wc.df_bw ≈ R` **must**
  use the cross-multiplied form
  `abs(wc.DF - R * wc.df_bw) <= 1e-15 * max(1.0, abs(R)) * wc.df_bw` rather
  than `abs(wc.DF / wc.df_bw - R) <= 1e-15 * max(1.0, abs(R))`. Under
  `fastmath=True`, evaluating `wc.DF / wc.df_bw` a second time in the same
  `njit` function body is subject to common-subexpression elimination that can
  perturb the loop's `(DF/df_bw)*k` products; the cross-multiplied form avoids
  this regardless of which `zmid` arithmetic is chosen.
- Compute `zmid` at a per-time-pixel or per-regime boundary and advance it by
  the integer stride `R` per frequency layer. The inner k-loop body must not
  recompute `(wc.DF / wc.df_bw) * k` independently for each iteration.
- Eliminate the per-k `if za < zmid:` reflection branch from the inner loop
  body by splitting the stripe into at most two contiguous k-regimes: a
  **low-k regime** where `zmid <= za` (no reflection required) and a **high-k
  regime** where `zmid > za` (reflection active). The regime boundary
  `k_star = ceil(za * wc.df_bw / wc.DF)`, or an equivalent expression using
  integer stride `R`, separates the two ranges. Each regime is a contiguous
  sub-range of `[nf_start, nf_start + stripe_height)`; the function may iterate
  over each sub-range with a separate loop containing no reflection conditional.
- Within each regime, compute `kk` and `dx` once at the regime entry point and
  advance them by the constant stride `R` (or `-R`) per layer, rather than
  recomputing `floor(za - zmid - 0.5)` for every `k`.
- Implement bandwidth and table-overflow guards as per-regime loop bounds.
  Specifically, clamp each regime's start and end to the intersection of the
  regime's k-range with `[kmin, kmax]` (the per-pixel bandwidth window) and
  with the range of `k` values for which `jj1` and `jj2` remain in bounds. The
  inner loop body within each clamped regime must contain no
  `if (kmin <= k <= kmax) and (0 <= jj1 ...) and (0 <= jj2 ...)` conditional.
- Accumulate into `wavelet_stripe` in place with `+=`.
- Match the oracle stripe slice within the faithful float64 tolerance as defined
  by the amended "Numerical Tolerances" section (ULP-boundary cells exempted
  when integer-stride arithmetic is used).

The previous wording "prioritize minimizing conditional branches in the hot loop
without changing observable behavior" is superseded. The phrase "without changing
observable behavior" implicitly required matching wavemaket's fastmath drop
decisions at ULP boundaries, which precluded integer-stride lift-out; it is
removed.

---

## Amended Section: Required Verification for the Aligned Table — Additional Test Requirement

The existing verification bullet list for the aligned table is retained unchanged.
The following bullet is added:

- A **ULP-boundary detection test** must verify that when the aligned
  implementation uses integer-stride arithmetic `R * k` for `zmid`, the test
  harness can identify and exclude ULP-boundary cells from the oracle comparison.
  The test must confirm that all non-ULP-boundary cells satisfy the standard
  `atol`/`rtol` tolerance and that ULP-boundary cells contain either `0.0` or
  the locally interpolated value (not an arbitrary value). This test is waived
  if the implementation uses `(wc.DF / wc.df_bw) * k` for `zmid` instead of
  integer-stride arithmetic.

---

## Finding Disposition Ledger Additions

| Finding | Disposition | Contract section affected | Exact revision made | Reasoning or authority | Remaining uncertainty |
|---|---|---|---|---|---|
| AM1-1 | Accepted and resolved | Numerical Tolerances | Added ULP-boundary cell exception permitting `R*k` arithmetic for `zmid` in the aligned path; excluded ULP-boundary cells from `atol`/`rtol` oracle comparison. | PR #40 review: tight tolerance forced expression-for-expression fastmath mirroring, defeating lift-out intent. | Exact count of ULP-boundary cells per test case depends on wavelet configuration and source frequency. |
| AM1-2 | Accepted and resolved | Implementation Variants — `wavemaket_stripe_dense_aligned` | Replaced vague "prioritize minimizing conditional branches without changing observable behavior" with explicit structural requirements: regime splitting, per-regime `kk`/`dx` computation, per-regime loop bounds replacing per-k conditionals, prohibition on per-k `(DF/df_bw)*k` recomputation. | PR #40 review: original language allowed full per-k recomputation with a single dense outer loop as compliant. | Exact regime-split threshold expression is implementation freedom subject to the structural requirements. |
| AM1-3 | Accepted and resolved | Implementation Variants — `wavemaket_stripe_dense_aligned`; Aligned Table Requirements | Required cross-multiplied form `abs(wc.DF - R*wc.df_bw) <= tol*wc.df_bw` for the precondition assertion; prohibited `abs(wc.DF/wc.df_bw - R) <= tol` form inside the same `njit` function. | fastmath CSE perturbs loop products when `wc.DF/wc.df_bw` is evaluated twice; already present in PR #40 implementation but not previously mandated by contract. | Moot when integer-stride arithmetic is used for `zmid`, but required regardless. |
| AM1-4 | Accepted and resolved | Required Verification for the Aligned Table | Added ULP-boundary detection test requirement (waived if implementation uses float `zmid`). | Necessary companion to AM1-1: the tolerance exemption is meaningless without a test that verifies non-boundary cells still satisfy the standard tolerance. | None. |

---

## Requirement Traceability Additions

| Requirement | Source of authority | Contract section | Verification method | Dependencies or unresolved questions |
|---|---|---|---|---|
| TR27: Aligned path must not recompute `(wc.DF / wc.df_bw) * k` per loop iteration; `zmid` must be advanced by stride `R` from a per-regime starting value. | Amendment AM1-2. | Implementation Variants — `wavemaket_stripe_dense_aligned` | Source inspection: no `(DF/df_bw)*k` expression inside the k-loop body. | Applies only when integer-stride arithmetic is chosen; float-expression path exempt. |
| TR28: Aligned path must split the stripe into at most two k-regimes separated by the reflection boundary and must contain no per-k `if za < zmid:` branch in the inner loop. | Amendment AM1-2. | Implementation Variants — `wavemaket_stripe_dense_aligned` | Source inspection: two regime loops or equivalent, no inner reflection conditional. | Regime-split threshold expression is implementation freedom. |
| TR29: Within each regime, bandwidth and table-overflow guards must be loop bounds, not per-k conditionals inside the hot loop body. | Amendment AM1-2. | Implementation Variants — `wavemaket_stripe_dense_aligned` | Source inspection: no `if (kmin <= k <= kmax) and ...` inside the k-loop body. | Clamped regime bounds must provably exclude all out-of-bounds `jj1`/`jj2` accesses. |
| TR30: ULP-boundary cells are exempt from `atol`/`rtol` oracle comparison for the aligned path when integer-stride `zmid` is used; standard tolerance applies to all other cells. | Amendment AM1-1. | Numerical Tolerances | ULP-boundary detection test and non-boundary tolerance assertions. | Wavelet configurations where no ULP-boundary cells exist in the tested domain trivially satisfy this requirement. |
| TR31: The precondition assertion for `wc.DF / wc.df_bw ≈ R` inside `wavemaket_stripe_dense_aligned` must use the cross-multiplied form to avoid fastmath CSE perturbation of the loop result. | Amendment AM1-3. | Implementation Variants; Aligned Table Requirements | Source inspection: assertion uses `abs(wc.DF - R*wc.df_bw) <= tol*wc.df_bw`, not `abs(wc.DF/wc.df_bw - R) <= tol`. | None. |
