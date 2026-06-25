# Contract Amendment 1: Aligned-Path Integer-Stride Arithmetic and Structural Requirements

## Status and Scope

This document amends `implementation_contract_taylor_time_wavelet_optimized_final.md`
for `wavemaket_stripe_dense_aligned` only. It is an addendum: do not edit the
final contract file to apply this amendment. All requirements in the final
contract remain authoritative except where this addendum explicitly supersedes a
specific aligned-path requirement.

This amendment is not final approval for implementation.

Authoritative decisions supplied for this revision:

1. Integer-stride `zmid` is required behavior for `wavemaket_stripe_dense_aligned`.
2. The numerical exception applies only to keep/drop decisions.
3. A multiplicative numerical mask is allowed only when implemented so that the
   project compiler is not expected to lower it to a control-flow branch.
4. Only this addendum may be edited; the existing final contract must remain
   unchanged.
5. The alternating sign-flip logic must also be implemented without branching
   control flow inside the aligned per-`k` loop.

This amendment addresses these issues discovered during review of PR #41:

- The prior addendum still allowed a float-expression path that could preserve
  per-`k` recomputation.
- The prior ULP-boundary exception applied to too many floor-index disagreements.
- Boundary detection was not operationally computable from Python-side formulas
  and was not pinned to observed compiled oracle keep/drop behavior.
- The final contract's pairwise comparison requirement did not receive the same
  aligned keep/drop boundary exception as oracle comparison.
- The existing float-expression aligned implementation must be identified as
  non-compliant under the mandatory integer-stride decision.
- `dx` was incorrectly described as advancing by `R`.
- The reflection split had an equality-boundary ambiguity.
- The cross-multiplied precondition did not clearly supersede retained
  division-form bullets in the final contract.
- The inner-loop branch prohibition targeted one syntax shape rather than the
  underlying keep/drop control-flow requirement.
- Boundary-exercising tests must not fail merely because a future compiler
  environment produces no fastmath keep/drop boundary cells.
- Compiled-code evidence for branchless masks is environment-pinned and must be
  treated as a portability risk.
- Source inspection must enumerate common floating-ratio rewrites, not only the
  exact `(wc.DF / wc.df_bw) * k` spelling.
- The alternating sign-flip calculation must be covered by the no-branching
  aligned-loop requirement.

---

## Root-Cause Analysis

`wavemaket` is compiled with `@njit(fastmath=True)`. Under fastmath, LLVM may
reassociate floating-point operations. The inner-loop expression
`(wc.DF / wc.df_bw) * k` can therefore produce a value that differs by roughly
one ULP from the mathematically equivalent integer-stride value `R * k`, where
`R = taylor_table_aligned.R`.

At a table-overflow boundary, that ULP-level difference can shift an interpolation
floor index by one and flip a keep/drop decision. The purpose of this amendment is
to preserve the design intent that `wavemaket_stripe_dense_aligned` lifts
`zmid`, interpolation floor, and interpolation-weight work out of the per-`k`
loop while allowing the limited numerical difference that follows from mandatory
integer-stride arithmetic.

The exception is not a general tolerance relaxation. It applies only when the
compiled oracle path and the independent integer-stride aligned reference differ
in a table-overflow keep/drop decision. It does not apply to interior cells where
both paths keep the contribution.

---

## Amended Section: Numerical Tolerances

The final contract's existing float64 tolerance remains:

```text
atol = 1e-10 * amplitude_source
rtol = 1e-9
```

The following paragraph is added immediately after that tolerance definition and
supersedes any contrary implication that `wavemaket_stripe_dense_aligned` must
match `wavemaket` expression-for-expression at fastmath table-overflow
boundaries.

> **Aligned-path keep/drop boundary exception.**
> `wavemaket_stripe_dense_aligned` must compute `zmid` with integer-stride
> arithmetic using `R = taylor_table_aligned.R`. Acceptable forms include `R * k`
> or an incrementally advanced value initialized at a per-time-pixel or per-regime
> boundary. The aligned implementation must not recompute
> `(wc.DF / wc.df_bw) * k` or an equivalent floating ratio product inside the
> per-`k` loop.
>
> A cell `(j, k, c)` is an **aligned keep/drop boundary cell** only when all of
> the following are true:
>
> - The compiled `wavemaket` oracle and an independent integer-stride aligned
>   reference agree on the derivative-index guard, the bandwidth guard, and the
>   stripe-window membership for `(j, k, c)`.
> - The two paths differ in the table-overflow keep/drop decision for `(j, k, c)`.
> - The table-overflow decision difference is caused solely by the floor-index
>   change induced by using integer-stride `zmid` instead of the oracle's compiled
>   fastmath `zmid` behavior.
>
> For aligned keep/drop boundary cells only, the aligned implementation is exempt
> from the standard `atol`/`rtol` comparison to the oracle. The aligned value must
> equal either `0.0`, when the integer-stride aligned reference drops the cell, or
> the independent integer-stride aligned reference contribution, when that
> reference keeps the cell. Arbitrary values, padding-only artifacts, stale
> prefill values, and values from unrelated cells are noncompliant.
>
> For every other cell, including cells where both the oracle and integer-stride
> reference keep the contribution, the standard `atol`/`rtol` oracle comparison
> applies in full.

Boundary-cell classification must be based on observed compiled oracle keep/drop
behavior plus an independent integer-stride aligned reference. The required
observable classifier is:

- A compiled oracle-decision helper, compiled in the project environment with the
  same fastmath setting and oracle-equivalent expression structure needed to
  reproduce `wavemaket` keep/drop decisions for the tested cells. This helper
  emits derivative-index, bandwidth, stripe-window, and table-overflow keep/drop
  booleans, not just coefficient values.
- An independent integer-stride aligned reference helper that emits the same
  keep/drop booleans and the integer-stride aligned contribution value for kept
  cells. This helper must not call `wavemaket_stripe_dense_aligned` or reuse its
  private loop body as the sole reference.

A test or helper must not classify exempt cells solely by comparing Python-side
formulas such as `floor(za - R * k - 0.5)` and
`floor(za - (wc.DF / wc.df_bw) * k - 0.5)`, because those formulas can yield an
empty boundary set even when the compiled `wavemaket` loop exhibits a keep/drop
flip through fastmath reassociation.

---

## Amended Section: Runtime Validation and Aligned Table Preconditions

The final contract's division-form aligned precondition bullets are superseded
for `wavemaket_stripe_dense_aligned` and for aligned-table structural tests.

The mathematical precondition is unchanged:

```text
2 * wc.Nsf % 3 == 0
R = 2 * wc.Nsf // 3
wc.DF / wc.df_bw ~= R
```

The required assertion and structural-test form is changed to the
cross-multiplied expression:

```text
abs(wc.DF - R * wc.df_bw) <= 1e-15 * max(1.0, abs(R)) * wc.df_bw
```

The aligned implementation must not use this superseded division-form assertion
inside the same `@njit(fastmath=True)` function body:

```text
abs((wc.DF / wc.df_bw) - R) <= 1e-15 * max(1.0, abs(R))
```

This supersession preserves the same intended ratio check while avoiding a second
evaluation of `wc.DF / wc.df_bw` in the compiled aligned function body.

---

## Amended Section: Implementation Variants - `wavemaket_stripe_dense_aligned`

The final contract's bullet list for `wavemaket_stripe_dense_aligned` is replaced
in its entirety by the following.

`wavemaket_stripe_dense_aligned` must:

- Use a fixed-`k` inner loop over the stripe window.
- Use the zero-padded, integer-aligned table representation defined in the final
  contract.
- Assert the aligned input preconditions before using the aligned path, using the
  cross-multiplied ratio form defined in this addendum.
- Compute `zmid` at a per-time-pixel or per-regime boundary and advance it by the
  integer stride `R` per frequency layer. Integer-stride `zmid` is required, not
  optional.
- Keep `(wc.DF / wc.df_bw) * k`, equivalent floating-ratio products, and
  per-iteration recomputation of `floor(...)` out of the aligned per-`k` loop.
  Prohibited floating-ratio products include, at minimum,
  `(wc.DF / wc.df_bw) * k`, `k * (wc.DF / wc.df_bw)`,
  `wc.DF * k / wc.df_bw`, `wc.DF * (k * (1.0 / wc.df_bw))`, and any expression
  whose value is a floating-point function of `k` and `wc.DF / wc.df_bw`.
  An implementation that computes
  `zmid = (wc.DF / wc.df_bw) * k` per `k` inside
  `wavemaket_stripe_dense_aligned`, including the implementation shape reviewed
  in PR #40 at `WaveletWaveforms/taylor_time_wavelet_optimized.py:496`, is
  non-compliant with this amendment even if it matches the prior oracle output.
- Split each stripe/time/channel work range into at most two reflection regimes:
  a low-`k` regime where `R * k <= za` and reflection is not active, and a high-`k`
  regime where `R * k > za` and reflection is active. With integer stride, the
  first high-regime index is:

  ```text
  k_high_start = floor(za / R) + 1
  ```

  The low regime is restricted to `k < k_high_start`; the high regime is
  restricted to `k >= k_high_start`. If `R * k == za`, that layer belongs to the
  low regime.
- Within each regime, compute the starting interpolation base index or indices
  and `dx` once at the regime entry point. The base index or indices, including
  `kk`, `jj1`, and `jj2` or equivalent values, advance by `+R` or `-R` per layer
  as appropriate for the regime. `dx` is constant within a regime because `R` is
  an integer stride; it must not be advanced by `R`.
- Implement reflection, bandwidth, and table-overflow keep/drop decisions without
  a control-flow branch, `continue`, early return, or helper-call gate inside the
  aligned per-`k` loop. Compliant mechanisms are:
  - Per-regime loop bounds that admit only kept cells.
  - A multiplicative numeric validity mask, provided source and compiled-code
    verification show that the mask is not lowered to a conditional branch whose
    predicate depends on reflection, bandwidth, or table-overflow keep/drop
    conditions.
  - An observably equivalent branchless mechanism with the same verification
    evidence.
- Ensure any branchless mask multiplies the contribution value and does not serve
  as the only protection against unsafe memory access. Table reads must remain in
  bounds through loop bounds, the required padded aligned layout, or another
  separately verified in-bounds mechanism.
- Implement the alternating sign flip associated with `(j_ind + k) % 2` without
  branching control flow inside the aligned per-`k` loop. This sign flip is not a
  keep/drop decision, but it is still part of the aligned hot-loop branch-removal
  requirement. Compliant mechanisms include an arithmetic sign factor, a
  branchless parity-to-sign transform, or an observably equivalent mechanism with
  compiled-code evidence that the compiler did not lower the parity decision to a
  conditional branch.
- Accumulate into `wavelet_stripe` in place with `+=`.
- Match the oracle stripe slice within the final contract's faithful float64
  tolerance, except for aligned keep/drop boundary cells as defined in this
  addendum.

The final contract phrase "prioritize minimizing conditional branches in the hot
loop without changing observable behavior" is superseded for
`wavemaket_stripe_dense_aligned`.

---

## Amended Section: Required Verification for the Aligned Table

The final contract's aligned-table verification requirements remain in force
except where this addendum supersedes the ratio-precondition expression. The
following additional verification is required.

- A precondition verification must check the cross-multiplied aligned ratio form
  and must not require the superseded division-form assertion.
- A source inspection must verify that the aligned implementation uses mandatory
  integer-stride `zmid` and contains no per-iteration aligned-loop computation of
  `(wc.DF / wc.df_bw) * k` or an equivalent floating-ratio product, including the
  representative prohibited forms listed in the Implementation Variants section.
- A source inspection must verify that `dx` is initialized at regime entry and is
  constant within each regime, while base indices advance by `+R` or `-R`.
- A boundary test must exercise at least one aligned keep/drop boundary cell for
  an aligned configuration before the exception is used in acceptance. The test
  must identify the boundary by comparing the compiled oracle-decision helper's
  observed keep/drop booleans with an independent integer-stride aligned
  reference. If the compiled project environment produces one or more aligned
  keep/drop boundary cells for the selected configuration, the test case that uses
  the exception must include at least one such cell. If the compiled project
  environment produces no aligned keep/drop boundary cells for the selected
  configuration, the exception is not exercised; the non-empty-boundary
  requirement is vacuously satisfied, and the test must pass by verifying the
  standard `atol`/`rtol` path for all cells rather than by skipping.
- The boundary test must verify that all non-boundary cells satisfy the standard
  `atol`/`rtol` oracle comparison.
- The boundary test must verify that boundary-cell aligned outputs are either
  `0.0` when the independent integer-stride reference drops the cell, or the
  independent integer-stride reference contribution when that reference keeps the
  cell.
- A negative or edge-case verification must ensure the exception is not applied
  to a cell where both the compiled oracle and integer-stride reference keep the
  contribution. Such cells remain subject to the standard oracle comparison.
- A structural or source inspection must verify the reflection split's equality
  convention: `R * k == za` is low-regime, and the first high-regime index is
  `floor(za / R) + 1`.
- Source inspection must reject direct or indirect per-`k` keep/drop control-flow
  gates inside the aligned loop, including split `if` statements, `continue`
  statements, conditional expressions, early returns, and helper calls whose
  purpose is to branch on reflection, bandwidth, or table-overflow keep/drop
  conditions.
- If a multiplicative numeric validity mask is used, compiled-code inspection in
  the project compiler environment must show no conditional branch inside the
  aligned per-`k` loop whose predicate is derived from reflection, bandwidth, or
  table-overflow keep/drop conditions. LLVM `select`, comparison-to-numeric-mask,
  and arithmetic masking are acceptable evidence when they do not introduce such
  a branch.
- If an arithmetic or parity-based sign-flip mechanism is used, compiled-code
  inspection in the project compiler environment must show no conditional branch
  inside the aligned per-`k` loop whose predicate is derived from `(j_ind + k) % 2`
  or an equivalent parity expression.

These tests must provide evidence independent of the aligned implementation logic
where required above. In particular, the boundary-cell oracle/reference
classification must not call into `wavemaket_stripe_dense_aligned` or reuse its
private helper as the sole oracle.

---

## Amended Section: Pairwise Variant Comparison

The final contract's pairwise comparison requirement remains in force for:

- `wavemaket_stripe_sparse` compared with `wavemaket_stripe_dense`.
- All non-boundary cells in comparisons involving
  `wavemaket_stripe_dense_aligned`.

For pairwise comparisons involving `wavemaket_stripe_dense_aligned`, the same
aligned keep/drop boundary exception defined in this addendum applies. At those
boundary cells, tests must not fail solely because the integer-stride aligned path
keeps a cell that the sparse or dense float-expression path drops, or drops a
cell that the sparse or dense float-expression path keeps. The aligned output at
those cells must still satisfy the boundary-cell value rule in the Numerical
Tolerances section: `0.0` for integer-stride reference drops, or the independent
integer-stride reference contribution for integer-stride reference keeps.

This exception does not relax pairwise comparison between `wavemaket_stripe_sparse`
and `wavemaket_stripe_dense`, and it does not exempt any non-boundary cell.

---

## QA and Repository Policy for This Amendment

This addendum does not authorize implementation changes to:

- Inline linter or type-checker suppressions.
- Checker configuration.
- File exclusions.
- Test skips or expected failures.
- Warning filters.
- Coverage exclusions.
- CI or pre-commit configuration.
- Broad public types, casts, protocols, or dynamic access.
- Generated-code designations.

Any such change must be disclosed and separately approved before it can satisfy
this contract.

---

## Finding Disposition Ledger

| Finding | Disposition | Contract section affected | Exact revision made | Reasoning or authority | Remaining uncertainty |
|---|---|---|---|---|---|
| Latest-review F001 | Accepted and resolved | Numerical Tolerances; Implementation Variants; Verification | Removed the float-expression path and made integer-stride `zmid` mandatory. Removed the verification waiver for float-expression implementations. | Human decision 1 says integer-stride `zmid` is required. The review marked this resolved by that decision. | None for this addendum. |
| Latest-review F002 | Accepted and resolved | Numerical Tolerances; Verification | Required boundary classification by a compiled oracle-decision helper plus an independent integer-stride aligned reference. Explicitly rejected Python-formula-only boundary classification. | Review showed the Python formula can yield an empty set even when compiled `wavemaket` exhibits a fastmath keep/drop flip. | Exact helper implementation remains implementation freedom subject to the observable classifier requirements. |
| Latest-review F003 | Accepted and resolved | Numerical Tolerances; Verification | Limited the exception to binary table-overflow keep/drop differences; required boundary values to be `0.0` for integer-stride drops or the independent integer-stride contribution for integer-stride keeps. | Human decision 2 says the exception only applies to keep/drop decisions. The review marked this resolved by that decision. | None. |
| Latest-review F004 | Accepted and resolved | Pairwise Variant Comparison; Verification | Extended the aligned keep/drop boundary exception to pairwise comparisons involving `wavemaket_stripe_dense_aligned`, while preserving full pairwise comparison for sparse-vs-dense and all non-boundary cells. | Review showed the final contract's pairwise comparison would otherwise fail at the same keep/drop boundary cells. | None. |
| Latest-review F005 | Accepted and resolved | Implementation Variants | Stated that an aligned implementation computing per-`k` float-expression `zmid`, including the PR #40 implementation shape, is non-compliant under this amendment. | Human decision 1 requires integer-stride `zmid`; review identified the existing implementation consequence that must be explicit. | None. |
| Latest-review F006 | Accepted and resolved | Numerical Tolerances; Verification | Made the keep/drop exception source-agnostic for aligned integer-stride behavior while retaining the limit to keep/drop boundary cells. | Human decision 1 requires integer-stride; human decision 2 limits the exception to keep/drop decisions. The review marked this resolved. | None. |
| Latest-review F007 | Accepted and resolved with authorized clarification | Implementation Variants; Verification | Replaced syntax-specific branch prohibition with a semantic ban on direct or indirect per-`k` keep/drop control-flow gates. Allowed multiplicative numeric masks only with source and compiled-code evidence that the mask is not lowered to a keep/drop branch. Required separate in-bounds read protection. | Human decision 3 authorizes multiplicative numerical masks under a no-compiler-branch condition. | Compiled-code inspection depends on the project compiler environment. |
| Prior-review F004 | Accepted and resolved | Implementation Variants; Verification | Stated that `dx` is computed once at regime entry and remains constant within the regime; base indices advance by `+R` or `-R`. | Earlier PR #41 review correctly identified `dx += R` as wrong. Integer stride makes the fractional interpolation offset constant within a regime. | None. |
| Prior-review F005 | Accepted and resolved | Implementation Variants; Verification | Defined `k_high_start = floor(za / R) + 1`, low regime `k < k_high_start`, high regime `k >= k_high_start`, and equality in the low regime. | Earlier PR #41 review identified half-open ambiguity. This formula preserves `R * k <= za` in the low regime. | None. |
| Prior-review F006 | Accepted and resolved | Runtime Validation and Aligned Table Preconditions; Verification | Explicitly superseded the final contract's division-form aligned precondition bullets with the cross-multiplied form for aligned implementation assertions and aligned structural tests. | Earlier PR #41 review showed an internal inconsistency. Prior amendment AM1-3 and PR #40 context require the cross-multiplied form. | None. |
| AM1-1 | Accepted and resolved by this revision | Numerical Tolerances | Narrowed the prior ULP exception to table-overflow keep/drop boundary cells and tied it to mandatory integer-stride behavior. | Latest review found the prior AM1-1 only partially resolved. Human decisions 1 and 2 resolve the ambiguity. | None. |
| AM1-2 | Accepted and resolved by this revision | Implementation Variants | Removed the float-expression exemption, corrected `dx`, defined reflection boundaries, and expanded the branch-control-flow requirement. | Latest review found the prior AM1-2 only partially resolved. | None. |
| AM1-3 | Accepted and resolved by this revision | Runtime Validation and Aligned Table Preconditions | Added explicit supersession of the final contract division-form bullets. | Latest review found the prior AM1-3 internally inconsistent with retained final-contract text. | None. |
| AM1-4 | Accepted and resolved by this revision | Verification | Removed the waiver for float-expression implementations and required observed-oracle plus independent-reference boundary testing. | Latest review found the prior AM1-4 test was waived for the loophole path and not independently pinned. | None. |
| Preservation-check F001 | Accepted and resolved | Verification | Changed the boundary-exercising requirement so a non-empty boundary set is required only when the compiled project environment produces one for the selected configuration; otherwise the test must pass by verifying the standard-tolerance path for all cells. | Review showed the fastmath flip is environment-dependent. If no boundary exists, the exception is unused and cannot be exercised. | Whether a particular environment produces boundary cells remains environment-dependent. |
| Preservation-check F002 | Accepted and resolved | Implementation Variants; Verification; Unresolved Blockers | Added an unresolved portability risk for compiled-code mask and sign-flip inspection; stated per-regime loop bounds are immune to this codegen risk. | Review showed compiled-code branch evidence is pinned to the inspected numba/LLVM environment. | Future compiler changes may require reinspection. |
| Preservation-check F003 | Accepted and resolved | Implementation Variants; Verification | Enumerated common prohibited floating-ratio rewrites and required source inspection to check representative forms. | Review showed exact-spelling prohibition made inspection easier to evade. | Source inspection still requires judgment for algebraically equivalent rewrites not listed. |
| Preservation-check F004 | Accepted and resolved with human clarification | Implementation Variants; Verification | Required the `(j_ind + k) % 2` sign flip to be implemented without branching control flow and added compiled-code verification for parity-derived branches. | User clarification says the sign-flip branch issue must be implemented without branching control flow as well. | Compiled-code inspection depends on the project compiler environment. |

---

## Requirement Traceability Table

| Requirement | Source of authority | Contract section | Verification method | Dependencies or unresolved questions |
|---|---|---|---|---|
| AM1-R1: This file is an addendum only; the final contract file must not be edited. | Human decision 4. | Status and Scope | `git diff --name-only` shows only addendum changes for this revision. | None. |
| AM1-R2: `wavemaket_stripe_dense_aligned` must use integer-stride `zmid`. | Human decision 1; original aligned-path lift-out intent. | Numerical Tolerances; Implementation Variants | Source inspection for integer-stride initialization/advancement and absence of per-`k` floating-ratio product. | None. |
| AM1-R3: The numerical exception applies only to table-overflow keep/drop decision differences. | Human decision 2; latest-review F003. | Numerical Tolerances | Boundary test classifies derivative, bandwidth, stripe, and table-overflow decisions separately. | Independent reference helper required. |
| AM1-R4: Boundary classification must use observed compiled oracle keep/drop behavior plus an independent integer-stride reference. | Latest-review F002; necessary consequence of fastmath compiler dependence. | Numerical Tolerances; Verification | Boundary test must not rely solely on Python-side formula comparisons or aligned implementation helpers. | Independent reference details remain implementation freedom. |
| AM1-R5: Non-boundary cells, including both-keep floor-index disagreements, must satisfy standard oracle tolerance. | Human decision 2; final contract tolerance. | Numerical Tolerances; Verification | Full-stripe comparison with boundary mask excluded only for aligned keep/drop boundary cells. | None. |
| AM1-R6: Aligned boundary outputs are limited to `0.0` for integer-stride drops or the independent integer-stride reference contribution for integer-stride keeps. | Latest-review F003; acceptance-criterion requirement. | Numerical Tolerances; Verification | Boundary-cell assertions compare to the independent reference keep/drop result. | None. |
| AM1-R7: The aligned ratio precondition must use the cross-multiplied assertion form. | PR #40 context; AM1-3; F006. | Runtime Validation and Aligned Table Preconditions | Source and structural-test inspection for `abs(wc.DF - R * wc.df_bw) <= ... * wc.df_bw`; absence of superseded division-form assertion in the aligned compiled function. | None. |
| AM1-R8: Reflection regimes must use `k_high_start = floor(za / R) + 1`, with equality in the low regime. | Prior-review F005; necessary consequence of `R * k <= za` low-regime definition. | Implementation Variants; Verification | Boundary/equality test or source inspection showing low/high half-open ranges. | None. |
| AM1-R9: `dx` is constant within a regime; base indices advance by `+R` or `-R`. | Prior-review F004; integer-stride interpolation property. | Implementation Variants; Verification | Source inspection and, where feasible, helper-level unit test over consecutive `k` values. | None. |
| AM1-R10: Inner aligned per-`k` loops must not contain direct or indirect keep/drop control-flow branches. | Original aligned-path lift-out/branch-elimination intent; F007. | Implementation Variants; Verification | Source inspection including helper calls; compiled-code inspection when a mask or equivalent branchless mechanism is used. | Compiler inspection depends on project environment. |
| AM1-R11: Multiplicative numeric masks are allowed only when not lowered to keep/drop control-flow branches. | Human decision 3. | Implementation Variants; Verification | Source inspection plus compiled LLVM or equivalent compiler-output evidence for relevant signature. | Exact inspection tooling remains implementation work. |
| AM1-R12: Branchless masks must not be the only protection against unsafe memory access. | Final contract aligned-padding policy; necessary memory-safety consequence. | Implementation Variants; Verification | Source inspection for in-bounds reads via loop bounds, padded layout, or equivalent mechanism. | None. |
| AM1-R13: QA suppressions, skips, checker/config changes, warning filters, coverage exclusions, CI/pre-commit changes, broad public typing, and generated-code designations are not authorized by this amendment. | Standing repository and QA policy from prompt. | QA and Repository Policy | Diff review of source, tests, and configuration files. | Future approval must be explicit. |
| AM1-R14: Pairwise comparisons involving `wavemaket_stripe_dense_aligned` use the same aligned keep/drop boundary exception as oracle comparisons. | Latest-review F004; necessary consequence of mandatory integer-stride aligned behavior. | Pairwise Variant Comparison | Pairwise tests apply the boundary classifier only to aligned-vs-sparse/dense comparisons and retain full comparison elsewhere. | None. |
| AM1-R15: A per-`k` float-expression aligned implementation is non-compliant even if it matched the older oracle tests. | Human decision 1; latest-review F005. | Implementation Variants | Source inspection rejects per-`k` `zmid = (wc.DF / wc.df_bw) * k` in `wavemaket_stripe_dense_aligned`. | None. |
| AM1-R16: Boundary-exercising tests must not fail solely because the compiled environment produces no aligned keep/drop boundary cells. | Preservation-check F001. | Verification | Test reports zero boundary cells, applies no exception, and verifies standard `atol`/`rtol` for all cells. | Environment-dependent fastmath behavior. |
| AM1-R17: Floating-ratio source inspection must cover representative rewrites, not only exact spelling. | Preservation-check F003; necessary consequence of mandatory integer-stride `zmid`. | Implementation Variants; Verification | Source inspection checks listed forms and equivalent floating-point functions of `k` and `wc.DF / wc.df_bw`. | Algebraically equivalent rewrites may require reviewer judgment. |
| AM1-R18: The alternating sign flip must be implemented without branching control flow in the aligned per-`k` loop. | Human decision 5. | Implementation Variants; Verification | Source inspection plus compiled-code inspection rejects parity-derived conditional branches inside the aligned per-`k` loop. | Compiler inspection depends on project environment. |
| AM1-R19: Compiled-code evidence for branchless masks and sign flips is environment-pinned. | Preservation-check F002. | Unresolved Blockers; Verification | Record inspected compiler environment for such evidence; reinspection is required after compiler changes. | Future numba/LLVM behavior may change. |

---

## Unresolved Blockers

- Inherited unresolved items from the final contract remain unchanged, including
  CI execution policy, objective performance thresholds, exact non-float64
  numerical tolerances, and cross-machine benchmark reproducibility policy.
- Compiled-code evidence that multiplicative masks or parity-based sign flips are
  not lowered to conditional branches is valid only for the inspected project
  compiler environment. A future numba/LLVM/codegen change may require
  reinspection. Per-regime loop bounds that exclude dropped cells before the
  aligned per-`k` loop are not subject to this particular codegen portability
  risk.
- No unresolved blocker remains for the latest PR #41 findings F001 through F007
  after applying the applicable human decisions listed in this addendum.
- No unresolved blocker remains for preservation-check findings F001 through F004
  after applying the sign-flip clarification listed in this addendum.

---

## Change Summary

### Clarifications That Preserve Intent

- Made the addendum explicitly supersede only aligned-path sections and specific
  precondition wording.
- Replaced vague ULP-boundary wording with observable keep/drop boundary behavior.
- Replaced formula-only boundary classification with a compiled oracle-decision
  helper plus independent integer-stride reference requirement.
- Extended the aligned keep/drop boundary exception to pairwise comparisons
  involving `wavemaket_stripe_dense_aligned`.
- Defined low/high reflection regimes and the equality case.
- Clarified that `dx` is constant within a regime while base indices advance.
- Replaced syntax-specific branch wording with a semantic keep/drop control-flow
  requirement.
- Stated the implementation consequence that per-`k` float-expression aligned
  code is non-compliant.
- Clarified that a zero-boundary compiled environment exercises no exception and
  must verify the standard-tolerance path rather than fail or skip.
- Enumerated representative prohibited floating-ratio rewrites.
- Required the alternating sign flip to be branchless in control-flow terms.

### Newly Authorized Requirements

- Integer-stride `zmid` is mandatory for `wavemaket_stripe_dense_aligned`.
- Multiplicative numeric masks are allowed when source and compiler evidence show
  they do not become keep/drop control-flow branches.
- Sign-flip handling must be branchless in aligned-loop control flow.

### Removed or Narrowed Requirements

- Removed the prior permission for a float-expression `zmid` path.
- Removed the waiver of boundary testing for float-expression implementations.
- Narrowed the numerical exception from any floor-index disagreement to
  table-overflow keep/drop decision differences only.
- Removed ungrounded PR40 placeholder ledger rows from the addendum.
- Superseded the retained division-form aligned ratio assertion with the
  cross-multiplied form for the aligned implementation and aligned tests.

### QA-Enforcement Changes

- Added explicit disclosure and separate-approval requirements for suppressions,
  skips, checker/config changes, warning filters, coverage exclusions,
  CI/pre-commit changes, broad typing/dynamic access, and generated-code labels.
- Added compiled-code verification when branchless masking or branchless sign-flip
  arithmetic is used.
- Added an explicit compiler-portability risk for compiled-code branch evidence.

### Out-of-Scope Recommendations

- None added. Performance thresholds, CI policy, and non-float64 tolerances remain
  inherited unresolved items from the final contract unless separately authorized.
