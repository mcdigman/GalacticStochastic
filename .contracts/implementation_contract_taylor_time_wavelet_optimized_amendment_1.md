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

This amendment addresses these issues discovered during review of PR #41:

- The prior addendum still allowed a float-expression path that could preserve
  per-`k` recomputation.
- The prior ULP-boundary exception applied to too many floor-index disagreements.
- Boundary detection was not pinned to observed oracle keep/drop behavior.
- `dx` was incorrectly described as advancing by `R`.
- The reflection split had an equality-boundary ambiguity.
- The cross-multiplied precondition did not clearly supersede retained
  division-form bullets in the final contract.
- The inner-loop branch prohibition targeted one syntax shape rather than the
  underlying keep/drop control-flow requirement.

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
behavior plus an independent integer-stride aligned reference. A test or helper
must not classify exempt cells solely by comparing two Python-side formulas such
as `floor(za - R * k - 0.5)` and
`floor(za - (wc.DF / wc.df_bw) * k - 0.5)`.

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
  `(wc.DF / wc.df_bw) * k` or an equivalent floating-ratio product.
- A source inspection must verify that `dx` is initialized at regime entry and is
  constant within each regime, while base indices advance by `+R` or `-R`.
- A boundary test must exercise at least one aligned keep/drop boundary cell for
  an aligned configuration before the exception is used in acceptance. The test
  must identify the boundary by comparing observed compiled oracle keep/drop
  behavior with an independent integer-stride aligned reference.
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

These tests must provide evidence independent of the aligned implementation logic
where required above. In particular, the boundary-cell oracle/reference
classification must not call into `wavemaket_stripe_dense_aligned` or reuse its
private helper as the sole oracle.

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
| F001 | Accepted and resolved | Numerical Tolerances; Implementation Variants; Verification | Removed the float-expression path and made integer-stride `zmid` mandatory. Removed the verification waiver for float-expression implementations. | Human decision 1 says integer-stride `zmid` is required. PR #41 review showed the float path preserved the original loophole. | None for this addendum. |
| F002 | Accepted and resolved | Numerical Tolerances; Verification | Replaced broad "ULP-boundary cell" language with "aligned keep/drop boundary cell" limited to table-overflow keep/drop decision differences. Required standard oracle tolerance for all other cells, including both-keep cells. | Human decision 2 says the exception only applies to keep/drop decisions. | Exact fixture construction remains implementation work. |
| F003 | Accepted and resolved | Numerical Tolerances; Verification | Required boundary classification from observed compiled oracle keep/drop behavior plus an independent integer-stride aligned reference. Prohibited Python formula comparison as the sole classifier and required at least one exercised boundary cell before using the exception. | Review finding showed formula-only detection was compiler/context dependent. | The independent reference helper's implementation details are not fixed. |
| F004 | Accepted and resolved | Implementation Variants; Verification | Stated that `dx` is computed once at regime entry and remains constant within the regime; base indices advance by `+R` or `-R`. | Review finding correctly identified `dx += R` as wrong. Integer stride makes the fractional interpolation offset constant within a regime. | None. |
| F005 | Accepted and resolved | Implementation Variants; Verification | Defined `k_high_start = floor(za / R) + 1`, low regime `k < k_high_start`, high regime `k >= k_high_start`, and equality in the low regime. | Review finding identified half-open ambiguity. This formula preserves `R * k <= za` in the low regime. | None. |
| F006 | Accepted and resolved | Runtime Validation and Aligned Table Preconditions; Verification | Explicitly superseded the final contract's division-form aligned precondition bullets with the cross-multiplied form for aligned implementation assertions and aligned structural tests. | Review finding showed an internal inconsistency. Prior amendment AM1-3 and PR #40 context require the cross-multiplied form. | None. |
| F007 | Accepted and resolved with authorized clarification | Implementation Variants; Verification | Replaced syntax-specific branch prohibition with a semantic ban on direct or indirect per-`k` keep/drop control-flow gates. Allowed multiplicative numeric masks only with source and compiled-code evidence that the mask is not lowered to a keep/drop branch. | Human decision 3 authorizes multiplicative numerical masks under a no-compiler-branch condition. Review finding showed the prior text only patched one conditional shape. | Compiled-code inspection depends on the project compiler environment. |
| AM1-1 | Accepted and resolved by this revision | Numerical Tolerances | Narrowed the prior ULP exception to table-overflow keep/drop boundary cells and tied it to mandatory integer-stride behavior. | Latest review found the prior AM1-1 only partially resolved. Human decisions 1 and 2 resolve the ambiguity. | None. |
| AM1-2 | Accepted and resolved by this revision | Implementation Variants | Removed the float-expression exemption, corrected `dx`, defined reflection boundaries, and expanded the branch-control-flow requirement. | Latest review found the prior AM1-2 only partially resolved. | None. |
| AM1-3 | Accepted and resolved by this revision | Runtime Validation and Aligned Table Preconditions | Added explicit supersession of the final contract division-form bullets. | Latest review found the prior AM1-3 internally inconsistent with retained final-contract text. | None. |
| AM1-4 | Accepted and resolved by this revision | Verification | Removed the waiver for float-expression implementations and required observed-oracle plus independent-reference boundary testing. | Latest review found the prior AM1-4 test was waived for the loophole path and not independently pinned. | None. |
| PR40-F001 | Deferred as explicitly out of scope for this addendum | None | No new language added. | Latest review states this was superseded by an authoritative owner decision in the last PR #40 comment. | None for PR #41 amendment. |
| PR40-F002 | Requires human or authoritative resolution if still in scope | Unresolved Blockers | No language added because the latest review says PR #41 did not address it but does not supply the underlying finding detail in this revision request. | The revision agent cannot resolve an unspecified prior finding without the finding text or an authority decision. | Exact PR #40 F002 content and intended scope are unknown from the supplied material. |

---

## Requirement Traceability Table

| Requirement | Source of authority | Contract section | Verification method | Dependencies or unresolved questions |
|---|---|---|---|---|
| AM1-R1: This file is an addendum only; the final contract file must not be edited. | Human decision 4. | Status and Scope | `git diff --name-only` shows only addendum changes for this revision. | None. |
| AM1-R2: `wavemaket_stripe_dense_aligned` must use integer-stride `zmid`. | Human decision 1; original aligned-path lift-out intent. | Numerical Tolerances; Implementation Variants | Source inspection for integer-stride initialization/advancement and absence of per-`k` floating-ratio product. | None. |
| AM1-R3: The numerical exception applies only to table-overflow keep/drop decision differences. | Human decision 2; F002. | Numerical Tolerances | Boundary test classifies derivative, bandwidth, stripe, and table-overflow decisions separately. | Independent reference helper required. |
| AM1-R4: Boundary classification must use observed compiled oracle keep/drop behavior plus an independent integer-stride reference. | F003; necessary consequence of fastmath compiler dependence. | Numerical Tolerances; Verification | Boundary test must not rely solely on Python-side formula comparisons or aligned implementation helpers. | Independent reference details remain implementation freedom. |
| AM1-R5: Non-boundary cells, including both-keep floor-index disagreements, must satisfy standard oracle tolerance. | Human decision 2; final contract tolerance. | Numerical Tolerances; Verification | Full-stripe comparison with boundary mask excluded only for aligned keep/drop boundary cells. | None. |
| AM1-R6: Aligned boundary outputs are limited to `0.0` for integer-stride drops or the independent integer-stride reference contribution for integer-stride keeps. | F002; acceptance-criterion requirement. | Numerical Tolerances; Verification | Boundary-cell assertions compare to the independent reference keep/drop result. | None. |
| AM1-R7: The aligned ratio precondition must use the cross-multiplied assertion form. | PR #40 context; AM1-3; F006. | Runtime Validation and Aligned Table Preconditions | Source and structural-test inspection for `abs(wc.DF - R * wc.df_bw) <= ... * wc.df_bw`; absence of superseded division-form assertion in the aligned compiled function. | None. |
| AM1-R8: Reflection regimes must use `k_high_start = floor(za / R) + 1`, with equality in the low regime. | F005; necessary consequence of `R * k <= za` low-regime definition. | Implementation Variants; Verification | Boundary/equality test or source inspection showing low/high half-open ranges. | None. |
| AM1-R9: `dx` is constant within a regime; base indices advance by `+R` or `-R`. | F004; integer-stride interpolation property. | Implementation Variants; Verification | Source inspection and, where feasible, helper-level unit test over consecutive `k` values. | None. |
| AM1-R10: Inner aligned per-`k` loops must not contain direct or indirect keep/drop control-flow branches. | Original aligned-path lift-out/branch-elimination intent; F007. | Implementation Variants; Verification | Source inspection including helper calls; compiled-code inspection when a mask or equivalent branchless mechanism is used. | Compiler inspection depends on project environment. |
| AM1-R11: Multiplicative numeric masks are allowed only when not lowered to keep/drop control-flow branches. | Human decision 3. | Implementation Variants; Verification | Source inspection plus compiled LLVM or equivalent compiler-output evidence for relevant signature. | Exact inspection tooling remains implementation work. |
| AM1-R12: Branchless masks must not be the only protection against unsafe memory access. | Final contract aligned-padding policy; necessary memory-safety consequence. | Implementation Variants; Verification | Source inspection for in-bounds reads via loop bounds, padded layout, or equivalent mechanism. | None. |
| AM1-R13: QA suppressions, skips, checker/config changes, warning filters, coverage exclusions, CI/pre-commit changes, broad public typing, and generated-code designations are not authorized by this amendment. | Standing repository and QA policy from prompt. | QA and Repository Policy | Diff review of source, tests, and configuration files. | Future approval must be explicit. |

---

## Unresolved Blockers

- PR40-F002 remains unresolved for this addendum because the latest review states
  it was not addressed by PR #41 but does not provide the underlying finding text
  or a human decision.
- Inherited unresolved items from the final contract remain unchanged, including
  CI execution policy, objective performance thresholds, exact non-float64
  numerical tolerances, and cross-machine benchmark reproducibility policy.
- No unresolved blocker remains for latest PR #41 findings F001 through F007 after
  applying the four human decisions listed in this addendum.

---

## Change Summary

### Clarifications That Preserve Intent

- Made the addendum explicitly supersede only aligned-path sections and specific
  precondition wording.
- Replaced vague ULP-boundary wording with observable keep/drop boundary behavior.
- Defined low/high reflection regimes and the equality case.
- Clarified that `dx` is constant within a regime while base indices advance.
- Replaced syntax-specific branch wording with a semantic keep/drop control-flow
  requirement.

### Newly Authorized Requirements

- Integer-stride `zmid` is mandatory for `wavemaket_stripe_dense_aligned`.
- Multiplicative numeric masks are allowed when source and compiler evidence show
  they do not become keep/drop control-flow branches.

### Removed or Narrowed Requirements

- Removed the prior permission for a float-expression `zmid` path.
- Removed the waiver of boundary testing for float-expression implementations.
- Narrowed the numerical exception from any floor-index disagreement to
  table-overflow keep/drop decision differences only.
- Superseded the retained division-form aligned ratio assertion with the
  cross-multiplied form for the aligned implementation and aligned tests.

### QA-Enforcement Changes

- Added explicit disclosure and separate-approval requirements for suppressions,
  skips, checker/config changes, warning filters, coverage exclusions,
  CI/pre-commit changes, broad typing/dynamic access, and generated-code labels.
- Added compiled-code verification when branchless masking is used.

### Out-of-Scope Recommendations

- None added. Performance thresholds, CI policy, and non-float64 tolerances remain
  inherited unresolved items from the final contract unless separately authorized.
