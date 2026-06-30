## Finding Disposition Ledger

Prior dispositions in the superseded `final` ledger and Amendment 1 ledger are
carried forward **except** as changed below.

| Source review pass or handoff | Finding identifier | Original classification | Original severity/confidence | Blocking flag | Disposition | Contract section affected | Exact revision made | Reasoning or supporting authority | Remaining uncertainty |
|---|---|---|---|---|---|---|---|---|---|
| PR #40 consolidation delta | CD-A | Under-specification / possible contract defect | Not recorded in split source | Blocking for auto-approval | Resolved | Numerical Tolerances; Implementation Variants; Operative Definitions | Replaced "match oracle exactly + minimize branches" prose with the divergence-free predication definition plus the ULP-boundary exemption that makes it satisfiable. | PR #40 maintainer finding; this consolidation. | Exact integer-stride arithmetic is implementation freedom. |
| PR #40 consolidation delta | CD-B | Over-specification / possible contract defect | Not recorded in split source | Blocking for auto-approval | Resolved | Runtime Validation; Aligned Table Requirements; TR14/TR25 | Deleted the "preserve `wc.df_bw`/`wc.DF`/`wc.Nf`/`wc.Nt`/`wc.dfd`/derivative-grid" storage list; re-expressed the anti-wrapper intent as a prohibited mechanism + must-fail test; prohibited any consistency claim based on re-storing `wc` fields. | PR #40 maintainer note (object has more metadata than the original; `wc` is passed at call time). | None. |
| Amendment 1 consolidation delta | CD-A′ | Recurrence of under-specification / mechanism-prescription defect | Not recorded in split source | Blocking for auto-approval | Resolved | Implementation Variants — aligned; Forbidden Shortcuts; Operative Definitions | Superseded the Amendment 1 regime-split / per-parity-sub-loop / per-regime-loop-bound recipe with an outcome requirement (divergence-free predication) plus an explicit anti-relocation clause and a divergence-free structural test. | This consolidation; GPU-readiness intent DI-2. | Whether a predicated CPU form beats dense is measured, not assumed (DI-3). |
| Amendment 1 consolidation delta | TR27–TR31 | Mechanism-prescription requirements requiring consolidation | Not recorded in split source | Blocking where conflicting with DI-2 | Superseded | Implementation Variants — aligned | The Amendment 1 mechanism-prescription requirements (no per-k `(DF/df_bw)*k`; two-regime split; per-regime loop bounds; reflection sub-loops) are replaced. TR27 (no per-k `(DF/df_bw)*k`) and TR31 (cross-multiplied precondition) are retained in spirit (folded into the aligned requirements and Runtime Validation). TR28/TR29 (regime split, per-regime loop bounds) are **removed** because they mandate divergence on the GPU target and conflict with DI-2. TR30 (ULP exemption) is retained. | This consolidation. | None. |
| Maintainer Step-0 clarification | DI-3 floor | Performance-gate clarification | Not recorded in split source | Non-auto-fail human-escalation trigger | Resolved | Benchmark/Demo Requirements; TR34 | Added the soft, flagged aligned-vs-dense median-ratio floor (≤1.10×), independently re-measured, escalated not auto-failed. | Maintainer Step-0 clarification (soft flagged floor). | Exact ratio threshold is a review heuristic. |
| Maintainer Step-0 decision | TR33 | Derivative-domain handling decision | Not recorded in split source | Blocking for aligned-path behavior | Resolved | Implementation Variants — aligned; Required Coverage; Test Requirements | Derivative-domain drop handled by in-kernel clamp-and-mask folded into the amplitude prefactor; control-flow guard removed; no-NaN/no-fault tests required. | Maintainer Step-0 decision (clamp-and-mask). | Caller-side compaction deferred (Non-Goals). |

---

## Expanded Requirement Traceability Table

Unchanged requirements TR1–TR13 and TR15–TR24 retain their meaning from the
superseded `final` contract. This reconciliation does not invent new
requirement identifiers in `cleaner_test_contract.md`; it records the available
traceability for the changed, added, retained, and superseded identifiers already
present in the split source.

| Requirement identifier | Authority identifier | Finding IDs associated with this requirement | Contract section | Verification method | Dependencies or unresolved questions |
|---|---|---|---|---|---|
| DI-1 | Current consolidation | None recorded in consolidation deltas | Design Intent; Required Files; Public API; Dense Stripe Shape And Indexing; Forbidden Shortcuts | Source inspection; diff scope check; full-stripe oracle tests; no full-band intermediate inspection. | None recorded. |
| DI-2 | Current consolidation | CD-A; CD-A′; MC8; MC9 | Design Intent; Operative Definitions; Implementation Variants — aligned; Forbidden Shortcuts; Test Requirements | Divergence-free structural test; source inspection for uniform `k` loop, masks/selects, anti-relocation, integer stride, and no fastmath product in the aligned body. | Future GPU implementation remains out of scope. |
| DI-3 | Maintainer Step-0 clarification | DI-3 floor | Design Intent; Benchmark/Demo Requirements | Benchmark reports aligned/dense median ratio; reviewer independently re-measures and escalates ratios above the soft floor. | Exact ratio threshold is a review heuristic; no hard CPU speed threshold applies. |
| DI-4 | Current consolidation | CD-A; MC9 | Design Intent; Numerical Tolerances; Scientific And Mathematical Basis | Full-stripe oracle comparisons with faithful float64 tolerance; ULP-boundary detection and exemption for aligned path. | Exact non-float64 tolerances remain unresolved. |
| TR1–TR13, TR15–TR24 (carried forward) | Superseded `final` contract | None recorded in consolidation deltas | Current contract body; prior `final` traceability not reproduced in this split artifact | Current Verification Methods plus retained prior verification intent. | Full per-requirement legacy traceability is not present in the split source. |
| TR14 (revised) | This consolidation; CD-B | CD-B | Runtime Validation; Aligned Table Requirements | Structural layout tests; identity-wrapper rejection; padding/drop/differential tests. | Aligned table requires layout (padding + valid-region metadata + `R` + predication-safe width); **does not** re-store `wc` fields. |
| TR25 (revised) | This consolidation; CD-B | CD-B | Aligned Table Requirements | Source/structural inspection of original-grid value reuse. | Original coefficient grid/values reused; metadata-storage clause removed. |
| TR26 (revised) | V3-2; this consolidation | CD-B | Aligned Table Requirements; Implementation Variants | Drop-behavior test. | Drops come from the multiplicative mask; padding/clamp only for address safety. |
| TR27 (retained, narrowed) | Amendment 1 AM1-2 | TR27–TR31; MC9 | Implementation Variants — aligned | Source inspection: no `(DF/df_bw)*k` in the function body. | Integer-stride `R*k` only. |
| TR28, TR29 (removed/superseded) | Amendment 1; this consolidation | CD-A′; TR27–TR31; MC8 | Implementation Variants — aligned | Review confirms the regime-split and per-regime-loop-bound requirements are absent from the operative aligned-path requirements. | Removed/superseded because regime split and per-regime loop bounds mandate divergence and conflict with DI-2. |
| TR30 (retained) | Amendment 1 AM1-1 | CD-A; TR27–TR31; MC9 | Numerical Tolerances | ULP-boundary detection test. | ULP-boundary cells exempt. |
| TR31 (retained) | Amendment 1 AM1-3 | TR27–TR31; MC9 | Runtime Validation | Source inspection of cross-multiplied assertion. | No second `wc.DF/wc.df_bw` in the njit body. |
| TR32 (new) | This consolidation; DI-2 | CD-A; CD-A′; MC8 | Operative Definitions; Implementation Variants — aligned | Divergence-free structural test (TR35). | Uniform iteration domain; masks/selects, not data-dependent control flow. |
| TR33 (new) | Maintainer Step-0 | TR33 | Implementation Variants — aligned; Required Coverage; Test Requirements | Out-of-domain → 0, no NaN/inf, no fault; in-domain unchanged. | Clamp-and-mask derivative handling, in-kernel; caller-side compaction deferred. |
| TR34 (new) | Maintainer Step-0; DI-3 | DI-3 floor | Benchmark/Demo Requirements | Reported and independently re-measured aligned/dense median ratio. | Soft, flagged floor ≤1.10×; escalated, not auto-failed. |
| TR35 (new) | This consolidation | CD-A′; MC8 | Test Requirements; Verification Methods | Python-source/AST inspection. | The divergence-free structural test itself. |
| TR36 (new) | This consolidation | TR33 | Test Requirements | No-NaN/no-inf and bounds-check-disabled no-fault assertions on out-of-domain input. | Finiteness invariant for mask-after-compute. |

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
- Full per-requirement traceability for carried-forward TR1–TR13 and TR15–TR24
  is not present in the split source; the artifact records the carried-forward
  statement rather than reconstructing prior ledgers.

---

## Change Summary

### Clarifications That Preserve Intent

- Reconciled the ledger table to include source review pass or handoff,
  original classification, severity/confidence, blocking flag, exact revision,
  reasoning/supporting authority, and remaining uncertainty.
- Reconciled the traceability table to include requirement identifier, authority
  identifier, finding IDs, contract section, verification method, and
  dependencies or unresolved questions.
- Added Design Intent traceability rows for DI-1 through DI-4 because the
  contract states Design Intent is normative.
- Preserved the carried-forward statement for TR1–TR13 and TR15–TR24 instead of
  inventing new identifiers or reconstructing unavailable prior ledgers.

### Newly Authorized Requirements

- None added by this traceability-artifact reconciliation.

### Removed Or Narrowed Requirements

- None removed or narrowed by this traceability-artifact reconciliation.
- The artifact records that TR28 and TR29 were already removed/superseded by the
  consolidation because they conflicted with DI-2.

### QA-Enforcement Changes

- None added by this traceability-artifact reconciliation.

### Out-Of-Scope Recommendations

- If exhaustive one-row-per-requirement traceability is required for TR1–TR13
  and TR15–TR24, recover the superseded `final` ledger and reconcile it in a
  separate authorized pass; this pass does not reconstruct missing history.
