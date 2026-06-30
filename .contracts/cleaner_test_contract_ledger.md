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
