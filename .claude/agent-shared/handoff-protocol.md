# Agent PR handoff protocol

When an agent produces a review, revision plan, verification result, or
implementation summary for a PR, it preserves the handoff as an append-only PR
comment (the canonical record). Do not commit agent reports as version-tracked
files unless explicitly requested.

A handoff comment has two parts, **both visible to humans**:

1. A self-contained, human-readable report — the substance.
2. A single collapsed `<details>` block containing a YAML **metadata** record —
   for routing and automation only.

**There is no hidden content.** The YAML is metadata only: it indexes the report.
It must not carry findings, rationale, corrections, limitations, questions, or any
agent-to-agent message that is not already in the visible report. A metadata-only,
schema-validated block has nowhere to smuggle substantive content, so there is no
covert channel to police — the earlier "hidden JSON / superset / parity-check"
machinery is retired.

## Required human-readable content

1. A summary line led by `status · finding-count · max-severity`.
2. Stable finding/task IDs (e.g. `F001`).
3. For each finding: classification (from `conventions.md`), severity/confidence,
   affected section, the problem, the consequence or malicious-compliance example,
   and the exact requested correction.
4. Verification performed, gaps, limitations, and questions for the next
   agent/human. Any escalation, `possible_contract_defect`, or `intent_defeat`
   MUST appear here in prose.
5. PR number, base SHA, head SHA, agent role, timestamp, status.

If the report is long, put detailed findings in a GitHub-rendered `<details>`
block — still visible, never hidden.

## YAML metadata block (visible, in a collapsed `<details>`)

Schema id: `chirp-agent-report:v2`. It is validated against
`.claude/agent-shared/handoff-report.schema.json` with
**`additionalProperties: false` (reject unknown fields)** — the block cannot carry
anything outside the named fields. The `findings` entries here are an INDEX
(`id` + class + severity + confidence + `blocking` flag); the detail lives in the
human-readable report above.

Fields (all metadata; see the schema for exact types):

- `schema`, `pr`, `base_sha`, `head_sha`
- `agent_role`, `resolved_model`, `orientation`
- `handoff_stage`: `single` | `phase1_preliminary` | `phase2_final`
- `inputs_considered`: hashes of the artifacts judged (e.g. `contract_sha256`,
  `diff_sha256`, `audibility_artifact_sha256`, `intended_outcome_sha256`) —
  supports the launcher's input attestation
- `status`: a RECOMMENDATION (`recommend_changes` / `recommend_approve` /
  `blocked_pending_human`) — never a binding decision
- `human_signoff`: `pending` | `approved` | `rejected`. Agents always emit
  `pending`; only a human sets `approved`/`rejected`
- `findings`: list of `{id, classification, severity, confidence, blocking}`
- `verification`: list of `{tool, result}` (`result` kept to a terse status)
- `supersedes_handoff`: optional identifier or URL for a prior handoff comment
  superseded by this one

Example comment body:

````md
<!-- visible, collapsed -->
<details><summary>chirp-agent-report:v2 (metadata)</summary>

```yaml
schema: chirp-agent-report:v2
pr: 43
base_sha: fb8e4d6...
head_sha: 91e7f50...
agent_role: impl-reviewer
resolved_model: gpt-5.5
orientation: claude-drafted
handoff_stage: single
inputs_considered:
  contract_sha256: "…"
  diff_sha256: "…"
status: recommend_changes
human_signoff: pending
findings:
  - {id: F001, classification: intent_defeat, severity: high, confidence: high, blocking: true}
verification:
  - {tool: pytest, result: "85 passed"}
```
</details>
````

## Two-phase handoffs

Most roles emit a single handoff and set `handoff_stage: single`. Roles that
must form an independent assessment before reading an audibility artifact emit
two handoff comments:

1. **Phase 1 preliminary handoff** — set `handoff_stage:
   phase1_preliminary`. The `status` field is the agent's preliminary
   recommendation based only on the Phase 1 inputs. In `inputs_considered`,
   include the contract/diff/intended-outcome hashes that were actually read and
   omit `audibility_artifact_sha256`.
2. **Phase 2 final handoff** — set `handoff_stage: phase2_final`. The `status`
   field is the agent's final recommendation after reading the audibility
   artifact. Include `audibility_artifact_sha256` in `inputs_considered` and set
   `supersedes_handoff` to the Phase 1 comment URL or stable identifier.

The Phase 2 report supersedes the Phase 1 recommendation for routing, but it
must not rewrite or erase the Phase 1 assessment. It may quote, contrast, or
explain changes from the Phase 1 assessment in visible prose. Metadata
`findings` in each handoff index only the findings that appear in that same
visible report.

## Human authority

`status` and `findings` are RECOMMENDATIONS. Contract freeze (approver) and
finding adjudication (adjudicator) are **human** decisions: the agent emits
`human_signoff: pending` and never sets it to `approved`. Any
`possible_contract_defect` or `intent_defeat` forces `status:
blocked_pending_human` and must also be surfaced in the visible prose.

## Intake

An agent receiving a handoff validates the YAML against the schema and confirms
the metadata is consistent with the visible report (every YAML `findings[].id`
appears in the report). For two-phase roles, prefer the `phase2_final` handoff
for routing once present and use `supersedes_handoff` to identify the earlier
`phase1_preliminary` comment. It must not act on anything present only in the
YAML.
