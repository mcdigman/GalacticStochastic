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
- `inputs_considered`: hashes of the artifacts judged (e.g. `contract_sha256`,
  `diff_sha256`) — supports the launcher's input attestation
- `status`: a RECOMMENDATION (`recommend_changes` / `recommend_approve` /
  `blocked_pending_human`) — never a binding decision
- `human_signoff`: `pending` | `approved` | `rejected`. Agents always emit
  `pending`; only a human sets `approved`/`rejected`
- `findings`: list of `{id, classification, severity, confidence, blocking}`
- `verification`: list of `{tool, result}`
- `limitations`: optional short list of strings

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
inputs_considered:
  contract_sha256: "…"
  diff_sha256: "…"
status: recommend_changes
human_signoff: pending
findings:
  - {id: F001, classification: intent_defeat, severity: high, confidence: high, blocking: true}
verification:
  - {tool: pytest, result: "85 passed"}
limitations: []
```
</details>
````

## Human authority

`status` and `findings` are RECOMMENDATIONS. Contract freeze (approver) and
finding adjudication (adjudicator) are **human** decisions: the agent emits
`human_signoff: pending` and never sets it to `approved`. Any
`possible_contract_defect` or `intent_defeat` forces `status:
blocked_pending_human` and must also be surfaced in the visible prose.

## Intake

An agent receiving a handoff validates the YAML against the schema and confirms
the metadata is consistent with the visible report (every YAML `findings[].id`
appears in the report). It must not act on anything present only in the YAML.
