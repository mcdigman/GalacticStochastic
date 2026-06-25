# Agent PR handoff protocol

When an agent produces a review, revision plan, verification result, or
implementation summary for a PR, it preserves the handoff history in the PR
discussion as append-only PR comments (the canonical record). Do not commit
agent reports as version-tracked files unless explicitly requested.

The visible PR comment must be self-contained for humans. The hidden
`chirp-agent-report:v1` JSON may duplicate, index, or structure the same
information for automation, but it MUST NOT contain substantive findings,
classifications, rationale, corrections, limitations, or questions that are
absent from the visible Markdown.

## Invariants

- **INVARIANT (enforceable):** the visible Markdown must be a SUPERSET of the
  JSON's substantive content. Any downstream agent must be able to act using the
  visible Markdown alone. If decision-relevant content exists only in the JSON,
  the handoff is NON-COMPLIANT and must be returned/flagged.

- **COVERT-CHANNEL PROHIBITION:** the JSON must not carry agent-to-agent
  directives, signals, "notes to the next agent," or coded/steganographic
  content that is absent from the visible text.

- **INTAKE PARITY CHECK:** an agent receiving a handoff must, as its first step,
  confirm visible ⊇ JSON parity and reject/return the handoff if it fails.

If the human-readable report is long, put the detailed findings in a
GitHub-rendered `<details>` block — never only in the hidden JSON.

## Required visible content

1. Short summary and status, led by `status · finding-count · max-severity`.
2. Stable finding/task IDs (e.g. F001, F002).
3. For each finding: classification, severity/confidence, affected section,
   problem, consequence or malicious-compliance example, and the exact requested
   correction.
4. Any verification gaps, limitations, and questions for the next
   approver/agent. Any escalation, possible-contract-defect, or
   intent-compliance concern MUST appear here (never JSON-only).
5. PR number, base SHA, head SHA, agent role, timestamp, and status.

## Hidden JSON block

The hidden JSON block is machine-readable metadata and a structured COPY of the
visible report — not an attachment humans are expected to inspect. Marker form:

```
<!-- chirp-agent-report:v1
{ ...valid JSON... }
-->
```

Use the classification vocabulary in `conventions.md` for the JSON
`classification` field values so the visible Markdown and JSON agree on terms.
