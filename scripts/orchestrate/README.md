# Agent-pipeline orchestrator

Launcher that resolves a pipeline role against the model-assignment policy, runs
it with role-appropriate isolation, attests the run, validates the handoff, and
routes blocking findings to the human.

## Files

- `launch-role.sh` — the launcher.
- `pipeline_lib.py` — `resolve` (fail-closed policy resolution) and
  `validate-handoff` (schema check + routing). Requires PyYAML.
- Policy, per-run manifest template, handoff schema, and conventions live in
  `../../.claude/agent-shared/`.

## What it addresses (PR review findings)

- **F1 — fail-closed resolution.** `resolve` checks the per-run manifest against
  the policy and against each harness role's `model:` frontmatter, and aborts
  before running anything on any mismatch or blank — so the in-harness model can't
  silently override the policy and an unavailable model can't slip through.
- **F3 — attested red-team isolation.** The intent red-team runs in an
  inputs-only directory holding only `{contract, diff, intended-outcome}`, under
  `codex exec -s read-only` (no writes, no network). An attestation manifest
  records the resolved model, sandbox mode, network state, input hashes, and
  `no_pr_thread_context`. Isolation is enforced, not aspirational.
- **F5 — handoff validation.** The visible `chirp-agent-report:v2` block is
  validated against `handoff-report.schema.json` (reject-unknown-fields). Any
  blocking finding, `intent_defeat`, `possible_contract_defect`, or
  `blocked_pending_human` status forces **HUMAN REVIEW REQUIRED** and suppresses
  auto-post.

## Usage

```sh
# 1. Fill a per-run manifest (drafter/implementer families, orientation, hashes).
cp .claude/agent-shared/run-manifest.template.yaml /tmp/run.yaml   # then edit

# 2. Materialize the diff under review.
git diff <base>...<head> > /tmp/pr.diff

# 3. Run a role.
CODEX_MODEL=<your-gpt-5.5-id> PIPELINE_PYTHON='mamba run -n gstoch-dev python' \
  scripts/orchestrate/launch-role.sh --role impl-intent-redteam --pr 40 \
    --manifest /tmp/run.yaml \
    --contract .contracts/implementation_contract_taylor_time_wavelet_optimized.md \
    --diff /tmp/pr.diff --intended-outcome /tmp/intent.md
```

- **Codex/GPT roles** run automatically, sandboxed.
- **Harness/Claude roles** print the worktree + run parameters; you launch that
  subagent via the Claude Code Agent tool, save its handoff text to the printed
  report path, then re-run `validate-handoff` on it.

## Isolation model

| Role kind | cwd | sandbox | network |
|---|---|---|---|
| `impl-intent-redteam` | inputs-only (`contract`,`diff`,`intended-outcome`) | `codex -s read-only` | none |
| other Codex roles | disposable git worktree | `codex -s workspace-write` | off |
| harness (Claude) roles | disposable git worktree | run via Agent tool (`isolation: worktree`, `tools:`) | — |

GitHub I/O is the orchestrator's job — agents never read or post PR comments.
`--post` (opt-in) archives the report to the PR; blocking findings suppress it.

## Seams to verify on your install

- `CODEX_MODEL` — set to your codex model id for GPT-5.5.
- `PIPELINE_PYTHON` — must have PyYAML (the `gstoch-dev` env does).
- **Codex network-off default** — `read-only` / `workspace-write` sandboxes block
  network by default; confirm on your codex version (`codex exec --help`) and, if
  needed, pin it via `-c` config.
- **Harness invocation** — Claude subagents are launched via the Agent tool, not a
  CLI flag here; the launcher resolves + stages them and prints the parameters.

## Fail-closed contract

`resolve` exits non-zero (and the launcher aborts) when: the role is missing or
blank in the manifest; the manifest `(family, model, run_via)` differs from the
policy for the run orientation; or a harness role's frontmatter `model:` differs
from the resolved model.
