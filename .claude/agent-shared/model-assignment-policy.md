# Model-Assignment Policy — Contract → Implementation Agent Pipeline

## Purpose

Role → model assignments in this pipeline are **not fixed**. They are derived per
run from **who produced each artifact**, so that the blind-spot-sensitive judges
are always in a different model family than the producer they check. This doc is
the rule plus the lookup tables; read the assignment off it at the start of each
run.

## Families available

- **Claude** (in-harness subagents): Opus, Sonnet, Haiku. Selected via each
  agent file's `model:` frontmatter.
- **GPT-5.5** (via the Codex orchestrator, out of harness). Runs the role's
  prompt body through Codex; the `.claude/agents/*.md` `model:` frontmatter does
  **not** apply to these runs.
- Fable is **not** available.

## Invariant rule (independent of orientation)

1. A blind-spot-sensitive **judge** (contract adversary, implementation intent
   red-team) must be a **different family** than the **producer** of the artifact
   it judges.
2. The **producer** family drafts + revises; the **judging** family
   adversary-reviews (twice) and assists the human approver.
3. No single family is both a major **producer** and a major **judge** of the
   same artifact — no self-checking.
4. **Objective** gates (literal compliance review, repair-closure verification)
   may use a same-family-but-different-tier model **only when** the scarce
   cross-family budget is reserved for the blind-spot-sensitive gates. Keep the
   two implementation review lenses (literal vs intent) in **different families**
   so their blind spots don't overlap.
5. The **human** owns the two irreversible decisions — contract **freeze** and
   implementation **adjudication**. Model agents only assist there.

## The asymmetry (why orientation matters)

Claude offers three tiers (so its judges can be diversified across Opus +
Sonnet); GPT-5.5 is a **single** model. Therefore:

- When **Claude judges** (GPT- or human-drafted artifact): two decorrelated
  judge voices are available → **stronger**.
- When **GPT judges** (Claude-drafted artifact): all cross-family judging rests
  on one model; the **human must actively carry the second-judge role** →
  **weaker**.

**Preference:** have **GPT or the human draft contracts**, and keep **Claude on
the implementation side** (build/repair). That spends Claude's tier-diversity on
judging — where decorrelation matters most — and avoids collapsing all
contract judging onto a single GPT model.

## Parametrization

Inputs: **(contract-drafter family, implementer family)**. The implementer is
Claude Opus under the current setup, so the implementation table is fixed; the
contract table **flips with the drafter**.

### Contract phase — keyed to the contract drafter

| Role | **GPT/human-drafted** (preferred · judge = Claude) | **Claude-drafted** (judge = GPT) |
|---|---|---|
| draft | GPT-5.5 or pure human | Claude (e.g. Opus) |
| contract-adversary (1st pass) | Claude **Opus** | **GPT-5.5** |
| contract-reviser | GPT-5.5 / human (producer side) | Claude **Opus** (producer side) |
| contract-design-adversary (2nd pass) | Claude **Sonnet** | **GPT-5.5** |
| contract-approver (freeze) | **Human** + Opus assist | **Human** + Sonnet assist |
| judge diversity | Opus **and** Sonnet (good) | single GPT model (human carries it) |

Both columns are the same shape — producer family drafts + revises, the **other**
family adversary-reviews twice, human freezes. Only the families swap.

### Implementation phase — keyed to the implementer (= Claude Opus)

| Role | Family / model | Rationale |
|---|---|---|
| impl-builder | Claude **Opus** | Producer of code. |
| impl-repair | Claude **Opus** | Producing family fixes its own code (context). |
| impl-reviewer (literal/structural) | Claude **Sonnet** | Objective compliance; different Claude than the implementer; decorrelated from the GPT red-team. |
| impl-intent-redteam | **GPT-5.5** (Codex, sandboxed) | Critical cross-family gate; intent defeats were what same-family review missed. |
| impl-repair-verifier | **GPT-5.5** (Codex) | Cross-family closure/regression check of Claude's repair. |
| impl-adjudicator | **Human** + Claude (Sonnet/Opus) assist | Human decision point; kept off GPT so GPT does not adjudicate its own reviews. |

This phase does not change with the contract drafter; it is keyed only to the
implementer.

## Per-run procedure

1. Identify who **drafted the contract** → pick the contract-phase column.
2. Implementer is Claude Opus → use the implementation-phase table as-is.
3. For **GPT-5.5** roles, run via the **Codex orchestrator**. The intent
   red-team runs in a **no-network sandbox** with the orchestrator relaying PR
   I/O (it never reads the PR thread or the implementer's narrative; the
   orchestrator posts its findings). Frontmatter `model:` is ignored for Codex
   runs.
4. The **human** performs freeze and adjudication; agents only assist.

## Per-run manifest (run-specific state lives elsewhere)

This file holds only the **stable routing rules**. The concrete role→model
assignment for a specific run is run-specific state and lives in a **per-run
manifest** (`.claude/agent-shared/run-manifest.template.yaml`), not here, so this
policy never goes stale. For each run, fill a manifest with the drafter and
implementer families; the orientation and resolved per-role assignments follow
from the tables above. The launcher reads the manifest, resolves it against this
policy, and **fails closed** if any resolved model/family conflicts with the
policy or with an agent's frontmatter default.

Worked example — contract v6 is **Claude-drafted**, so its manifest selects the
Claude-drafted column (GPT-5.5 contract-adversaries, Claude Opus reviser, human
freeze with a Sonnet assist; the human is the load-bearing second judge because
cross-family contract judging is single-model). Implementer is Claude Opus →
standard implementation table.

## Relationship to agent `model:` frontmatter

Each `.claude/agents/*.md` carries a `model:` used **only** when that role runs
as an in-harness Claude subagent. **This policy overrides the frontmatter per
run.** Roles routed to GPT-5.5 run via Codex and ignore the frontmatter model
entirely. The launcher resolves each role to its policy model and **fails closed**
on any mismatch with the frontmatter default, so the in-harness `model:` can never
silently override the policy.
