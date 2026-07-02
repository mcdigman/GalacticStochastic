#!/usr/bin/env bash
# Launcher for the contract->implementation agent pipeline.
#
# Resolves a role from the per-run manifest against the model-assignment policy and
# FAILS CLOSED on any mismatch, then runs it with isolation appropriate to the role,
# emits an attestation manifest, validates the handoff, and flags blocking findings
# for human review.
#
# Isolation:
#   - impl-intent-redteam (Codex/GPT): runs in a clean inputs-only directory holding
#     ONLY {contract, diff, intended-outcome}, under `codex exec -s read-only`
#     with --ignore-user-config --ignore-rules --ephemeral. Network-off is provided
#     by the Codex read-only sandbox (recorded as assumed_by_sandbox, not probed).
#     It cannot reach the repo, the PR thread, or the implementer's narrative.
#   - other Codex roles: run in a disposable git worktree under
#     `codex exec -s workspace-write`, so they can re-run pytest/pyrefly/ruff
#     without touching the submitted branch.
#   - harness (Claude) roles: this script resolves + sets up the worktree and prints
#     the run parameters; you launch the subagent via the Claude Code Agent tool.
#
# GitHub I/O is the orchestrator's job: agents never reach the PR thread. Posting
# the captured report is opt-in (--post) so a human stays in the loop.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: launch-role.sh --role <role> --manifest <manifest.yaml> [options]
  --role ROLE                 pipeline role (e.g. impl-intent-redteam)
  --manifest FILE             per-run manifest (see ../.claude/agent-shared/run-manifest.template.yaml)
  --pr N                      PR number (for the report + attestation)
  --contract FILE             frozen contract (curated input + hashed)
  --diff FILE                 materialized diff (curated input + hashed)
  --intended-outcome FILE     intended-outcome statement (red-team input + hashed)
  --post                      post the captured report to the PR (default: print the gh command)

Env: CODEX_MODEL (your codex model id for GPT-5.5), PIPELINE_PYTHON (interpreter with PyYAML + jsonschema).
EOF
}

REPO_ROOT="$(git rev-parse --show-toplevel)"
HERE="$REPO_ROOT/scripts/orchestrate"
AGENTS_DIR="$REPO_ROOT/.claude/agents"
SCHEMA="$REPO_ROOT/.claude/agent-shared/handoff-report.schema.json"
PIPELINE_PYTHON="${PIPELINE_PYTHON:-python3}"
# PIPELINE_PYTHON may be a multi-word command (e.g. 'mamba run -n gstoch-dev python');
# split it into an array so each word is passed as a separate argv entry.
read -r -a _PY <<<"$PIPELINE_PYTHON"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.5}"   # SEAM: set to your codex model id for GPT-5.5

ROLE=""; MANIFEST=""; PR=""; CONTRACT=""; DIFF=""; INTENDED=""; POST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) ROLE="$2"; shift 2;;
    --manifest) MANIFEST="$2"; shift 2;;
    --pr) PR="$2"; shift 2;;
    --contract) CONTRACT="$2"; shift 2;;
    --diff) DIFF="$2"; shift 2;;
    --intended-outcome) INTENDED="$2"; shift 2;;
    --post) POST=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 1;;
  esac
done
[[ -n "$ROLE" && -n "$MANIFEST" ]] || { usage; exit 1; }

# 1. Resolve role against the policy. Exits non-zero (fail closed) on any mismatch.
RESOLVED="$("${_PY[@]}" "$HERE/pipeline_lib.py" resolve \
  --role "$ROLE" --manifest "$MANIFEST" --agents-dir "$AGENTS_DIR")"
get() { printf '%s' "$RESOLVED" | "${_PY[@]}" -c "import sys,json;print(json.load(sys.stdin).get('$1') or '')"; }
FAMILY="$(get family)"; MODEL="$(get model)"; RUN_VIA="$(get run_via)"; SANDBOX="$(get sandbox_mode)"
PRODUCER="$(get producer_family)"; JUDGE="$(get judge_family)"
ARTIFACT="$(get artifact_under_review)"; ORIENT="$(get orientation)"
echo ">> resolved $ROLE: family=$FAMILY model=$MODEL run_via=$RUN_VIA sandbox=${SANDBOX:-n/a}"

RUN_ID="${ROLE}-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pipeline-${RUN_ID}.XXXX")"
INPUTS="$RUN_DIR/inputs"; mkdir -p "$INPUTS"
WT=""
cleanup() { [[ -n "$WT" ]] && git worktree remove --force "$WT" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# 2. Curate inputs + hashes (portable: macOS shasum, Linux sha256sum).
sha() {
  [[ -f "$1" ]] || { echo ""; return 0; }
  { shasum -a 256 "$1" 2>/dev/null || sha256sum "$1"; } | awk '{print $1}'
}
CONTRACT_SHA=""; DIFF_SHA=""; INTENDED_SHA=""
[[ -n "$CONTRACT" ]] && { cp "$CONTRACT" "$INPUTS/contract.md"; CONTRACT_SHA="$(sha "$CONTRACT")"; }
[[ -n "$DIFF" ]] && { cp "$DIFF" "$INPUTS/diff.patch"; DIFF_SHA="$(sha "$DIFF")"; }
[[ -n "$INTENDED" ]] && { cp "$INTENDED" "$INPUTS/intended-outcome.md"; INTENDED_SHA="$(sha "$INTENDED")"; }

# 3. Build the role prompt: strip YAML frontmatter, substitute run parameters, prepend input note.
ROLE_FILE="$AGENTS_DIR/$ROLE.md"
[[ -f "$ROLE_FILE" ]] || { echo "no agent file: $ROLE_FILE" >&2; exit 1; }
PROMPT_FILE="$RUN_DIR/prompt.md"
{
  echo "Run parameters: producer_family=$PRODUCER judge_family=$JUDGE artifact=\"$ARTIFACT\" orientation=$ORIENT"
  echo "Curated inputs in your working directory: $(ls -1 "$INPUTS" 2>/dev/null | tr '\n' ' ')"
  echo "---"
  awk 'BEGIN{fm=0} NR==1&&$0=="---"{fm=1;next} fm==1&&$0=="---"{fm=0;next} fm==0{print}' "$ROLE_FILE"
} | sed -e "s|{{producer_family}}|$PRODUCER|g" \
        -e "s|{{judge_family}}|$JUDGE|g" \
        -e "s|{{artifact_under_review}}|$ARTIFACT|g" \
        -e "s|{{run_orientation}}|$ORIENT|g" > "$PROMPT_FILE"

REPORT="$RUN_DIR/report.md"
CODEX_VERSION="$(codex --version 2>/dev/null | head -1 || echo unknown)"
NPRT=false
NET="host-not-guaranteed"
CODEX_INVOCATION=""

# 4. Run with role-appropriate isolation. Codex writes its final message to $REPORT.
if [[ "$RUN_VIA" == "codex" ]]; then
  if [[ "$ROLE" == "impl-intent-redteam" ]]; then
    NPRT=true
    CODEX_CMD=(codex exec -m "$CODEX_MODEL" -s read-only
      --ignore-user-config --ignore-rules --ephemeral --skip-git-repo-check
      -C "$INPUTS" -o "$REPORT" -)
    NET="assumed_by_sandbox:read-only"
    echo ">> codex (ISOLATED): cwd=inputs-only, sandbox=read-only, model=$CODEX_MODEL"
  else
    WT="$RUN_DIR/wt"; git worktree add --detach --quiet "$WT" HEAD
    cp -R "$INPUTS" "$WT/.pipeline-inputs"
    CODEX_CMD=(codex exec -m "$CODEX_MODEL" -s "${SANDBOX:-workspace-write}"
      --ignore-user-config --ignore-rules --ephemeral
      -C "$WT" -o "$REPORT" -)
    NET="assumed_by_sandbox:${SANDBOX:-workspace-write}"
    echo ">> codex: cwd=worktree, sandbox=${SANDBOX:-workspace-write}, model=$CODEX_MODEL"
  fi
  CODEX_INVOCATION="${CODEX_CMD[*]}"
  "${CODEX_CMD[@]}" < "$PROMPT_FILE"
else
  WT="$RUN_DIR/wt"; git worktree add --detach --quiet "$WT" HEAD
  cp -R "$INPUTS" "$WT/.pipeline-inputs"
  echo ">> HARNESS role '$ROLE': launch it via the Claude Code Agent tool with:"
  echo "     worktree:   $WT"
  echo "     inputs:     $WT/.pipeline-inputs"
  echo "     run params: producer_family=$PRODUCER judge_family=$JUDGE artifact='$ARTIFACT' orientation=$ORIENT"
  echo "     prompt:     $PROMPT_FILE   (frontmatter-stripped, substituted)"
  echo "     then save its handoff comment text to: $REPORT  and re-run validate-handoff on it."
fi

# 5. Attestation manifest. network is recorded as assumed_by_sandbox (the Codex
# sandbox mode), not a runtime probe.
ATTEST="$RUN_DIR/attestation.json"
CODEX_VER_FIELD="$([[ "$RUN_VIA" == "codex" ]] && printf '%s' "$CODEX_VERSION" || printf '')"
cat > "$ATTEST" <<EOF
{
  "run_id": "$RUN_ID",
  "pr": "${PR:-}",
  "role": "$ROLE",
  "resolved": {"family": "$FAMILY", "model": "$MODEL", "run_via": "$RUN_VIA", "sandbox_mode": "${SANDBOX:-}"},
  "network": "$NET",
  "no_pr_thread_context": $NPRT,
  "inputs_considered": {"contract_sha256": "$CONTRACT_SHA", "diff_sha256": "$DIFF_SHA", "intended_outcome_sha256": "$INTENDED_SHA"},
  "codex_version": "$CODEX_VER_FIELD",
  "codex_invocation": "$(printf '%s' "$CODEX_INVOCATION" | sed 's/\\/\\\\/g; s/"/\\"/g')",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
echo ">> attestation: $ATTEST"; cat "$ATTEST"

# 6. Validate handoff + route (if a report was captured).
if [[ -s "$REPORT" ]]; then
  echo ">> validating handoff against $SCHEMA"
  set +e
  ROUTE="$("${_PY[@]}" "$HERE/pipeline_lib.py" validate-handoff --report "$REPORT" --schema "$SCHEMA")"
  RC=$?
  set -e
  echo "$ROUTE"
  if [[ $RC -eq 3 ]]; then
    echo "!! HUMAN REVIEW REQUIRED (blocking / intent_defeat / possible_contract_defect). Not auto-posting."
    POST=0
  fi
  if [[ "$POST" == "1" && -n "$PR" ]]; then
    echo ">> posting report to PR #$PR"; gh pr comment "$PR" -F "$REPORT"
  elif [[ -n "$PR" ]]; then
    echo ">> archive with:  gh pr comment $PR -F $REPORT"
  fi
fi
