#!/usr/bin/env python3
"""Resolution and validation helpers for the agent-pipeline launcher.

Subcommands:

  resolve --role R --manifest M --agents-dir D
      Resolve role R from the per-run manifest M against the model-assignment
      policy encoded below. FAILS CLOSED (exit 2) on any blank assignment or any
      mismatch with the policy, and (for harness roles) on any mismatch with the
      agent file's `model:` frontmatter. On success prints a JSON object:
      {family, model, run_via, sandbox_mode, decision_owner}.

  validate-handoff --report FILE --schema S
      Extract the `chirp-agent-report:v2` block from FILE, validate it against
      schema S (required fields, enums, additionalProperties:false), and print a
      routing decision. Exits 3 when a human-review-required condition is present
      (any blocking finding, status blocked_pending_human, or an intent_defeat /
      possible_contract_defect classification).

Requires PyYAML. Point the launcher at an interpreter that has it (e.g.
`PIPELINE_PYTHON='mamba run -n gstoch-dev python'`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - dependency guard
    sys.stderr.write(
        'pipeline_lib: PyYAML is required. Use an interpreter that has it, e.g.\n'
        "  PIPELINE_PYTHON='mamba run -n gstoch-dev python' ...\n"
    )
    sys.exit(4)


# --- Policy: the expected (family, model, run_via, sandbox_mode, decision_owner)
# per role. This mirrors model-assignment-policy.md. The implementation phase is
# fixed while the implementer is Claude Opus; the contract phase is keyed to the
# run orientation. Keep this in sync with the policy doc.
HUMAN = 'human'

IMPL_PHASE = {
    'impl-builder': ('claude', 'opus', 'harness', None, None),
    'impl-repair': ('claude', 'opus', 'harness', None, None),
    'impl-reviewer': ('claude', 'sonnet', 'harness', 'workspace-write', None),
    'impl-intent-redteam': ('gpt', 'gpt-5.5', 'codex', 'read-only', None),
    'impl-repair-verifier': ('gpt', 'gpt-5.5', 'codex', 'workspace-write', None),
    'impl-adjudicator': ('claude', 'opus', 'harness', None, HUMAN),
}

CONTRACT_PHASE = {
    'claude-drafted': {
        'contract-adversary': ('gpt', 'gpt-5.5', 'codex', 'workspace-write', None),
        'contract-reviser': ('claude', 'opus', 'harness', None, None),
        'contract-design-adversary': ('gpt', 'gpt-5.5', 'codex', 'workspace-write', None),
        'contract-approver': ('claude', 'sonnet', 'harness', None, HUMAN),
    },
    'gpt-drafted': {
        'contract-adversary': ('claude', 'opus', 'harness', 'workspace-write', None),
        'contract-reviser': ('gpt', 'gpt-5.5', 'codex', None, None),
        'contract-design-adversary': ('claude', 'sonnet', 'harness', 'workspace-write', None),
        'contract-approver': ('claude', 'opus', 'harness', None, HUMAN),
    },
}
# human-drafted contracts use the same judge orientation as gpt-drafted (Claude judges).
CONTRACT_PHASE['human-drafted'] = CONTRACT_PHASE['gpt-drafted']


def _fail(msg: str) -> None:
    sys.stderr.write(f'FAIL-CLOSED: {msg}\n')
    sys.exit(2)


def _frontmatter_model(agents_dir: Path, role: str) -> str | None:
    path = agents_dir / f'{role}.md'
    if not path.is_file():
        return None
    text = path.read_text(encoding='utf-8')
    m = re.search(r'^model:\s*([A-Za-z0-9._-]+)', text, re.MULTILINE)
    return m.group(1) if m else None


def cmd_resolve(args: argparse.Namespace) -> None:
    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding='utf-8')) or {}
    role = args.role
    orientation = (manifest.get('orientation') or '').strip()
    roles = manifest.get('roles') or {}

    if role not in roles:
        _fail(f'role {role!r} is not present in the manifest.')
    entry = roles[role] or {}
    fam = (entry.get('family') or '').strip()
    model = (entry.get('model') or '').strip()
    run_via = (entry.get('run_via') or '').strip()
    if not fam or not model or not run_via:
        _fail(f'role {role!r} has a blank family/model/run_via in the manifest.')

    # Determine the policy expectation.
    if role in IMPL_PHASE:
        exp = IMPL_PHASE[role]
    else:
        if orientation not in CONTRACT_PHASE:
            _fail(f'orientation {orientation!r} is not one of {sorted(CONTRACT_PHASE)}.')
        if role not in CONTRACT_PHASE[orientation]:
            _fail(f'role {role!r} is not a known pipeline role.')
        exp = CONTRACT_PHASE[orientation][role]
    exp_fam, exp_model, exp_run_via, exp_sandbox, exp_owner = exp

    if (fam, model, run_via) != (exp_fam, exp_model, exp_run_via):
        _fail(
            f'role {role!r} manifest=({fam},{model},{run_via}) '
            f'!= policy=({exp_fam},{exp_model},{exp_run_via}).'
        )

    # Frontmatter cross-check for harness roles: the in-harness model: must match.
    if run_via == 'harness':
        fm = _frontmatter_model(Path(args.agents_dir), role)
        if fm is None:
            _fail(f'could not read model: frontmatter for harness role {role!r}.')
        if fm != model:
            _fail(
                f'role {role!r} frontmatter model {fm!r} != resolved model {model!r} '
                '(frontmatter could silently override the policy).'
            )

    if role in IMPL_PHASE:
        producer_family = ((manifest.get('implementation') or {}).get('implementer_family') or '').strip()
        artifact = 'the implementation diff'
    else:
        producer_family = ((manifest.get('contract') or {}).get('drafter_family') or '').strip()
        artifact = 'the proposed contract'

    print(json.dumps({
        'family': fam,
        'model': model,
        'run_via': run_via,
        'sandbox_mode': exp_sandbox,
        'decision_owner': exp_owner,
        'producer_family': producer_family,
        'judge_family': fam,
        'artifact_under_review': artifact,
        'orientation': orientation,
    }))


def _load_metadata_block(report_path: Path) -> dict:
    text = report_path.read_text(encoding='utf-8')
    # Find the first fenced yaml/json block that declares schema chirp-agent-report:v2.
    for m in re.finditer(r'```(?:yaml|json)?\s*\n(.*?)\n```', text, re.DOTALL):
        body = m.group(1)
        if 'chirp-agent-report:v2' in body:
            try:
                data = yaml.safe_load(body)
            except yaml.YAMLError as exc:
                sys.stderr.write(f'handoff: metadata block is not valid YAML/JSON: {exc}\n')
                sys.exit(3)
            if isinstance(data, dict):
                return data
    sys.stderr.write('handoff: no chirp-agent-report:v2 metadata block found.\n')
    sys.exit(3)


def _check_schema(data: dict, schema: dict, where: str = 'root') -> list[str]:
    """Minimal, dependency-free schema check for our v2 metadata shape."""
    errs: list[str] = []
    props = schema.get('properties', {})
    if schema.get('additionalProperties') is False:
        extra = set(data) - set(props)
        if extra:
            errs.append(f'{where}: unknown field(s) {sorted(extra)} (reject-unknown).')
    errs.extend(
        f'{where}: missing required field {req!r}.'
        for req in schema.get('required', [])
        if req not in data
    )
    for key, spec in props.items():
        if key not in data:
            continue
        val = data[key]
        if 'const' in spec and val != spec['const']:
            errs.append(f"{where}.{key}: must equal {spec['const']!r}.")
        if 'enum' in spec and val not in spec['enum']:
            errs.append(f"{where}.{key}: {val!r} not in {spec['enum']}.")
        if spec.get('type') == 'array' and isinstance(val, list):
            item_spec = spec.get('items', {})
            if item_spec.get('type') == 'object':
                for i, item in enumerate(val):
                    if isinstance(item, dict):
                        errs.extend(_check_schema(item, item_spec, f'{where}.{key}[{i}]'))
    return errs


def cmd_validate_handoff(args: argparse.Namespace) -> None:
    schema = json.loads(Path(args.schema).read_text(encoding='utf-8'))
    data = _load_metadata_block(Path(args.report))
    errs = _check_schema(data, schema)
    if errs:
        sys.stderr.write('handoff: schema validation failed:\n  ' + '\n  '.join(errs) + '\n')
        sys.exit(3)

    findings = data.get('findings', []) or []
    blocking = [f for f in findings if f.get('blocking')]
    escalate = [f for f in findings if f.get('classification') in ('intent_defeat', 'possible_contract_defect')]
    human_required = bool(blocking) or bool(escalate) or data.get('status') == 'blocked_pending_human'

    print(json.dumps({
        'status': data.get('status'),
        'human_signoff': data.get('human_signoff'),
        'blocking_ids': [f.get('id') for f in blocking],
        'escalate_ids': [f.get('id') for f in escalate],
        'human_review_required': human_required,
    }, indent=2))
    if human_required:
        sys.exit(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('resolve')
    r.add_argument('--role', required=True)
    r.add_argument('--manifest', required=True)
    r.add_argument('--agents-dir', required=True)
    r.set_defaults(func=cmd_resolve)

    v = sub.add_parser('validate-handoff')
    v.add_argument('--report', required=True)
    v.add_argument('--schema', required=True)
    v.set_defaults(func=cmd_validate_handoff)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
