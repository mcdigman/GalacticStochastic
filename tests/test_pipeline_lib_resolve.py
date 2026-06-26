"""Fail-closed tests for the agent-pipeline resolver (scripts/orchestrate/pipeline_lib.py).

Run ONLY this file (the full repository suite has slow/failing cases)::

    mamba run -n gstoch-dev pytest tests/test_pipeline_lib_resolve.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / 'scripts' / 'orchestrate' / 'pipeline_lib.py'
AGENTS_DIR = REPO_ROOT / '.claude' / 'agents'

_SPEC = importlib.util.spec_from_file_location('pipeline_lib', PIPELINE)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    _MSG = f'cannot load {PIPELINE}'
    raise RuntimeError(_MSG)
pipeline_lib = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pipeline_lib)

VALID_IMPL = """
orientation: claude-drafted
contract: {drafter_family: claude}
implementation: {implementer_family: claude}
roles:
  impl-reviewer: {family: claude, model: sonnet, run_via: harness}
"""

BLANK_ENTRY = """
orientation: claude-drafted
roles:
  impl-reviewer: {family: "", model: "", run_via: ""}
"""

MODEL_MISMATCH = """
orientation: claude-drafted
implementation: {implementer_family: claude}
roles:
  impl-reviewer: {family: claude, model: opus, run_via: harness}
"""

ORIENTATION_CONTRADICTION = """
orientation: claude-drafted
contract: {drafter_family: gpt}
roles:
  contract-adversary: {family: gpt, model: gpt-5.5, run_via: codex}
"""

NON_CLAUDE_IMPLEMENTER = """
contract: {drafter_family: claude}
implementation: {implementer_family: gpt}
roles:
  impl-reviewer: {family: claude, model: sonnet, run_via: harness}
"""


def _ns(manifest: Path, role: str, agents_dir: Path = AGENTS_DIR) -> argparse.Namespace:
    return argparse.Namespace(role=role, manifest=str(manifest), agents_dir=str(agents_dir))


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / 'manifest.yaml'
    path.write_text(text, encoding='utf-8')
    return path


def test_valid_impl_role_resolves(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pipeline_lib.cmd_resolve(_ns(_write(tmp_path, VALID_IMPL), 'impl-reviewer'))
    out = json.loads(capsys.readouterr().out)
    assert (out['family'], out['model'], out['run_via']) == ('claude', 'sonnet', 'harness')


def test_blank_entry_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        pipeline_lib.cmd_resolve(_ns(_write(tmp_path, BLANK_ENTRY), 'impl-reviewer'))
    assert exc.value.code == 2


def test_model_mismatch_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        pipeline_lib.cmd_resolve(_ns(_write(tmp_path, MODEL_MISMATCH), 'impl-reviewer'))
    assert exc.value.code == 2


def test_frontmatter_mismatch_fails_closed(tmp_path: Path) -> None:
    # Manifest matches policy (sonnet) but a fake agents dir declares a different model.
    fake = tmp_path / 'agents'
    fake.mkdir()
    (fake / 'impl-reviewer.md').write_text('---\nname: impl-reviewer\nmodel: opus\n---\nbody\n', encoding='utf-8')
    with pytest.raises(SystemExit) as exc:
        pipeline_lib.cmd_resolve(_ns(_write(tmp_path, VALID_IMPL), 'impl-reviewer', fake))
    assert exc.value.code == 2


def test_orientation_contradicts_drafter_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        pipeline_lib.cmd_resolve(_ns(_write(tmp_path, ORIENTATION_CONTRADICTION), 'contract-adversary'))
    assert exc.value.code == 2
    assert 'contradicts' in capsys.readouterr().err


def test_non_claude_implementer_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        pipeline_lib.cmd_resolve(_ns(_write(tmp_path, NON_CLAUDE_IMPLEMENTER), 'impl-reviewer'))
    assert exc.value.code == 2
    assert 'implementer_family' in capsys.readouterr().err
