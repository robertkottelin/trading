"""Tests for config_editor — validate, diff, backup, save."""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.lib import config_editor, state


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch):
    d = tmp_path / "config"
    d.mkdir()
    (d / "settings.yaml").write_text("execution:\n  mode: paper\n  network: testnet\n")
    (d / "strategy_params.yaml").write_text("foo: 1\n")
    monkeypatch.setattr(state, "CONFIG_DIR", d)
    monkeypatch.setattr(config_editor, "BACKUP_DIR", d / ".backups")
    return d


def test_list_configs(cfg_dir):
    files = config_editor.list_yaml_configs()
    names = [p.name for p in files]
    assert "settings.yaml" in names
    assert "strategy_params.yaml" in names


def test_validate_ok():
    ok, err, parsed = config_editor.validate_yaml("a: 1\nb: 2\n")
    assert ok
    assert err is None
    assert parsed == {"a": 1, "b": 2}


def test_validate_bad():
    ok, err, _ = config_editor.validate_yaml("a: [unclosed\n")
    assert not ok
    assert "expected" in err.lower() or "could not find" in err.lower() or "while parsing" in err.lower()


def test_diff_text():
    diff = config_editor.diff_text("a: 1\nb: 2\n", "a: 1\nb: 3\n", path="x.yaml")
    assert "-b: 2" in diff
    assert "+b: 3" in diff


def test_diff_identical():
    assert config_editor.diff_text("a: 1\n", "a: 1\n") == ""


def test_save_creates_backup_and_writes(cfg_dir):
    target = cfg_dir / "settings.yaml"
    original = target.read_text()
    new = "execution:\n  mode: live\n  network: testnet\n"
    bp = config_editor.save(target, new)
    assert target.read_text() == new
    assert bp is not None
    assert bp.exists()
    assert bp.read_text() == original


def test_save_rejects_invalid_yaml(cfg_dir):
    target = cfg_dir / "settings.yaml"
    pre = target.read_text()
    with pytest.raises(ValueError):
        config_editor.save(target, "a: [unclosed\n")
    # File on disk must not have changed.
    assert target.read_text() == pre


def test_save_creates_no_backup_when_target_new(cfg_dir):
    target = cfg_dir / "new_file.yaml"
    bp = config_editor.save(target, "a: 1\n")
    assert target.read_text() == "a: 1\n"
    assert bp is None
