"""Age guard for transcript-parent mining (2026-08-29 incident).

A hook fired by a resumed old session mined its whole date-coded session
folder (~/.codex/sessions/YYYY/MM/DD) in the background for hours at 88% CPU.
_get_mine_dir must refuse stale transcripts instead of returning their parent.
"""
import time
from datetime import datetime
from pathlib import Path

from mempalace.hooks_cli import _get_mine_dir


DAY = 86400


def _make_session_file(tmp_path: Path, day_offset_days: int, path_template: str) -> Path:
    """Create a fake transcript inside a date-coded sessions dir."""
    ts = time.time() - day_offset_days * DAY
    stamp = datetime.fromtimestamp(ts)
    target = tmp_path / path_template.format(
        y=stamp.year, m=stamp.month, d=stamp.day
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}")
    # Pin BOTH file times so the dir-date and the mtime agree on "old"
    os_time = (ts, ts)
    import os

    os.utime(target, os_time)
    return target


def test_stale_codex_session_dir_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPAL_DIR", raising=False)
    monkeypatch.delenv("MEMPAL_MINE_MAX_AGE_DAYS", raising=False)
    old = _make_session_file(
        tmp_path, day_offset_days=23, path_template="codex/sessions/{y:04d}/{m:02d}/{d:02d}/rollout-1.jsonl"
    )
    assert _get_mine_dir(str(old)) == ""


def test_fresh_transcript_still_mined(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPAL_DIR", raising=False)
    monkeypatch.delenv("MEMPAL_MINE_MAX_AGE_DAYS", raising=False)
    fresh = _make_session_file(
        tmp_path, day_offset_days=0, path_template="codex/sessions/{y:04d}/{m:02d}/{d:02d}/rollout-1.jsonl"
    )
    assert _get_mine_dir(str(fresh)) == str(fresh.parent)


def test_stale_by_mtime_without_date_in_path(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPAL_DIR", raising=False)
    monkeypatch.delenv("MEMPAL_MINE_MAX_AGE_DAYS", raising=False)
    old = _make_session_file(tmp_path, day_offset_days=10, path_template="claude/projects/xyz/session-1.jsonl")
    assert _get_mine_dir(str(old)) == ""


def test_env_zero_disables_guard(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMPAL_DIR", raising=False)
    monkeypatch.setenv("MEMPAL_MINE_MAX_AGE_DAYS", "0")
    old = _make_session_file(
        tmp_path, day_offset_days=90, path_template="codex/sessions/{y:04d}/{m:02d}/{d:02d}/rollout-1.jsonl"
    )
    assert _get_mine_dir(str(old)) == str(old.parent)


def test_resumed_old_session_with_fresh_mtime_is_caught_by_dir_date(tmp_path, monkeypatch):
    """The 08-29 case: an old rollout file whose mtime was touched today.

    The dir-date check must catch it even when the file looks fresh.
    """
    monkeypatch.delenv("MEMPAL_DIR", raising=False)
    monkeypatch.delenv("MEMPAL_MINE_MAX_AGE_DAYS", raising=False)
    stale = _make_session_file(
        tmp_path, day_offset_days=23, path_template="codex/sessions/{y:04d}/{m:02d}/{d:02d}/rollout-1.jsonl"
    )
    import os

    os.utime(stale, (time.time(), time.time()))  # simulate a resume touching the file
    assert _get_mine_dir(str(stale)) == ""
