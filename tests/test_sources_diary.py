"""First-party diary source adapter (RFC 002 reference implementation).

Covers the adapter contract (stream shape, skip semantics, is_current,
schema), first-party registration, the dry ``sources status`` diff, and
the CLI surface — plus one integration test proving the migrated
``ingest_diaries`` runner stamps adapter provenance on drawers.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from mempalace.diary_ingest import _state_file_for, ingest_diaries
from mempalace.sources import available_adapters, get_adapter
from mempalace.sources.base import (
    SourceItemMetadata,
    SourceNotFoundError,
    SourceRef,
)
from mempalace.sources.context import PalaceContext
from mempalace.sources.diary import DiarySourceAdapter, split_entries
from mempalace.sources.status import adapter_status


def _write_diary(root: Path, name: str, body: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / name
    p.write_text(body, encoding="utf-8")
    return p


def _make_context() -> PalaceContext:
    """A context with no collections — enough for extraction-only adapters."""
    return PalaceContext(
        drawer_collection=None,
        knowledge_graph=None,
        palace_path="/tmp/unused-palace",
    )


ENTRY_A = "## Entry one\nWorked on the write lock all morning and it holds.\n"
ENTRY_B = "## Entry two\nRefactored diary ingest onto the adapter cleanly.\n"


# ---------------------------------------------------------------------------
# Adapter stream shape
# ---------------------------------------------------------------------------


def test_yields_metadata_then_record_per_dated_file(tmp_path):
    _write_diary(tmp_path, "2026-08-14.md", ENTRY_A + ENTRY_B)
    _write_diary(tmp_path, "2026-08-15.md", ENTRY_A)

    adapter = DiarySourceAdapter()
    items = list(adapter.ingest(source=SourceRef(local_path=str(tmp_path)), palace=_make_context()))

    # Two files, metadata+record each, sorted by filename.
    assert [type(i).__name__ for i in items] == [
        "SourceItemMetadata",
        "DrawerRecord",
        "SourceItemMetadata",
        "DrawerRecord",
    ]
    meta_a, rec_a, meta_b, rec_b = items
    assert meta_a.source_file.endswith("2026-08-14.md")
    assert meta_a.version.count(":") == 1  # "size:mtime_ns"
    assert meta_a.size_hint == Path(meta_a.source_file).stat().st_size
    assert meta_a.route_hint.wing == "diary"
    assert meta_a.route_hint.room == "daily"

    assert rec_a.metadata["date"] == "2026-08-14"
    assert rec_a.metadata["wing"] == "diary"
    assert rec_a.metadata["room"] == "daily"
    assert rec_a.metadata["source_session"] == "daily_diary"
    assert rec_a.metadata["entry_count"] == 2
    assert rec_b.metadata["date"] == "2026-08-15"
    assert rec_b.chunk_index == 0


def test_record_content_is_verbatim_and_size_is_bytes(tmp_path):
    # Non-ASCII on purpose: char count != byte count.
    body = "## Entrada\nHoy trabajé en el candado de escritura ñandú 🦤.\n"
    p = _write_diary(tmp_path, "2026-08-14.md", body)

    adapter = DiarySourceAdapter()
    _, rec = list(
        adapter.ingest(source=SourceRef(local_path=str(tmp_path)), palace=_make_context())
    )

    assert rec.content == body  # verbatim, no transformation
    assert rec.metadata["size"] == p.stat().st_size  # bytes, not chars
    assert rec.metadata["size"] != len(body)


def test_skips_undated_and_too_short_files(tmp_path):
    _write_diary(tmp_path, "notes.md", ENTRY_A * 10)  # undated stem
    _write_diary(tmp_path, "2026-08-14.md", "## tiny\nshort\n")  # < 50 chars

    adapter = DiarySourceAdapter()
    items = list(adapter.ingest(source=SourceRef(local_path=str(tmp_path)), palace=_make_context()))

    # The short-but-dated file yields metadata (stat is free) but no record.
    assert [type(i).__name__ for i in items] == ["SourceItemMetadata"]


def test_skip_flag_suppresses_read_and_record(tmp_path):
    p = _write_diary(tmp_path, "2026-08-14.md", ENTRY_A)
    original_read = Path.read_text

    def _no_reads(self, *a, **kw):
        raise AssertionError(f"adapter read file body during skip: {self}")

    adapter = DiarySourceAdapter()
    ctx = _make_context()
    stream = adapter.ingest(source=SourceRef(local_path=str(tmp_path)), palace=ctx)

    meta = next(stream)
    assert isinstance(meta, SourceItemMetadata)
    ctx.skip_current_item()
    try:
        Path.read_text = _no_reads  # type: ignore[method-assign]
        rest = list(stream)
    finally:
        Path.read_text = original_read  # type: ignore[method-assign]

    assert rest == []  # skipped: no record, and no body read
    assert p.exists()


def test_wing_option_flows_into_route_and_metadata(tmp_path):
    _write_diary(tmp_path, "2026-08-14.md", ENTRY_A)
    adapter = DiarySourceAdapter()
    meta, rec = list(
        adapter.ingest(
            source=SourceRef(local_path=str(tmp_path), options={"wing": "work"}),
            palace=_make_context(),
        )
    )
    assert meta.route_hint.wing == "work"
    assert rec.metadata["wing"] == "work"


def test_missing_dir_raises_source_not_found(tmp_path):
    adapter = DiarySourceAdapter()
    with pytest.raises(SourceNotFoundError):
        list(
            adapter.ingest(
                source=SourceRef(local_path=str(tmp_path / "nope")), palace=_make_context()
            )
        )


# ---------------------------------------------------------------------------
# is_current + schema
# ---------------------------------------------------------------------------


def _meta(version: str) -> SourceItemMetadata:
    return SourceItemMetadata(source_file="/x/2026-08-14.md", version=version, size_hint=1)


def test_is_current_matches_on_size_only():
    adapter = DiarySourceAdapter()
    item = _meta("123:999")
    assert adapter.is_current(item=item, existing_metadata={"size": 123}) is True
    # mtime drift alone (same size) must NOT trigger a re-ingest
    assert adapter.is_current(item=item, existing_metadata={"size": 123}) is True
    assert adapter.is_current(item=item, existing_metadata={"size": 124}) is False


def test_is_current_rejects_missing_or_malformed_state():
    adapter = DiarySourceAdapter()
    item = _meta("123:999")
    assert adapter.is_current(item=item, existing_metadata=None) is False
    assert adapter.is_current(item=item, existing_metadata={}) is False
    assert adapter.is_current(item=item, existing_metadata={"size": "123"}) is False
    assert adapter.is_current(item=_meta("not-a-number"), existing_metadata={"size": 123}) is False


def test_describe_schema_declares_every_field():
    schema = DiarySourceAdapter().describe_schema()
    assert schema.version == DiarySourceAdapter.adapter_version
    for field in ("date", "wing", "room", "source_session", "size", "entry_count"):
        spec = schema.fields[field]
        assert spec.required, field
    assert schema.fields["size"].type == "int"
    entities = schema.fields["entities"]
    assert entities.required is False
    assert entities.delimiter == ";"
    # The adapter transforms nothing — verbatim is the whole promise.
    assert DiarySourceAdapter.declared_transformations == frozenset()
    assert DiarySourceAdapter.default_privacy_class == "pii_potential"


def test_source_summary_counts_dated_files_only(tmp_path):
    _write_diary(tmp_path, "2026-08-14.md", ENTRY_A)
    _write_diary(tmp_path, "random.md", ENTRY_A)
    summary = DiarySourceAdapter().source_summary(source=SourceRef(local_path=str(tmp_path)))
    assert summary.item_count == 1


def test_split_entries_pairs_headers_with_bodies():
    entries = split_entries(f"{ENTRY_A}{ENTRY_B}")
    assert len(entries) == 2
    assert entries[0][0] == "## Entry one"
    assert "write lock" in entries[0][1]
    assert entries[1][0] == "## Entry two"


# ---------------------------------------------------------------------------
# First-party registration
# ---------------------------------------------------------------------------


def test_diary_registered_first_party():
    assert "diary" in available_adapters()
    assert isinstance(get_adapter("diary"), DiarySourceAdapter)


# ---------------------------------------------------------------------------
# sources status (dry diff)
# ---------------------------------------------------------------------------


def _forge_state(palace_path: str, diary_dir: Path, entries: dict) -> None:
    state_file = _state_file_for(palace_path, diary_dir.resolve())
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(entries))


def test_adapter_status_classifies_new_stale_current(tmp_path):
    palace = str(tmp_path / "palace")
    a = _write_diary(tmp_path / "diary", "2026-08-14.md", ENTRY_A)
    _write_diary(tmp_path / "diary", "2026-08-15.md", ENTRY_A)
    source = SourceRef(local_path=str(tmp_path / "diary"))

    # No state at all → everything new.
    result = adapter_status(adapter_name="diary", source=source, palace_path=palace)
    assert result == {"adapter": "diary", "items": 2, "current": 0, "stale": 0, "new": 2}

    # One ingested (size matches), one stale (size drifted), one absent.
    _forge_state(
        palace,
        tmp_path / "diary",
        {
            "diary|2026-08-14.md": {"size": a.stat().st_size, "entry_count": 1},
            "diary|2026-08-15.md": {"size": a.stat().st_size + 5, "entry_count": 1},
        },
    )
    result = adapter_status(adapter_name="diary", source=source, palace_path=palace)
    assert result["current"] == 1
    assert result["stale"] == 1
    assert result["new"] == 0


def test_adapter_status_never_reads_file_bodies(tmp_path):
    """The strongest oracle: make bodies unreadable — status must still work."""
    a = _write_diary(tmp_path / "diary", "2026-08-14.md", ENTRY_A)
    _forge_state(
        str(tmp_path / "palace"),
        tmp_path / "diary",
        {"diary|2026-08-14.md": {"size": a.stat().st_size, "entry_count": 1}},
    )
    os.chmod(a, 0o000)
    try:
        result = adapter_status(
            adapter_name="diary",
            source=SourceRef(local_path=str(tmp_path / "diary")),
            palace_path=str(tmp_path / "palace"),
        )
        assert result["current"] == 1
    finally:
        os.chmod(a, 0o644)


def test_adapter_status_unknown_adapter_raises_keyerror(tmp_path):
    with pytest.raises(KeyError):
        adapter_status(
            adapter_name="nope",
            source=SourceRef(local_path=str(tmp_path)),
            palace_path=str(tmp_path / "palace"),
        )


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_sources_list_shows_diary(monkeypatch, capsys):
    from mempalace.cli import main

    monkeypatch.setattr(sys, "argv", ["mempalace", "sources", "list"])
    main()
    out = capsys.readouterr().out
    assert "ADAPTER" in out
    assert "diary" in out
    assert "whole_record" in out


def test_cli_sources_status_renders_counts(tmp_path, monkeypatch, capsys):
    from mempalace.cli import main

    a = _write_diary(tmp_path / "diary", "2026-08-14.md", ENTRY_A)
    _forge_state(
        str(tmp_path / "palace"),
        tmp_path / "diary",
        {"diary|2026-08-14.md": {"size": a.stat().st_size - 1, "entry_count": 1}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mempalace",
            "--palace",
            str(tmp_path / "palace"),
            "sources",
            "status",
            "diary",
            "--dir",
            str(tmp_path / "diary"),
        ],
    )
    main()
    out = capsys.readouterr().out
    assert "stale:   1" in out
    assert "Run ingest" in out


def test_cli_sources_no_action_prints_help(monkeypatch, capsys):
    from mempalace.cli import main

    monkeypatch.setattr(sys, "argv", ["mempalace", "sources"])
    main()
    out = capsys.readouterr().out
    assert "list" in out and "status" in out


# ---------------------------------------------------------------------------
# Integration: migrated runner stamps provenance
# ---------------------------------------------------------------------------


def test_ingest_diaries_stamps_adapter_and_byte_state(tmp_path):
    palace = str(tmp_path / "palace")
    diary_dir = tmp_path / "diary"
    p = _write_diary(diary_dir, "2026-08-14.md", ENTRY_A + ENTRY_B)

    result = ingest_diaries(diary_dir, palace)
    assert result["days_updated"] == 1

    from mempalace.palace import get_collection

    col = get_collection(palace)
    got = col.get(where={"wing": "diary"}, include=["metadatas", "documents"])
    assert len(got.ids) == 1
    meta = got.metadatas[0]
    # RFC 002 §5.1 — core stamps adapter provenance on every drawer.
    assert meta["adapter_name"] == "diary"
    assert meta["adapter_version"] == DiarySourceAdapter.adapter_version
    assert meta["date"] == "2026-08-14"
    assert got.documents[0] == (ENTRY_A + ENTRY_B)  # verbatim

    # State records BYTES — the same quantity is_current compares against.
    state = json.loads(_state_file_for(palace, diary_dir.resolve()).read_text())
    assert state["diary|2026-08-14.md"]["size"] == p.stat().st_size
    assert state["diary|2026-08-14.md"]["entry_count"] == 2

    # Re-ingest with no changes is a no-op.
    again = ingest_diaries(diary_dir, palace)
    assert again["days_updated"] == 0
    assert col.count() == 1
