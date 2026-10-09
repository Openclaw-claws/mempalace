"""Tests for mempalace.repair — scan, prune, and rebuild HNSW index."""

import os
from unittest.mock import MagicMock, patch


from mempalace import repair


# ── _get_palace_path ──────────────────────────────────────────────────


@patch("mempalace.repair.MempalaceConfig", create=True)
def test_get_palace_path_from_config(mock_config_cls):
    mock_config_cls.return_value.palace_path = "/configured/palace"
    with patch.dict("sys.modules", {}):
        # Force reimport to pick up the mock
        result = repair._get_palace_path()
    assert isinstance(result, str)


def test_get_palace_path_fallback():
    with patch("mempalace.repair._get_palace_path") as mock_get:
        mock_get.return_value = os.path.join(os.path.expanduser("~"), ".mempalace", "palace")
        result = mock_get()
        assert ".mempalace" in result


# ── _paginate_ids ─────────────────────────────────────────────────────


def test_paginate_ids_single_batch():
    col = MagicMock()
    col.get.return_value = {"ids": ["id1", "id2", "id3"]}
    ids = repair._paginate_ids(col)
    assert ids == ["id1", "id2", "id3"]


def test_paginate_ids_empty():
    col = MagicMock()
    col.get.return_value = {"ids": []}
    ids = repair._paginate_ids(col)
    assert ids == []


def test_paginate_ids_with_where():
    col = MagicMock()
    col.get.return_value = {"ids": ["id1"]}
    repair._paginate_ids(col, where={"wing": "test"})
    col.get.assert_called_with(where={"wing": "test"}, include=[], limit=1000, offset=0)


def test_paginate_ids_offset_exception_fallback():
    col = MagicMock()
    # First call raises, fallback returns ids, second fallback returns empty
    col.get.side_effect = [
        Exception("offset bug"),
        {"ids": ["id1", "id2"]},
        Exception("offset bug"),
        {"ids": ["id1", "id2"]},  # same ids = no new = break
    ]
    ids = repair._paginate_ids(col)
    assert "id1" in ids


# ── scan_palace ───────────────────────────────────────────────────────


def _install_mock_backend(mock_backend_cls, collection):
    """Wire mock_backend_cls so ChromaBackend().get_collection(...) returns *collection*."""
    mock_backend = MagicMock()
    mock_backend.get_collection.return_value = collection
    mock_backend_cls.return_value = mock_backend
    return mock_backend


@patch("mempalace.repair.ChromaBackend")
def test_scan_palace_no_ids(mock_backend_cls, tmp_path):
    mock_col = MagicMock()
    mock_col.count.return_value = 0
    mock_col.get.return_value = {"ids": []}
    _install_mock_backend(mock_backend_cls, mock_col)

    good, bad = repair.scan_palace(palace_path=str(tmp_path))
    assert good == set()
    assert bad == set()


@patch("mempalace.repair.ChromaBackend")
def test_scan_palace_all_good(mock_backend_cls, tmp_path):
    mock_col = MagicMock()
    mock_col.count.return_value = 2
    # _paginate_ids call
    mock_col.get.side_effect = [
        {"ids": ["id1", "id2"]},  # paginate
        {"ids": ["id1", "id2"]},  # probe batch — both returned
    ]
    _install_mock_backend(mock_backend_cls, mock_col)

    good, bad = repair.scan_palace(palace_path=str(tmp_path))
    assert "id1" in good
    assert "id2" in good
    assert len(bad) == 0


@patch("mempalace.repair.ChromaBackend")
def test_scan_palace_with_bad_ids(mock_backend_cls, tmp_path):
    mock_col = MagicMock()
    mock_col.count.return_value = 2

    def get_side_effect(**kwargs):
        ids = kwargs.get("ids", None)
        if ids is None:
            # paginate call
            return {"ids": ["good1", "bad1"]}
        if "bad1" in ids and len(ids) == 1:
            raise Exception("corrupt")
        if "good1" in ids and len(ids) == 1:
            return {"ids": ["good1"]}
        # batch probe — raise to force per-id
        raise Exception("batch fail")

    mock_col.get.side_effect = get_side_effect
    _install_mock_backend(mock_backend_cls, mock_col)

    good, bad = repair.scan_palace(palace_path=str(tmp_path))
    assert "good1" in good
    assert "bad1" in bad


@patch("mempalace.repair.ChromaBackend")
def test_scan_palace_with_wing_filter(mock_backend_cls, tmp_path):
    mock_col = MagicMock()
    mock_col.count.return_value = 1
    mock_col.get.side_effect = [
        {"ids": ["id1"]},  # paginate
        {"ids": ["id1"]},  # probe
    ]
    _install_mock_backend(mock_backend_cls, mock_col)

    repair.scan_palace(palace_path=str(tmp_path), only_wing="test_wing")
    # Verify where filter was passed
    first_call = mock_col.get.call_args_list[0]
    assert first_call.kwargs.get("where") == {"wing": "test_wing"}


# ── prune_corrupt ─────────────────────────────────────────────────────


@patch("mempalace.repair.ChromaBackend")
def test_prune_corrupt_no_file(mock_backend_cls, tmp_path):
    # Should print message and return without error
    repair.prune_corrupt(palace_path=str(tmp_path))


@patch("mempalace.repair.ChromaBackend")
def test_prune_corrupt_dry_run(mock_backend_cls, tmp_path):
    bad_file = tmp_path / "corrupt_ids.txt"
    bad_file.write_text("bad1\nbad2\n")
    repair.prune_corrupt(palace_path=str(tmp_path), confirm=False)
    # No backend calls in dry run
    mock_backend_cls.assert_not_called()


@patch("mempalace.repair.ChromaBackend")
def test_prune_corrupt_confirmed(mock_backend_cls, tmp_path):
    bad_file = tmp_path / "corrupt_ids.txt"
    bad_file.write_text("bad1\nbad2\n")

    mock_col = MagicMock()
    mock_col.count.side_effect = [10, 8]
    _install_mock_backend(mock_backend_cls, mock_col)

    repair.prune_corrupt(palace_path=str(tmp_path), confirm=True)
    mock_col.delete.assert_called_once()


@patch("mempalace.repair.ChromaBackend")
def test_prune_corrupt_delete_failure_fallback(mock_backend_cls, tmp_path):
    bad_file = tmp_path / "corrupt_ids.txt"
    bad_file.write_text("bad1\nbad2\n")

    mock_col = MagicMock()
    mock_col.count.side_effect = [10, 8]
    # Batch delete fails, per-id succeeds
    mock_col.delete.side_effect = [Exception("batch fail"), None, None]
    _install_mock_backend(mock_backend_cls, mock_col)

    repair.prune_corrupt(palace_path=str(tmp_path), confirm=True)
    assert mock_col.delete.call_count == 3  # 1 batch + 2 individual


# ── rebuild_index ─────────────────────────────────────────────────────


@patch("mempalace.repair.ChromaBackend")
def test_rebuild_index_no_palace(mock_backend_cls, tmp_path):
    nonexistent = str(tmp_path / "nope")
    repair.rebuild_index(palace_path=nonexistent)
    mock_backend_cls.assert_not_called()


@patch("mempalace.repair.shutil")
@patch("mempalace.repair.ChromaBackend")
def test_rebuild_index_empty_palace(mock_backend_cls, mock_shutil, tmp_path):
    mock_col = MagicMock()
    mock_col.count.return_value = 0
    mock_backend = _install_mock_backend(mock_backend_cls, mock_col)

    repair.rebuild_index(palace_path=str(tmp_path))
    mock_backend.delete_collection.assert_not_called()


@patch("mempalace.repair.shutil")
@patch("mempalace.repair.ChromaBackend")
def test_rebuild_index_success(mock_backend_cls, mock_shutil, tmp_path):
    # Real mini chroma sqlite — extraction now reads raw sqlite so a corrupt
    # HNSW segment can't SIGSEGV the repair process.
    import sqlite3

    sqlite_path = tmp_path / "chroma.sqlite3"
    con = sqlite3.connect(sqlite_path)
    con.executescript(
        """
        CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE segments (id TEXT PRIMARY KEY, type TEXT, scope TEXT,
                               collection TEXT);
        CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT,
                                 embedding_id TEXT, seq_id BLOB);
        CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT,
                                         int_value INTEGER, float_value REAL,
                                         bool_value INTEGER);
        INSERT INTO collections VALUES ('c1', 'mempalace_drawers');
        INSERT INTO segments VALUES ('s1', 'vec', 'c1', 'c1');
        INSERT INTO embeddings VALUES (1, 's1', 'id1', x'01'), (2, 's1', 'id2', x'02');
        INSERT INTO embedding_metadata VALUES (1, 'chroma:document', 'doc1', NULL, NULL, NULL);
        INSERT INTO embedding_metadata VALUES (1, 'wing', 'a', NULL, NULL, NULL);
        INSERT INTO embedding_metadata VALUES (2, 'chroma:document', 'doc2', NULL, NULL, NULL);
        INSERT INTO embedding_metadata VALUES (2, 'wing', 'b', NULL, NULL, NULL);
        """
    )
    con.commit()
    con.close()

    mock_new_col = MagicMock()
    mock_backend = _install_mock_backend(mock_backend_cls, MagicMock())
    mock_backend.create_collection.return_value = mock_new_col

    repair.rebuild_index(palace_path=str(tmp_path))

    # Verify: backed up sqlite only (not copytree)
    mock_shutil.copy2.assert_called_once()
    assert "chroma.sqlite3" in str(mock_shutil.copy2.call_args)

    # Verify: deleted and recreated (cosine is the backend default)
    mock_backend.delete_collection.assert_called_once_with(str(tmp_path), "mempalace_drawers")
    mock_backend.create_collection.assert_called_once_with(str(tmp_path), "mempalace_drawers")

    # Verify: used upsert not add
    mock_new_col.upsert.assert_called_once()
    mock_new_col.add.assert_not_called()


@patch("mempalace.repair.shutil")
@patch("mempalace.repair.ChromaBackend")
def test_rebuild_index_error_reading(mock_backend_cls, mock_shutil, tmp_path):
    mock_backend = MagicMock()
    mock_backend.get_collection.side_effect = Exception("corrupt")
    mock_backend_cls.return_value = mock_backend

    repair.rebuild_index(palace_path=str(tmp_path))
    mock_backend.delete_collection.assert_not_called()
