"""Tests for the cross-process palace write lock (RFC: concurrent-writer fix).

Reproduces the 2026-08-15 incident class: multiple processes writing one
palace. The lock must serialize them, fail loudly on timeout, and never
leave a stale lock behind when a holder dies.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

from mempalace.backends.base import BackendError
from mempalace.backends.chroma import ChromaBackend, ChromaCollection
from mempalace.backends.writelock import (
    WriteLockTimeoutError,
    palace_write_lock,
)

_HOLD_SCRIPT = textwrap.dedent(
    """
    import sys, time
    from mempalace.backends.writelock import palace_write_lock
    with palace_write_lock(sys.argv[1]):
        print("held", flush=True)
        time.sleep(float(sys.argv[2]))
    """
)

_DIE_SCRIPT = textwrap.dedent(
    """
    import sys
    from mempalace.backends.writelock import palace_write_lock
    with palace_write_lock(sys.argv[1]):
        print("held", flush=True)
        sys.stdout.flush()
        # Simulate a crashed writer: hard exit while holding the lock.
        os_exit = getattr(__import__("os"), "_exit")
        os_exit(1)
    """
)


class _RecordingCollection:
    """Fake chroma collection that records delegated mutations."""

    def __init__(self):
        self.calls = []

    def add(self, **kwargs):
        self.calls.append(("add", kwargs))

    def upsert(self, **kwargs):
        self.calls.append(("upsert", kwargs))

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))


def _spawn(script: str, palace: str, arg: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", script, str(palace), arg],
        stdout=subprocess.PIPE,
        text=True,
    )


def test_lock_file_is_created_inside_palace(tmp_path):
    palace = tmp_path / "palace"
    palace.mkdir()
    with palace_write_lock(str(palace)):
        assert (palace / ".writes.lock").is_file()


def test_timeout_raises_loudly_never_yields_unlocked(tmp_path):
    palace = tmp_path / "palace"
    palace.mkdir()
    holder = _spawn(_HOLD_SCRIPT, str(palace), "3")
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(WriteLockTimeoutError) as exc_info:
            with palace_write_lock(str(palace), timeout=0.2):
                pytest.fail("context body must never run when the lock is held")
        assert f"pid={holder.pid}" in str(exc_info.value)
    finally:
        holder.wait(timeout=10)


def test_timeout_error_is_a_backend_error(tmp_path):
    # Loud failure contract: callers catching BackendError see it.
    assert issubclass(WriteLockTimeoutError, BackendError)


def test_lock_released_when_holder_dies(tmp_path):
    """flock is kernel-released: a crashed writer must not wedge the palace."""
    palace = tmp_path / "palace"
    palace.mkdir()
    crasher = _spawn(_DIE_SCRIPT, str(palace), "0")
    crasher.wait(timeout=10)
    assert crasher.returncode == 1
    # Must acquire immediately — no stale lock, no manual cleanup.
    start = time.monotonic()
    with palace_write_lock(str(palace), timeout=5):
        pass
    assert time.monotonic() - start < 4


def test_chroma_collection_write_takes_the_lock(tmp_path):
    palace = tmp_path / "palace"
    palace.mkdir()
    fake = _RecordingCollection()
    col = ChromaCollection(fake, palace_path=str(palace))
    col.add(documents=["d"], ids=["i"])
    col.upsert(documents=["d"], ids=["i"])
    col.update(ids=["i"], documents=["d2"])
    col.delete(ids=["i"])
    assert len(fake.calls) == 4
    assert (palace / ".writes.lock").is_file()


def test_chroma_collection_without_palace_path_still_writes(tmp_path):
    """Backward compat: path-less collections (legacy/tests) keep working."""
    fake = _RecordingCollection()
    col = ChromaCollection(fake)
    col.add(documents=["d"], ids=["i"])
    assert fake.calls[0][0] == "add"


def test_env_var_overrides_default_timeout(tmp_path, monkeypatch):
    from mempalace.backends import writelock

    monkeypatch.setenv("MEMPALACE_WRITE_LOCK_TIMEOUT", "0.15")
    palace = tmp_path / "palace"
    palace.mkdir()
    holder = _spawn(_HOLD_SCRIPT, str(palace), "3")
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(WriteLockTimeoutError):
            with palace_write_lock(str(palace)):
                pass
        assert writelock._resolve_timeout(None) == 0.15
    finally:
        holder.wait(timeout=10)


def test_client_init_quarantines_stale_hnsw_segment(tmp_path):
    """The corruption sentinel: opening a client on a drifted palace must
    rename the stale HNSW segment out of the way before chroma touches it."""
    import chromadb

    palace = tmp_path / "palace"
    palace.mkdir()
    backend = ChromaBackend()
    col = backend.get_collection(str(palace), "sentinel_test", create=True)
    col.add(documents=["seed"], ids=["seed-id"], embeddings=[[1.0, 2.0, 3.0]])
    backend.close()

    # Forge the crashed-mid-write state: HNSW segment 2h older than sqlite.
    for entry in palace.iterdir():
        if entry.is_dir() and (entry / "data_level0.bin").is_file():
            old = time.time() - 7200
            os.utime(entry / "data_level0.bin", (old, old))
    old = time.time() - 3600
    os.utime(palace / "chroma.sqlite3", (old, old))

    # A fresh process-equivalent: new backend instance, cold client cache.
    # Drop chromadb's in-process system singleton so the client is rebuilt
    # from disk (what a restarted MCP server would see).
    try:
        chromadb.instance.SharedSystemClient._identifier_to_system = {}
    except AttributeError:
        pass

    backend2 = ChromaBackend()
    reopened = backend2.get_collection(str(palace), "sentinel_test")
    quarantined = [p.name for p in palace.iterdir() if ".drift-" in p.name]
    assert quarantined, "sentinel should have quarantined the stale segment"
    assert reopened.count() == 1  # sqlite is the source of truth; count intact
    # chroma rebuilds the index lazily — a query must not segfault.
    result = reopened.query(query_embeddings=[[1.0, 2.0, 3.0]], n_results=1)
    assert result.ids[0] == ["seed-id"]
    backend2.close()
    try:
        chromadb.instance.SharedSystemClient._identifier_to_system = {}
    except AttributeError:
        pass


def test_cold_reader_waits_for_active_writer_lock(tmp_path, monkeypatch):
    """Client startup/repair must not race a writer holding the palace lock."""
    palace = tmp_path / "palace"
    palace.mkdir()
    holder = _spawn(_HOLD_SCRIPT, str(palace), "0.5")

    class _FakeClient:
        def get_collection(self, _name):
            return _RecordingCollection()

    monkeypatch.setattr(
        "mempalace.backends.chroma.chromadb.PersistentClient",
        lambda *, path: _FakeClient(),
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        start = time.monotonic()
        ChromaBackend().get_collection(str(palace), "existing")
        assert time.monotonic() - start >= 0.35
    finally:
        holder.wait(timeout=10)


def test_writes_from_two_processes_serialize(tmp_path):
    """Two processes mutating one palace through the backend chokepoint."""
    palace = tmp_path / "palace"
    palace.mkdir()
    script = textwrap.dedent(
        """
        import sys
        from mempalace.backends.chroma import ChromaBackend
        base = sys.argv[2]
        b = ChromaBackend()
        col = b.get_collection(sys.argv[1], "serial", create=True)
        col.add(documents=[f"doc-{base}"], ids=[f"id-{base}"],
                embeddings=[[float(base), 1.0, 2.0]])
        b.close()
        """
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", script, str(palace), str(i)]) for i in range(2)
    ]
    for p in procs:
        assert p.wait(timeout=60) == 0

    backend = ChromaBackend()
    col = backend.get_collection(str(palace), "serial")
    assert col.count() == 2
    got = col.get(ids=["id-0", "id-1"])
    assert sorted(got.ids) == ["id-0", "id-1"]
