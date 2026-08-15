"""Chaos test: reproduce the 2026-08-15 concurrent-writer corruption.

Six processes (one per simulated Claude-session MCP server) hammer adds
against ONE palace through the backend chokepoint. Before the write lock,
this raced PersistentClients and corrupted the HNSW segment (SIGSEGV in
chromadb_rust_bindings). With the lock, every drawer must survive and the
index must answer queries from a fresh process afterwards.
"""

import subprocess
import sys
import textwrap

from mempalace.backends.chroma import ChromaBackend

N_WORKERS = 6
WRITES_PER_WORKER = 8

_WORKER = textwrap.dedent(
    """
    import sys
    from mempalace.backends.chroma import ChromaBackend
    palace, worker_id = sys.argv[1], int(sys.argv[2])
    b = ChromaBackend()
    col = b.get_collection(palace, "chaos", create=True)
    for i in range(%d):
        n = worker_id * 1000 + i
        col.add(
            documents=[f"drawer-{n} verbatim content"],
            ids=[f"id-{n}"],
            metadatas=[{"worker": worker_id, "seq": i}],
            embeddings=[[float(worker_id), float(i), 1.0]],
        )
    b.close()
    """
    % WRITES_PER_WORKER
)

_VERIFY = textwrap.dedent(
    """
    import sys
    from mempalace.backends.chroma import ChromaBackend
    col = ChromaBackend().get_collection(sys.argv[1], "chaos")
    print(col.count())
    r = col.query(query_embeddings=[[1.0, 0.0, 1.0]], n_results=5)
    print(len(r.ids[0]))
    """
)


def test_chaos_six_concurrent_writers_zero_loss(tmp_path):
    palace = tmp_path / "palace"
    palace.mkdir()

    procs = [
        subprocess.Popen([sys.executable, "-c", _WORKER, str(palace), str(w)])
        for w in range(N_WORKERS)
    ]
    returncodes = [p.wait(timeout=300) for p in procs]
    assert returncodes == [0] * N_WORKERS, f"worker crashed: {returncodes}"

    # Integrity check from a fresh reader.
    backend = ChromaBackend()
    col = backend.get_collection(str(palace), "chaos")
    expected = N_WORKERS * WRITES_PER_WORKER
    assert col.count() == expected, f"data loss: expected {expected} drawers, got {col.count()}"
    got = col.get(
        ids=[f"id-{w * 1000 + i}" for w in range(N_WORKERS) for i in range(WRITES_PER_WORKER)]
    )
    assert len(got.ids) == expected
    assert len(set(got.ids)) == expected, "duplicate ids written"
    backend.close()

    # And from a fresh PROCESS: index must be queryable, not segfaulted.
    out = subprocess.run(
        [sys.executable, "-c", _VERIFY, str(palace)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, f"fresh-process query crashed:\n{out.stderr[-800:]}"
    count, hits = out.stdout.split()
    assert int(count) == expected
    assert int(hits) == 5
