"""
repair.py — Scan, prune corrupt entries, and rebuild HNSW index
================================================================

When ChromaDB's HNSW index accumulates duplicate entries (from repeated
add() calls with the same ID), link_lists.bin can grow unbounded —
terabytes on large palaces — eventually causing segfaults.

This module provides three operations:

  scan    — find every corrupt/unfetchable ID in the palace
  prune   — delete only the corrupt IDs (surgical)
  rebuild — extract all drawers, delete the collection, recreate with
            correct HNSW settings, and upsert everything back

The rebuild backs up ONLY chroma.sqlite3 (the source of truth), not the
full palace directory — so it works even when link_lists.bin is bloated.

Usage (standalone):
    python -m mempalace.repair scan [--wing X]
    python -m mempalace.repair prune --confirm
    python -m mempalace.repair rebuild

Usage (from CLI):
    mempalace repair
    mempalace repair-scan [--wing X]
    mempalace repair-prune --confirm
"""

import argparse
import os
import shutil
import sqlite3
import time

from .backends.chroma import ChromaBackend


def _extract_drawers_from_sqlite(palace_path, collection_name=None):
    """Extract (ids, docs, metadatas) reading chroma.sqlite3 directly.

    Immune to the hard SIGSEGV a corrupt HNSW segment causes inside the
    Rust bindings — col.get()/col.count() kill the whole process (exit 139,
    uncatchable by try/except), so repair must never depend on them while
    the corrupt segment exists. Documents live in embedding_metadata under
    the ``chroma:document`` key.
    """
    collection_name = collection_name or COLLECTION_NAME
    if not os.path.exists(os.path.join(palace_path, "chroma.sqlite3")):
        return [], [], []
    con = sqlite3.connect(os.path.join(palace_path, "chroma.sqlite3"))
    con.row_factory = sqlite3.Row
    try:
        have = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='collections'"
        ).fetchone()
        if not have:
            return [], [], []
        cid = con.execute(
            "SELECT id FROM collections WHERE name=?", (collection_name,)
        ).fetchone()
        if not cid:
            return [], [], []
        segs = [
            r["id"]
            for r in con.execute("SELECT id FROM segments WHERE collection=?", (cid["id"],))
        ]
        if not segs:
            return [], [], []
        marks = ",".join("?" * len(segs))
        rows = con.execute(
            """SELECT e.id AS eid, e.embedding_id, em.key, em.string_value,
                      em.int_value, em.float_value, em.bool_value
               FROM embeddings e LEFT JOIN embedding_metadata em ON em.id = e.id
               WHERE e.segment_id IN (%s) ORDER BY e.id""" % marks,
            segs,
        )
        ids, docs, metas = [], [], []
        cur = None
        for r in rows:
            if r["eid"] != cur:
                cur = r["eid"]
                ids.append(r["embedding_id"])
                docs.append("")
                metas.append({})
            if r["key"] is None:
                continue
            if r["key"] == "chroma:document":
                docs[-1] = r["string_value"] or ""
            else:
                v = r["string_value"]
                if v is None and r["int_value"] is not None:
                    v = r["int_value"]
                elif v is None and r["float_value"] is not None:
                    v = r["float_value"]
                elif v is None and r["bool_value"] is not None:
                    v = bool(r["bool_value"])
                metas[-1][r["key"]] = v
        return ids, docs, metas
    except sqlite3.DatabaseError:
        # palace/fixture inválido o no-sqlite — tratado como vacío
        return [], [], []
    finally:
        con.close()


COLLECTION_NAME = "mempalace_drawers"


def _get_palace_path():
    """Resolve palace path from config."""
    try:
        from .config import MempalaceConfig

        return MempalaceConfig().palace_path
    except Exception:
        default = os.path.join(os.path.expanduser("~"), ".mempalace", "palace")
        return default


def _paginate_ids(col, where=None):
    """Pull all IDs in a collection using pagination."""
    ids = []
    page = 1000
    offset = 0
    while True:
        try:
            r = col.get(where=where, include=[], limit=page, offset=offset)
        except Exception:
            try:
                r = col.get(where=where, include=[], limit=page)
                new_ids = [i for i in r["ids"] if i not in set(ids)]
                if not new_ids:
                    break
                ids.extend(new_ids)
                offset += len(new_ids)
                continue
            except Exception:
                break
        n = len(r["ids"]) if r["ids"] else 0
        if n == 0:
            break
        ids.extend(r["ids"])
        offset += n
        if n < page:
            break
    return ids


def scan_palace(palace_path=None, only_wing=None):
    """Scan the palace for corrupt/unfetchable IDs.

    Probes in batches of 100, falls back to per-ID on failure.
    Writes corrupt_ids.txt to the palace directory for the prune step.

    Returns (good_set, bad_set).
    """
    palace_path = palace_path or _get_palace_path()
    print(f"\n  Palace: {palace_path}")
    print("  Loading...")

    col = ChromaBackend().get_collection(palace_path, COLLECTION_NAME)

    where = {"wing": only_wing} if only_wing else None
    total = col.count()
    print(f"  Collection: {COLLECTION_NAME}, total: {total:,}")
    if only_wing:
        print(f"  Scanning wing: {only_wing}")

    print("\n  Step 1: listing all IDs...")
    t0 = time.time()
    all_ids = _paginate_ids(col, where=where)
    print(f"  Found {len(all_ids):,} IDs in {time.time() - t0:.1f}s\n")

    if not all_ids:
        print("  Nothing to scan.")
        return set(), set()

    print("  Step 2: probing each ID (batches of 100)...")
    t0 = time.time()
    good_set = set()
    bad_set = set()
    batch = 100

    for i in range(0, len(all_ids), batch):
        chunk = all_ids[i : i + batch]
        try:
            r = col.get(ids=chunk, include=["documents"])
            for got in r["ids"]:
                good_set.add(got)
            for mid in chunk:
                if mid not in good_set:
                    bad_set.add(mid)
        except Exception:
            for sid in chunk:
                try:
                    r = col.get(ids=[sid], include=["documents"])
                    if r["ids"]:
                        good_set.add(sid)
                    else:
                        bad_set.add(sid)
                except Exception:
                    bad_set.add(sid)

        if (i // batch) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + batch) / max(elapsed, 0.01)
            eta = (len(all_ids) - i - batch) / max(rate, 0.01)
            print(
                f"    {i + batch:>6}/{len(all_ids):>6}  "
                f"good={len(good_set):>6}  bad={len(bad_set):>6}  "
                f"eta={eta:.0f}s"
            )

    print(f"\n  Scan complete in {time.time() - t0:.1f}s")
    print(f"  GOOD: {len(good_set):,}")
    print(f"  BAD:  {len(bad_set):,}  ({len(bad_set) / max(len(all_ids), 1) * 100:.1f}%)")

    bad_file = os.path.join(palace_path, "corrupt_ids.txt")
    with open(bad_file, "w") as f:
        for bid in sorted(bad_set):
            f.write(bid + "\n")
    print(f"\n  Bad IDs written to: {bad_file}")
    return good_set, bad_set


def prune_corrupt(palace_path=None, confirm=False):
    """Delete corrupt IDs listed in corrupt_ids.txt."""
    palace_path = palace_path or _get_palace_path()
    bad_file = os.path.join(palace_path, "corrupt_ids.txt")

    if not os.path.exists(bad_file):
        print("  No corrupt_ids.txt found — run scan first.")
        return

    with open(bad_file) as f:
        bad_ids = [line.strip() for line in f if line.strip()]
    print(f"  {len(bad_ids):,} corrupt IDs queued for deletion")

    if not confirm:
        print("\n  DRY RUN — no deletions performed.")
        print("  Re-run with --confirm to actually delete.")
        return

    col = ChromaBackend().get_collection(palace_path, COLLECTION_NAME)
    before = col.count()
    print(f"  Collection size before: {before:,}")

    batch = 100
    deleted = 0
    failed = 0
    for i in range(0, len(bad_ids), batch):
        chunk = bad_ids[i : i + batch]
        try:
            col.delete(ids=chunk)
            deleted += len(chunk)
        except Exception:
            for sid in chunk:
                try:
                    col.delete(ids=[sid])
                    deleted += 1
                except Exception:
                    failed += 1
        if (i // batch) % 20 == 0:
            print(f"    deleted {deleted}/{len(bad_ids)}  (failed: {failed})")

    after = col.count()
    print(f"\n  Deleted: {deleted:,}")
    print(f"  Failed:  {failed:,}")
    print(f"  Collection size: {before:,} → {after:,}")


def rebuild_index(palace_path=None):
    """Rebuild the HNSW index from scratch.

    1. Extract all drawers from raw sqlite (immune to the SIGSEGV a corrupt
       HNSW segment causes inside the Rust bindings — col.get()/col.count()
       kill the whole process with exit 139, uncatchable by try/except)
    2. Back up ONLY chroma.sqlite3 (not the bloated HNSW files)
    3. Delete and recreate the collection with hnsw:space=cosine
    4. Upsert all drawers back
    """
    palace_path = palace_path or _get_palace_path()

    if not os.path.isdir(palace_path):
        print(f"\n  No palace found at {palace_path}")
        return

    print(f"\n{'=' * 55}")
    print("  MemPalace Repair — Index Rebuild")
    print(f"{'=' * 55}\n")
    print(f"  Palace: {palace_path}")

    backend = ChromaBackend()

    print("  Extracting drawers from sqlite (segfault-safe)...")
    t0 = time.time()
    all_ids, all_docs, all_metas = _extract_drawers_from_sqlite(palace_path)
    total = len(all_ids)
    print(f"  Drawers found: {total} (in {time.time() - t0:.1f}s)")

    if total == 0:
        print("  Nothing to repair.")
        return

    # Back up ONLY the SQLite database, not the bloated HNSW files
    sqlite_path = os.path.join(palace_path, "chroma.sqlite3")
    if os.path.exists(sqlite_path):
        backup_path = sqlite_path + ".backup"
        print(f"  Backing up chroma.sqlite3 ({os.path.getsize(sqlite_path) / 1e6:.0f} MB)...")
        shutil.copy2(sqlite_path, backup_path)
        print(f"  Backup: {backup_path}")

    # Rebuild with correct HNSW settings
    batch_size = 5000
    print("  Rebuilding collection with hnsw:space=cosine...")
    backend.delete_collection(palace_path, COLLECTION_NAME)
    new_col = backend.create_collection(palace_path, COLLECTION_NAME)

    filed = 0
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        batch_docs = all_docs[i : i + batch_size]
        batch_metas = all_metas[i : i + batch_size]
        new_col.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
        filed += len(batch_ids)
        print(f"  Re-filed {filed}/{len(all_ids)} drawers...")

    print(f"\n  Repair complete. {filed} drawers rebuilt.")
    print("  HNSW index is now clean with cosine distance metric.")
    print(f"\n{'=' * 55}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MemPalace repair tools")
    p.add_argument("command", choices=["scan", "prune", "rebuild"])
    p.add_argument("--palace", default=None, help="Palace directory path")
    p.add_argument("--wing", default=None, help="Scan only this wing")
    p.add_argument("--confirm", action="store_true", help="Actually delete corrupt IDs")
    args = p.parse_args()

    path = os.path.expanduser(args.palace) if args.palace else None

    if args.command == "scan":
        scan_palace(palace_path=path, only_wing=args.wing)
    elif args.command == "prune":
        prune_corrupt(palace_path=path, confirm=args.confirm)
    elif args.command == "rebuild":
        rebuild_index(palace_path=path)
