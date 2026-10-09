#!/usr/bin/env python3
"""repair-drawers-from-sqlite — rebuild mempalace_drawers when the HNSW segment
hard-segfaults the ChromaDB Rust binding (SIGSEGV, exit 139).

`mempalace repair` extracts via col.get(), which dies when the segment is
corrupt — try/except cannot catch SIGSEGV. This script never touches the
corrupt segment through the binding until AFTER it is deleted:

  phase backup   — copy chroma.sqlite3 to trash/palace-bak-<date>/
  phase extract  — read (id, document, metadata) from raw sqlite → drawers-extract.jsonl
  phase verify   — extracted count must equal sqlite embeddings count
  phase drop     — surgical sqlite DELETE of the collection + its segments,
                   then remove the corrupt segment dirs from disk
  phase refile   — create the collection fresh (hnsw:space=cosine) and upsert
                   in batches; the default local ONNX EF re-embeds ($0, offline)
  phase check    — count + query + get in this same process; exit 0 only if sane

Resumable: skips any phase whose checkpoint exists in .repair-drawers-state.json.
Usage: python3 scripts/repair-drawers-from-sqlite.py [--palace PATH]
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime

PALACE = os.path.expanduser("~/.mempalace/palace")
COLLECTION = "mempalace_drawers"
STATE_FILE = os.path.join(PALACE, ".repair-drawers-state.json")
EXTRACT_FILE = os.path.join(PALACE, "drawers-extract.jsonl")
BATCH = 1000


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"done": []}


def mark(phase):
    st = load_state()
    st["done"].append(phase)
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)


def db_conn():
    con = sqlite3.connect(os.path.join(PALACE, "chroma.sqlite3"))
    con.row_factory = sqlite3.Row
    return con


def get_collection_id(con):
    row = con.execute("SELECT id FROM collections WHERE name=?", (COLLECTION,)).fetchone()
    return row["id"] if row else None


def phase_backup():
    src = os.path.join(PALACE, "chroma.sqlite3")
    bak_dir = os.path.expanduser("~/.mempalace/trash/palace-bak-20261009-pre-drawers-fix")
    os.makedirs(bak_dir, exist_ok=True)
    dst = os.path.join(bak_dir, "chroma.sqlite3")
    if os.path.exists(dst):
        log(f"backup ya existe: {dst}")
        return
    log(f"backup {src} → {dst} ({os.path.getsize(src)/1e9:.1f} GB)...")
    shutil.copy2(src, dst)
    log("backup OK")


def phase_extract():
    con = db_conn()
    cid = get_collection_id(con)
    if not cid:
        log("ERROR: no existe la colección drawers")
        sys.exit(1)
    segs = [r["id"] for r in con.execute("SELECT id FROM segments WHERE collection=?", (cid,))]
    log(f"collection={cid} segmentos={segs}")

    total = con.execute(
        "SELECT COUNT(*) c FROM embeddings WHERE segment_id IN (%s)" % ",".join("?" * len(segs)),
        segs,
    ).fetchone()["c"]
    log(f"embeddings a extraer: {total}")

    n = 0
    t0 = time.time()
    with open(EXTRACT_FILE, "w") as out:
        # stream todas las filas de metadatos de drawers y agrupalas por embedding
        rows = con.execute(
            """SELECT e.id AS eid, e.embedding_id, em.key, em.string_value, em.int_value,
                      em.float_value, em.bool_value
               FROM embeddings e LEFT JOIN embedding_metadata em ON em.id = e.id
               WHERE e.segment_id IN (%s) ORDER BY e.id""" % ",".join("?" * len(segs)),
            segs,
        )
        cur_eid = None
        rec = None

        def flush(rec):
            nonlocal n
            if rec is not None:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 50000 == 0:
                    rate = n / (time.time() - t0)
                    log(f"  {n:,} extraídos ({rate:.0f}/s)")

        for r in rows:
            if r["eid"] != cur_eid:
                flush(rec)
                cur_eid = r["eid"]
                rec = {"id": r["embedding_id"], "doc": None, "meta": {}}
            if r["key"] is None:
                continue
            if r["key"] == "chroma:document":
                rec["doc"] = r["string_value"]
            else:
                v = r["string_value"]
                if v is None and r["int_value"] is not None:
                    v = r["int_value"]
                elif v is None and r["float_value"] is not None:
                    v = r["float_value"]
                elif v is None and r["bool_value"] is not None:
                    v = bool(r["bool_value"])
                rec["meta"][r["key"]] = v
        flush(rec)
    log(f"extracción completa: {n:,} registros en {EXTRACT_FILE}")
    if n != total:
        log(f"ERROR: extraídos {n} != esperados {total}")
        sys.exit(1)
    with open(EXTRACT_FILE + ".count", "w") as f:
        f.write(str(n))


def phase_drop():
    con = db_conn()
    cid = get_collection_id(con)
    if not cid:
        log("colección ya no existe — drop omitido")
        return
    segs = [r["id"] for r in con.execute("SELECT id FROM segments WHERE collection=?", (cid,))]
    eids = [r[0] for r in con.execute(
        "SELECT id FROM embeddings WHERE segment_id IN (%s)" % ",".join("?" * len(segs)), segs)]
    log(f"borrando {len(eids):,} embeddings de sqlite (cirugía, backup ya tomado)")
    con.execute("DELETE FROM embedding_fulltext_search WHERE rowid IN (%s)"
                % ",".join("?" * len(eids)), eids) if eids else None
    con.execute("DELETE FROM embedding_metadata WHERE id IN (%s)"
                % ",".join("?" * len(eids)), eids) if eids else None
    con.execute("DELETE FROM embeddings WHERE segment_id IN (%s)" % ",".join("?" * len(segs)), segs)
    con.execute("DELETE FROM segments WHERE collection=?", (cid,))
    con.execute("DELETE FROM collection_metadata WHERE collection_id=?", (cid,))
    con.execute("DELETE FROM collections WHERE id=?", (cid,))
    con.commit()
    log("rows borradas; eliminando dirs de segmento del disco")
    for seg in segs:
        d = os.path.join(PALACE, seg)
        if os.path.isdir(d):
            shutil.rmtree(d)
            log(f"  rm {d}")
    # VACUUM al final para reclamar espacio (4.8GB sqlite)
    log("VACUUM (puede tardar)...")
    con.execute("VACUUM")
    con.close()
    log("drop completo")


def phase_refile():
    import chromadb

    count = int(open(EXTRACT_FILE + ".count").read().strip())
    log(f"re-filing {count:,} drawers con embedding local (ONNX)...")
    client = chromadb.PersistentClient(path=PALACE)
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    filed = 0
    t0 = time.time()
    with open(EXTRACT_FILE) as f:
        for line in f:
            r = json.loads(line)
            ids.append(r["id"])
            docs.append(r["doc"] if r["doc"] is not None else "")
            metas.append(r["meta"] or None)
            if len(ids) >= BATCH:
                col.upsert(ids=ids, documents=docs, metadatas=metas)
                filed += len(ids)
                ids, docs, metas = [], [], []
                if filed % 20000 < BATCH:
                    rate = filed / (time.time() - t0)
                    eta = (count - filed) / rate / 60 if rate > 0 else 0
                    log(f"  {filed:,}/{count:,} ({rate:.0f}/s, ETA {eta:.0f} min)")
    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
        filed += len(ids)
    log(f"re-file completo: {filed:,}")
    if filed != count:
        log(f"ERROR: re-filed {filed} != extraídos {count}")
        sys.exit(1)


def phase_check():
    import chromadb

    client = chromadb.PersistentClient(path=PALACE)
    cols = {c.name for c in client.list_collections()}
    assert COLLECTION in cols, "drawers no existe"
    drawers = client.get_collection(COLLECTION)
    n = drawers.count()
    log(f"drawers count = {n:,}")
    r = drawers.query(query_texts=["memory palace test"], n_results=3)
    assert r["ids"] and r["ids"][0], "query vacía"
    log(f"query OK → {r['ids'][0][:2]}")
    g = drawers.get(limit=2, include=["documents"])
    assert g["ids"], "get vacío"
    for other in ("mempalace_closets", "mempalace_compressed"):
        c = client.get_collection(other)
        log(f"{other} count = {c.count():,}")
    log("STORE SANA ✔")


def main():
    global PALACE
    ap = argparse.ArgumentParser()
    ap.add_argument("--palace", default=PALACE)
    args = ap.parse_args()
    PALACE = os.path.abspath(args.palace)

    # Toma el write-lock de mempalace para que los hooks (SessionStart/Stop)
    # se serialicen con esta reparación en vez de escribir en paralelo.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mempalace.backends.writelock import palace_write_lock

    with palace_write_lock(PALACE, timeout=60):
        phases = [("backup", phase_backup), ("extract", phase_extract), ("drop", phase_drop),
                  ("refile", phase_refile), ("check", phase_check)]
        done = set(load_state()["done"])
        for name, fn in phases:
            if name in done:
                log(f"fase {name}: YA HECHA (skip)")
                continue
            log(f"=== fase {name} ===")
            fn()
            mark(name)
    log("REPARACIÓN COMPLETA")


if __name__ == "__main__":
    main()
