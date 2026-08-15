"""
diary_ingest.py — Ingest daily summary files into the palace.

Extraction lives in :class:`mempalace.sources.diary.DiarySourceAdapter`
(the RFC 002 reference adapter); this module is the core-side write
orchestrator that drains the adapter's record stream and routes it into
the palace. Behavior is unchanged from the pre-adapter implementation:

- ONE drawer per (wing, day) — full verbatim content, upserted as the day grows.
- Closets pack topics up to CLOSET_CHAR_LIMIT, never split mid-topic.
- A re-ingest fully purges the prior day's closets before rebuilding so a
  shorter day never leaves orphans behind.
- Only new entries are processed by default (tracks entry count in a state
  file under ``~/.mempalace/state/`` — never inside the user's diary dir).
- Per-file ``mine_lock`` so concurrent ingest from two terminals can't race.
- Entities extracted and stamped on metadata for filterable search.

Usage:
    python -m mempalace.diary_ingest --dir ~/daily_summaries --palace ~/.mempalace/palace
    python -m mempalace.diary_ingest --dir ~/daily_summaries --palace ~/.mempalace/palace --force
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .palace import (
    build_closet_lines,
    get_closets_collection,
    get_collection,
    mine_lock,
    purge_file_closets,
    upsert_closet_lines,
)
from .sources.base import SourceItemMetadata, SourceRef
from .sources.context import PalaceContext
from .sources.diary import DiarySourceAdapter, split_entries


def _state_file_for(palace_path: str, diary_dir: Path) -> Path:
    """Return the per-(palace, diary-dir) state-file path under ~/.mempalace/state.

    Keyed by sha256 of (palace_path, diary_dir) so multiple diary folders
    pointing at the same palace each get an independent state file. The
    state file is *never* written inside the user's diary directory.
    """
    state_root = Path(os.path.expanduser("~")) / ".mempalace" / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{palace_path}|{diary_dir}".encode()).hexdigest()[:24]
    return state_root / f"diary_ingest_{key}.json"


def _diary_drawer_id(wing: str, date_str: str) -> str:
    """Stable, wing-scoped drawer ID. Two diaries (e.g. 'work' vs 'personal')
    sharing the same date never collide."""
    suffix = hashlib.sha256(f"{wing}|{date_str}".encode()).hexdigest()[:24]
    return f"drawer_diary_{suffix}"


def _diary_closet_id_base(wing: str, date_str: str) -> str:
    suffix = hashlib.sha256(f"{wing}|{date_str}".encode()).hexdigest()[:24]
    return f"closet_diary_{suffix}"


def ingest_diaries(
    diary_dir,
    palace_path,
    wing="diary",
    force=False,
):
    """Ingest daily summary files into the palace.

    Each date file gets ONE drawer keyed by ``(wing, date)`` and closets that
    pack topics atomically up to ``CLOSET_CHAR_LIMIT``. ``force=True`` rebuilds
    every entry's closets from scratch (purging stale ones); the default
    incremental mode only processes entries appended since the last run.
    """
    diary_dir = Path(diary_dir).expanduser().resolve()
    if not diary_dir.exists():
        print(f"Diary directory not found: {diary_dir}")
        return {"days_updated": 0, "closets_created": 0}

    state_file = _state_file_for(str(palace_path), diary_dir)
    if force or not state_file.exists():
        state: dict = {}
    else:
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            state = {}

    drawers_col = get_collection(palace_path)
    closets_col = get_closets_collection(palace_path)

    adapter = DiarySourceAdapter()
    context = PalaceContext(
        drawer_collection=drawers_col,
        closet_collection=closets_col,
        knowledge_graph=None,
        palace_path=str(palace_path),
        adapter_name=adapter.name,
        adapter_version=adapter.adapter_version,
    )

    days_updated = 0
    closets_created = 0
    # Per-item routing state, keyed by the SourceItemMetadata currently in
    # flight: (state_key, previous entry count).
    pending: dict[str, object] = {}

    stream = adapter.ingest(source=_diary_source(diary_dir, wing), palace=context)
    for record in stream:
        if isinstance(record, SourceItemMetadata):
            context._skip_requested = False
            state_key = f"{wing}|{Path(record.source_file).name}"
            prev_entry_count = state.get(state_key, {}).get("entry_count", 0)
            pending[state_key] = prev_entry_count
            if not force and adapter.is_current(
                item=record, existing_metadata=state.get(state_key)
            ):
                context.skip_current_item()
            continue

        # DrawerRecord — route into the palace.
        source_file = record.source_file
        date_str = record.metadata["date"]
        record_wing = record.metadata.get("wing", wing)
        state_key = f"{wing}|{Path(source_file).name}"
        prev_entry_count = int(pending.pop(state_key, 0))

        now_iso = datetime.now(timezone.utc).isoformat()
        drawer_id = _diary_drawer_id(record_wing, date_str)

        # Serialize per source — two terminals running ingest at once must
        # not interleave the upsert + closet-rebuild.
        with mine_lock(source_file):
            drawer_meta = dict(record.metadata)
            drawer_meta["source_file"] = source_file
            drawer_meta["filed_at"] = now_iso
            # RFC 002 §5.1 — core stamps the adapter identity on every
            # drawer it files, so provenance survives schema evolution.
            drawer_meta["adapter_name"] = adapter.name
            drawer_meta["adapter_version"] = adapter.adapter_version
            drawers_col.upsert(
                documents=[record.content],
                ids=[drawer_id],
                metadatas=[drawer_meta],
            )

            entries = split_entries(record.content)
            new_entries = entries if force else entries[prev_entry_count:]

            if new_entries:
                all_lines = []
                for header, body in new_entries:
                    entry_text = f"{header}\n{body}"
                    entry_lines = build_closet_lines(
                        source_file, [drawer_id], entry_text, record_wing, "daily"
                    )
                    all_lines.extend(entry_lines)

                if all_lines:
                    closet_id_base = _diary_closet_id_base(record_wing, date_str)
                    closet_meta = {
                        "date": date_str,
                        "wing": record_wing,
                        "room": "daily",
                        "source_file": source_file,
                        "filed_at": now_iso,
                    }
                    if record.metadata.get("entities"):
                        closet_meta["entities"] = record.metadata["entities"]
                    # On a force rebuild, wipe any leftover numbered closets
                    # from a longer prior run before re-writing.
                    if force:
                        purge_file_closets(closets_col, source_file)
                    n = upsert_closet_lines(closets_col, closet_id_base, all_lines, closet_meta)
                    closets_created += n

            state[state_key] = {
                "size": record.metadata["size"],
                "entry_count": len(entries),
                "ingested_at": now_iso,
            }
        days_updated += 1

    state_file.write_text(json.dumps(state, indent=2))
    if days_updated:
        print(f"Diary: {days_updated} days updated, {closets_created} new closets")

    return {"days_updated": days_updated, "closets_created": closets_created}


def _diary_source(diary_dir: Path, wing: str) -> SourceRef:
    """Build the SourceRef pointing the diary adapter at ``diary_dir``."""
    return SourceRef(local_path=str(diary_dir), options={"wing": wing})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest daily summaries into the palace")
    parser.add_argument("--dir", required=True, help="Path to daily_summaries directory")
    parser.add_argument("--palace", default=os.path.expanduser("~/.mempalace/palace"))
    parser.add_argument("--wing", default="diary")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ingest_diaries(args.dir, args.palace, wing=args.wing, force=args.force)
