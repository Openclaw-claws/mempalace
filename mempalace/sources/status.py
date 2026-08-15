"""Dry status diff between a source and core's ingest state (RFC 002 §4).

``mempalace sources status`` answers "if I ran ingest now, what would
happen?" without writing anything. It drives the adapter with the skip
flag permanently set, so only stat()-level ``SourceItemMetadata`` flows —
item bodies are never read. That makes a status pass safe on huge
sources and safe to run while another ingest holds the write lock.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import BaseSourceAdapter, SourceItemMetadata, SourceRef
from .context import PalaceContext
from .registry import get_adapter


def _load_state(palace_path: str, source: SourceRef) -> dict:
    """Load core-side ingest state for the (palace, source) pair.

    Mirrors the first-party state layout written by ``diary_ingest``
    (``~/.mempalace/state/diary_ingest_<sha(palace|dir)>.json``).
    """
    from ..diary_ingest import _state_file_for

    raw = source.local_path or source.uri or ""
    # Resolve exactly like the runner does (ingest_diaries resolve()s the
    # dir before hashing) — on macOS /tmp vs /private/tmp would otherwise
    # produce different state-file keys for the same directory.
    state_file = _state_file_for(str(palace_path), Path(raw).expanduser().resolve())
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text())
    except Exception:
        return {}


def _state_key(source: SourceRef, source_file: str) -> str:
    """Mirror the runner's state key: ``wing|filename`` (diary_ingest)."""
    wing = source.options.get("wing", "diary")
    return f"{wing}|{Path(source_file).name}"


def adapter_status(
    *,
    adapter_name: str,
    source: SourceRef,
    palace_path: str,
) -> dict:
    """Diff a source against core's ingest state. Never reads item bodies.

    Returns ``{"adapter", "items", "current", "stale", "new"}``:

    * ``current`` — adapter says the palace is up to date (no work)
    * ``stale``   — previously ingested, source changed since (re-ingest)
    * ``new``     — never ingested (initial ingest)
    """
    adapter: BaseSourceAdapter = get_adapter(adapter_name)
    state = _load_state(palace_path, source)

    # drawer_collection is None: with skip always requested, the adapter
    # must never reach the point of yielding a record, and we treat any
    # record that leaks through as a defensive no-op.
    context = PalaceContext(
        drawer_collection=None,
        knowledge_graph=None,
        palace_path=str(palace_path),
        adapter_name=adapter.name,
        adapter_version=adapter.adapter_version,
    )

    current = stale = new = 0
    for item in adapter.ingest(source=source, palace=context):
        if not isinstance(item, SourceItemMetadata):
            continue  # defensive: skip-always should suppress records
        prev = state.get(_state_key(source, item.source_file))
        if adapter.is_current(item=item, existing_metadata=prev):
            current += 1
        elif prev:
            stale += 1
        else:
            new += 1
        # Set AFTER consuming the metadata: the suspended generator sees
        # this on resume and skips reading the file entirely.
        context.skip_current_item()

    return {
        "adapter": adapter_name,
        "items": current + stale + new,
        "current": current,
        "stale": stale,
        "new": new,
    }
