"""First-party diary source adapter (RFC 002 reference implementation).

This is the adapter community contributors should copy: it migrates the
smallest in-tree miner (``mempalace/diary_ingest.py``) onto
:class:`BaseSourceAdapter` without changing palace behavior. The adapter is
extraction-only — it turns diary ``.md`` files into typed records and never
writes to the palace itself; core (``diary_ingest.ingest_diaries``) drains
the record stream and handles routing, closets, and state.

What this adapter demonstrates (the checklist a new adapter needs):

* ``ingest`` yields ``SourceItemMetadata`` BEFORE reading the file, so core
  can call :meth:`is_current` and skip unchanged days without paying for I/O.
* One ``DrawerRecord`` per diary file — ``whole_record`` mode, verbatim
  content, flat scalar metadata only (the chroma constraint, RFC 001 §1.4).
* ``describe_schema`` declares every metadata field it stamps.
* Incremental support via a ``version`` string (``size:entry_count``) that
  core compares against its per-source state.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

from .base import (
    AdapterSchema,
    BaseSourceAdapter,
    DrawerRecord,
    FieldSpec,
    RouteHint,
    SourceItemMetadata,
    SourceNotFoundError,
    SourceRef,
    SourceSummary,
)
from .context import PalaceContext

DIARY_ENTRY_RE = re.compile(r"^## .+", re.MULTILINE)
_DATE_STEM_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# Files below this many characters of content are skipped — matches the
# legacy miner's noise floor for empty/template days.
_MIN_FILE_CHARS = 50


def split_entries(text: str) -> list[tuple[str, str]]:
    """Split diary text into (header, body) pairs per ``##`` entry.

    Shared with the write side (``diary_ingest``) so closets are built from
    exactly the same split the adapter counted.
    """
    parts = DIARY_ENTRY_RE.split(text)
    headers = DIARY_ENTRY_RE.findall(text)
    entries = []
    for i, header in enumerate(headers):
        body = parts[i + 1] if i + 1 < len(parts) else ""
        entries.append((header.strip(), body.strip()))
    return entries


class DiarySourceAdapter(BaseSourceAdapter):
    """Daily-summary ``.md`` files → one verbatim drawer per (wing, day)."""

    name = "diary"
    adapter_version = "1.0.0"
    capabilities = frozenset({"supports_incremental"})
    supported_modes = frozenset({"whole_record"})
    # The adapter applies no transformations to source bytes: drawers hold
    # the file's exact text (utf-8 decoded, same as the legacy miner).
    declared_transformations = frozenset()
    default_privacy_class = "pii_potential"

    def ingest(
        self,
        *,
        source: SourceRef,
        palace: PalaceContext,
    ) -> Iterator[object]:
        dir_path = self._resolve_dir(source)
        wing = source.options.get("wing", "diary")

        for diary_path in sorted(dir_path.glob("*.md")):
            date_match = _DATE_STEM_RE.match(diary_path.stem)
            if not date_match:
                continue
            date_str = date_match.group(1)

            # Version from stat() — cheap enough to compute for every file
            # without reading it. Core compares this against its state.
            stat = diary_path.stat()
            version = f"{stat.st_size}:{stat.st_mtime_ns}"

            yield SourceItemMetadata(
                source_file=str(diary_path),
                version=version,
                size_hint=stat.st_size,
                route_hint=RouteHint(wing=wing, room="daily"),
            )
            if palace._skip_requested:
                # Core says the palace already has this day up to date.
                continue

            text = diary_path.read_text(encoding="utf-8", errors="replace")
            if len(text.strip()) < _MIN_FILE_CHARS:
                continue

            yield self._build_record(text, str(diary_path), date_str, wing)

    def is_current(
        self,
        *,
        item: SourceItemMetadata,
        existing_metadata: Optional[dict],
    ) -> bool:
        if not existing_metadata:
            return False
        size = existing_metadata.get("size")
        if not isinstance(size, int):
            return False
        try:
            item_size = int(item.version.split(":", 1)[0])
        except ValueError:
            return False
        return item_size == size

    def describe_schema(self) -> AdapterSchema:
        return AdapterSchema(
            version=self.adapter_version,
            fields={
                "date": FieldSpec(
                    type="string", required=True, description="Diary day (YYYY-MM-DD)", indexed=True
                ),
                "wing": FieldSpec(
                    type="string",
                    required=True,
                    description="Wing the diary files into",
                    indexed=True,
                ),
                "room": FieldSpec(
                    type="string", required=True, description="Room (always 'daily')", indexed=True
                ),
                "source_session": FieldSpec(
                    type="string",
                    required=True,
                    description="Session tag (always 'daily_diary')",
                    indexed=False,
                ),
                "size": FieldSpec(
                    type="int",
                    required=True,
                    description="File size in chars at ingest time (incremental key)",
                    indexed=False,
                ),
                "entry_count": FieldSpec(
                    type="int",
                    required=True,
                    description="Number of ## entries in the file",
                    indexed=False,
                ),
                "entities": FieldSpec(
                    type="delimiter_joined_string",
                    required=False,
                    description="Semicolon-joined entity names for filtering",
                    indexed=True,
                    delimiter=";",
                ),
                "adapter_name": FieldSpec(
                    type="string",
                    required=True,
                    description="Stamped by core (RFC 002 §5.1)",
                    indexed=True,
                ),
                "adapter_version": FieldSpec(
                    type="string",
                    required=True,
                    description="Stamped by core (RFC 002 §5.1)",
                    indexed=False,
                ),
            },
        )

    def source_summary(self, *, source: SourceRef) -> SourceSummary:
        try:
            dir_path = self._resolve_dir(source)
        except SourceNotFoundError:
            return SourceSummary(description="diary directory not found", item_count=0)
        dated = [p for p in dir_path.glob("*.md") if _DATE_STEM_RE.match(p.stem)]
        return SourceSummary(
            description=f"daily summary .md files in {dir_path}",
            item_count=len(dated),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_dir(source: SourceRef) -> Path:
        raw = source.local_path or source.uri
        if not raw:
            raise SourceNotFoundError("diary adapter requires source.local_path")
        dir_path = Path(raw).expanduser().resolve()
        if not dir_path.is_dir():
            raise SourceNotFoundError(f"diary directory not found: {dir_path}")
        return dir_path

    @staticmethod
    def _build_record(text: str, source_file: str, date_str: str, wing: str) -> DrawerRecord:
        from pathlib import Path

        from ..miner import _extract_entities_for_metadata

        metadata: dict = {
            "date": date_str,
            "wing": wing,
            "room": "daily",
            "source_session": "daily_diary",
            # Bytes, matching the stat()-based version string — char counts
            # and byte counts diverge on unicode and would break is_current.
            "size": Path(source_file).stat().st_size,
            "entry_count": len(split_entries(text)),
        }
        entities = _extract_entities_for_metadata(text)
        if entities:
            metadata["entities"] = entities
        return DrawerRecord(
            content=text,
            source_file=source_file,
            chunk_index=0,
            metadata=metadata,
            route_hint=RouteHint(wing=wing, room="daily"),
        )
