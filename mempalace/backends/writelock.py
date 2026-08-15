"""Cross-process write lock for palace mutations.

ChromaDB's PersistentClient is not safe for concurrent multi-process writes
to the same palace directory: independent writers racing ``add``/``upsert``
can corrupt the on-disk HNSW segment and segfault the Rust graph-walk
(observed 2026-08-15: six concurrent MCP server instances, one per Claude
session, corrupted ``mempalace_drawers``).

This module serializes mutations at the backend chokepoint
(``ChromaCollection.add/upsert/update/delete``) with an OS-level advisory
file lock scoped to the palace directory:

* POSIX: ``fcntl.flock(LOCK_EX)`` — the kernel releases the lock when the
  owning process dies, so there is no stale-lock recovery problem.
* Windows: ``msvcrt.locking`` retry loop (same semantics as ``mine_lock``).
* The lock file is ``<palace>/.writes.lock`` — dot-prefixed so the
  ``quarantine_stale_hnsw`` segment scan skips it.
* On timeout the lock fails LOUDLY (``WriteLockTimeoutError``). Silently
  writing without the lock is forbidden — an unlocked write is exactly the
  corruption this module exists to prevent.
* Waits longer than ``_CONTENTION_WARN_SECONDS`` are logged as contention
  events, giving operators the data to tune or escalate later.
"""

import contextlib
import logging
import os
import time

from .base import BackendError

logger = logging.getLogger(__name__)

_LOCK_FILENAME = ".writes.lock"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_CONTENTION_WARN_SECONDS = 1.0
_POLL_INTERVAL_SECONDS = 0.025


class WriteLockTimeoutError(BackendError):
    """Raised when a palace write lock cannot be acquired within the timeout.

    Callers MUST NOT retry the write without the lock — surface the error.
    """


def _resolve_timeout(timeout: float | None) -> float:
    """Return the effective timeout: explicit arg > env var > default."""
    if timeout is not None:
        return float(timeout)
    raw = os.environ.get("MEMPALACE_WRITE_LOCK_TIMEOUT", "")
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT_SECONDS


def _try_acquire(lf) -> bool:
    """Attempt one non-blocking acquisition. Returns True on success."""
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(lf.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl

    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _release(lf) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lf, fcntl.LOCK_UN)
    except OSError:
        logger.debug("Releasing palace write lock failed (fd already closed?)", exc_info=True)


@contextlib.contextmanager
def palace_write_lock(palace_path: str, timeout: float | None = None):
    """Hold an exclusive cross-process lock for a palace mutation.

    Args:
        palace_path: the palace directory (lock file lives inside it).
        timeout: seconds to wait before raising ``WriteLockTimeoutError``.
            Defaults to ``MEMPALACE_WRITE_LOCK_TIMEOUT`` or 30 s.

    Raises:
        WriteLockTimeoutError: if the lock is not acquired in time. Never
            yields unlocked.
    """
    lock_path = os.path.join(palace_path, _LOCK_FILENAME)
    effective_timeout = _resolve_timeout(timeout)

    lf = open(lock_path, "w")
    waited = 0.0
    acquired = _try_acquire(lf)
    while not acquired and waited < effective_timeout:
        time.sleep(_POLL_INTERVAL_SECONDS)
        waited += _POLL_INTERVAL_SECONDS
        acquired = _try_acquire(lf)

    if not acquired:
        lf.close()
        holder = ""
        try:
            with open(lock_path) as probe:
                holder = probe.read().strip()
        except OSError:
            pass
        raise WriteLockTimeoutError(
            f"could not acquire palace write lock {lock_path} within "
            f"{effective_timeout}s (holder: {holder or 'unknown'})"
        )

    # Holder diagnostics for timeout messages: flock never goes stale, but
    # knowing WHO held it when you timed out halves the debugging time.
    try:
        lf.seek(0)
        lf.truncate()
        lf.write(f"pid={os.getpid()} since={time.time():.0f}\n")
        lf.flush()
    except OSError:
        pass

    if waited >= _CONTENTION_WARN_SECONDS:
        logger.warning("Palace write lock contention: waited %.2fs for %s", waited, lock_path)

    try:
        yield
    finally:
        _release(lf)
        lf.close()
