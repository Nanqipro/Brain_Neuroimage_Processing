"""File-handling helpers for the local calcium-analysis API.

The API is intended for trusted, local research workflows.  These helpers keep
client supplied names inside the configured storage directories and enforce a
small allow-list of spreadsheet formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Iterable
from uuid import uuid4


ALLOWED_SPREADSHEET_SUFFIXES = frozenset({".xlsx", ".xls"})
DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class InvalidFile(ValueError):
    """Raised when a client-supplied file name or payload is not acceptable."""


def safe_client_filename(
    filename: str | None,
    *,
    allowed_suffixes: Iterable[str] = ALLOWED_SPREADSHEET_SUFFIXES,
) -> str:
    """Return a basename-only client filename after validating its suffix."""

    normalized = (filename or "").replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1].strip()
    if not basename or basename in {".", ".."}:
        raise InvalidFile("文件名无效")

    suffixes = {suffix.lower() for suffix in allowed_suffixes}
    if Path(basename).suffix.lower() not in suffixes:
        supported = ", ".join(sorted(suffixes))
        raise InvalidFile(f"仅支持以下文件格式：{supported}")

    return basename


def unique_upload_path(directory: Path, filename: str | None) -> Path:
    """Create a collision-resistant path below *directory* for an upload."""

    safe_name = safe_client_filename(filename)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{uuid4().hex}_{safe_name}"


def resolve_existing_file(
    directory: Path,
    filename: str | None,
    *,
    allowed_suffixes: Iterable[str] = ALLOWED_SPREADSHEET_SUFFIXES,
) -> Path:
    """Resolve an existing file while preventing traversal outside *directory*."""

    safe_name = safe_client_filename(filename, allowed_suffixes=allowed_suffixes)
    root = directory.resolve()
    candidate = (root / safe_name).resolve()
    if candidate.parent != root or not candidate.is_file():
        raise FileNotFoundError(safe_name)
    return candidate


def copy_limited(
    source: BinaryIO,
    destination: Path,
    *,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    chunk_size: int = 1024 * 1024,
) -> int:
    """Copy a stream to disk, deleting partial output when it exceeds the limit."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with destination.open("xb") as output:
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise InvalidFile(
                        f"上传文件不能超过 {max_bytes // (1024 * 1024)} MiB"
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return written
