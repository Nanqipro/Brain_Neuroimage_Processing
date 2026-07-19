#!/usr/bin/env python3
"""Compile first-party Python sources in memory without creating ``.pyc`` files."""

from __future__ import annotations

from pathlib import Path
import sys
import tokenize


REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {
    ".git",
    ".local",
    ".venv",
    "node_modules",
    "CaImAn-main",
    "suite2p-main",
    "DeepCAD-master",
    "DeepCAD-RT-main",
    "deepinterpolation-master",
}


def source_files() -> list[Path]:
    return sorted(
        path
        for path in REPO_ROOT.rglob("*.py")
        if not SKIP_PARTS.intersection(path.relative_to(REPO_ROOT).parts)
    )


def main() -> int:
    failures: list[tuple[Path, Exception]] = []
    files = source_files()
    for path in files:
        try:
            with tokenize.open(path) as source:
                compile(source.read(), str(path.relative_to(REPO_ROOT)), "exec")
        except (OSError, SyntaxError, UnicodeError) as exc:
            failures.append((path, exc))

    if failures:
        for path, exc in failures:
            print(f"ERROR {path.relative_to(REPO_ROOT)}: {exc}", file=sys.stderr)
        print(f"Checked {len(files)} files; {len(failures)} failed.", file=sys.stderr)
        return 1

    print(f"Checked {len(files)} first-party Python files; syntax is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
