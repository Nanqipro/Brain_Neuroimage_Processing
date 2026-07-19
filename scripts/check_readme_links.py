#!/usr/bin/env python3
"""Verify that local Markdown links in README.md resolve inside the repository."""

from __future__ import annotations

from pathlib import Path
import re
import sys
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def local_targets(markdown: str) -> list[str]:
    targets = []
    for raw_target in MARKDOWN_LINK.findall(markdown):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        targets.append(unquote(target.split("#", 1)[0]))
    return targets


def main() -> int:
    missing = []
    for target in local_targets(README.read_text(encoding="utf-8")):
        path = (README.parent / target).resolve()
        try:
            path.relative_to(REPO_ROOT)
        except ValueError:
            missing.append((target, "outside repository"))
            continue
        if not path.exists():
            missing.append((target, "not found"))

    if missing:
        for target, reason in missing:
            print(f"ERROR {target}: {reason}", file=sys.stderr)
        return 1

    print("All local README links resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
