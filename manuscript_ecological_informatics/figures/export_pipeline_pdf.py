#!/usr/bin/env python3
"""Export figures/pipeline.svg -> figures/pipeline.pdf for LaTeX includegraphics.

LaTeX \\includesvg often drops text (CSS/fonts). Commit pipeline.pdf and use:
  \\includegraphics[width=0.92\\linewidth]{figures/pipeline.pdf}

Requires Google Chrome or Chromium (headless print-to-pdf).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SVG = HERE / "pipeline.svg"
PDF = HERE / "pipeline.pdf"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]


def find_browser() -> str | None:
    for path in CHROME_CANDIDATES:
        if Path(path).is_file():
            return path
    for name in ("chrome", "google-chrome", "chromium", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    return None


def main() -> int:
    if not SVG.is_file():
        print(f"Missing {SVG}", file=sys.stderr)
        return 1
    browser = find_browser()
    if not browser:
        print(
            "Chrome/Chromium/Edge not found. Install Chrome or export manually:\n"
            "  Inkscape: inkscape pipeline.svg --export-type=pdf --export-filename=pipeline.pdf",
            file=sys.stderr,
        )
        return 2
    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF}",
        SVG.resolve().as_uri(),
    ]
    print("Exporting:", SVG.name, "->", PDF.name)
    subprocess.run(cmd, check=True, timeout=120)
    print("Wrote", PDF, f"({PDF.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
