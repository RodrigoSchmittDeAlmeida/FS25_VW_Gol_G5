"""Build the FS25_VW_Gol_G5.zip mod archive inside dist/."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MOD_DIR = ROOT_DIR / "FS25_VW_Gol_G5"
DIST_DIR = ROOT_DIR / "dist"
ZIP_PATH = DIST_DIR / "FS25_VW_Gol_G5.zip"

SKIP_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
SKIP_DIRS = {"__pycache__", ".git"}


def build_zip(mod_dir: Path = MOD_DIR, zip_path: Path = ZIP_PATH) -> tuple[Path, int, int]:
    if not mod_dir.is_dir():
        raise FileNotFoundError(f"Mod directory not found: {mod_dir}")
    if not (mod_dir / "modDesc.xml").is_file():
        raise FileNotFoundError(f"modDesc.xml not found in {mod_dir}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for current_root, dirs, files in os.walk(mod_dir):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for filename in sorted(files):
                if filename in SKIP_NAMES or filename.endswith(".pyc"):
                    continue
                full_path = Path(current_root) / filename
                rel_posix = full_path.relative_to(mod_dir).as_posix()
                zf.write(full_path, arcname=rel_posix)
                file_count += 1

    size_bytes = zip_path.stat().st_size
    return zip_path, file_count, size_bytes


def main() -> int:
    zip_path, file_count, size_bytes = build_zip()
    print(f"[build_mod_zip] Created {zip_path} ({file_count} files, {size_bytes:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
