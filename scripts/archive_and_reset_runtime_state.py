#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""归档并重置运行态记忆与检查点，避免脏数据持续影响输出。"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def move_if_exists(src: Path, archive_dir: Path) -> bool:
    if not src.exists():
        return False
    archive_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(archive_dir / src.name))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive and reset agent runtime state.")
    parser.add_argument("--project-root", default=".", help="Project root path")
    parser.add_argument("--dry-run", action="store_true", help="Only print actions without moving files")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = root / ".runtime_state_archive" / timestamp

    targets = [
        root / ".agent_memory",
        root / "checkpoints.db",
        root / "checkpoints.db-shm",
        root / "checkpoints.db-wal",
    ]

    moved = []
    for target in targets:
        if not target.exists():
            continue
        if args.dry_run:
            print(f"[DRY-RUN] would archive: {target}")
            moved.append(str(target))
            continue
        if move_if_exists(target, archive_dir):
            moved.append(str(target))
            print(f"archived: {target} -> {archive_dir / target.name}")

    if not moved:
        print("nothing to archive")
        return

    if args.dry_run:
        print(f"[DRY-RUN] archive dir would be: {archive_dir}")
    else:
        print(f"archive complete: {archive_dir}")
        print("next startup will rebuild fresh memory and checkpoint state")


if __name__ == "__main__":
    main()
