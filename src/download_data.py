from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def download(project_dir: Path) -> None:
    import kagglehub

    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(kagglehub.dataset_download("pratyushakar/rossmann-store-sales"))
    copied = []
    for name in ["train.csv", "store.csv"]:
        matches = list(source_dir.rglob(name))
        if not matches:
            raise FileNotFoundError(f"{name} was not found in downloaded dataset at {source_dir}")
        target = data_dir / name
        shutil.copy2(matches[0], target)
        copied.append(target)

    print("Copied:")
    for path in copied:
        print(f"- {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download(args.project_dir.resolve())
