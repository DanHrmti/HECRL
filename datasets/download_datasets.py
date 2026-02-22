from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

from huggingface_hub import hf_hub_download


REPO_ID = "DanHrmti/hecrl"

DATASETS: Dict[str, str] = {
    # "name": "filename.npz"
    "ppp-cube": "ppp-cube-noisy-v0-mv.npz",
    "stack-cube": "stack-cube-noisy-v0-mv.npz",
    "visual-scene": "visual-scene-noisy-v0.npz",
    "push-tetris": "push-tetris-constrained_random-v0.npz",
}


def download_dataset(name: str, target_dir: Path) -> None:
    filename = DATASETS[name]

    print(f"Downloading '{name}' from '{REPO_ID}' as '{filename}'...")
    hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
        local_dir=str(target_dir),
    )
    print(f"Finished downloading '{name}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download one or more datasets from the Hugging Face Hub.\n"
            "By default (no arguments) all datasets are downloaded."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help=(
            "Names of datasets to download. "
            "If omitted, all available datasets will be downloaded. "
            f"Available: {', '.join(sorted(DATASETS.keys()))}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = Path(__file__).resolve().parent / "data"

    if args.datasets:
        to_download: Iterable[str] = args.datasets
    else:
        to_download = DATASETS.keys()

    unknown = [name for name in to_download if name not in DATASETS]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(DATASETS.keys()))}"
        )

    for name in to_download:
        download_dataset(name, target_dir)


if __name__ == "__main__":
    main()

