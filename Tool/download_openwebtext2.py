#!/usr/bin/env python3
"""
Download OpenWebText2 datasets from official links listed in:
https://openwebtext2.readthedocs.io/en/latest/

Usage examples:
  python Tool/download_openwebtext2.py --variant clean
  python Tool/download_openwebtext2.py --variant raw --extract
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path


DATASET_URLS = {
    "clean": "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar",
    "raw": "https://eaidata.bmk.sh/data/openwebtext2_raw.tar",
}

CHUNK_SIZE = 1024 * 1024  # 1 MB


def format_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        print(f"[download] File already exists, skipping: {destination}")
        return

    print(f"[download] URL: {url}")
    print(f"[download] Destination: {destination}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response, destination.open("wb") as out:
        total = response.headers.get("Content-Length")
        total_size = int(total) if total and total.isdigit() else None
        downloaded = 0

        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)

            if total_size:
                pct = (downloaded / total_size) * 100.0
                print(
                    f"\r[download] {pct:6.2f}%  {format_bytes(downloaded)} / {format_bytes(total_size)}",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r[download] {format_bytes(downloaded)}", end="", flush=True)

    print("\n[download] Completed")


def extract_tar(archive_path: Path, output_dir: Path) -> None:
    print(f"[extract] Archive: {archive_path}")
    print(f"[extract] Output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r") as tar:
        members = tar.getmembers()
        total = len(members)
        for idx, member in enumerate(members, start=1):
            tar.extract(member, path=output_dir)
            if idx % 25 == 0 or idx == total:
                print(f"\r[extract] {idx}/{total} files", end="", flush=True)

    print("\n[extract] Completed")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download OpenWebText2 dataset.")
    parser.add_argument(
        "--variant",
        choices=sorted(DATASET_URLS.keys()),
        default="clean",
        help="Dataset variant from OpenWebText2 docs.",
    )
    parser.add_argument(
        "--output-dir",
        default="Data/openwebtext2",
        help="Folder to place downloaded files.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded .tar archive after download.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    url = DATASET_URLS[args.variant]

    output_dir = Path(args.output_dir)
    filename = os.path.basename(url)
    archive_path = output_dir / filename

    try:
        download_file(url, archive_path)
        if args.extract:
            extract_dir = output_dir / f"{args.variant}_extracted"
            extract_tar(archive_path, extract_dir)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[done] {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

