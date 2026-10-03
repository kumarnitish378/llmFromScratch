#!/usr/bin/env python3
"""Dispatcher for crawling, processing and the existing C++ n-gram baseline."""
import argparse
import importlib.util
import sqlite3
import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


def train_chat():
    corpus = ROOT / "Data" / "clean_training_corpus.txt"
    if not corpus.exists() or corpus.stat().st_size == 0:
        print("No non-empty clean_training_corpus.txt found. Run process first.")
        return 2

    build = shutil.which("make") or shutil.which("mingw32-make")
    if not build:
        print("make/mingw32-make not found. Install it or build the C++ project first.")
        return 2

    rc = run([build])
    if rc:
        return rc

    exe = ROOT / "build" / "app.exe"
    if not exe.exists():
        print("Build completed but build/app.exe was not found.")
        return 2

    print(
        "Running the existing C++ n-gram chat baseline. "
        "This is NOT Transformer training."
    )
    # These inputs depend on the current C++ menu and remain project-specific.
    return subprocess.run(
        [str(exe)], cwd=ROOT, input="4\n1\n0\n", text=True
    ).returncode


def show_status():
    db_path = ROOT / "Data" / "crawl" / "queue.sqlite3"
    raw_path = ROOT / "Data" / "crawl" / "raw_pages.jsonl"
    if not db_path.exists():
        print("No crawl queue found.")
    else:
        try:
            with sqlite3.connect(db_path) as con:
                rows = con.execute(
                    "SELECT status, COUNT(*) FROM urls GROUP BY status ORDER BY status"
                ).fetchall()
                print("Crawl queue:")
                for status, count in rows:
                    print(f"  {status}: {count}")
                errors = con.execute(
                    "SELECT url, last_error FROM urls "
                    "WHERE last_error IS NOT NULL AND last_error != '' "
                    "ORDER BY updated_at DESC LIMIT 10"
                ).fetchall()
                if errors:
                    print("Recent crawl errors:")
                    for url, error in errors:
                        print(f"  {url}: {error}")
        except sqlite3.Error as exc:
            print(f"Could not read crawl queue: {exc}")
            return 2

    count = 0
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
    print("Raw records:", count)
    return 0


def main():
    parser = argparse.ArgumentParser(description="LLM data pipeline dispatcher")
    parser.add_argument(
        "--mode",
        choices=["smoke-test", "crawl", "process", "train", "pipeline", "status"],
        required=True,
    )
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--max-disk-gb", type=float)
    parser.add_argument("--max-ram-gb", type=float)
    parser.add_argument("--seed-file")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    python = sys.executable

    if args.mode == "smoke-test":
        for module in ("yaml", "crawl4ai", "psutil"):
            if importlib.util.find_spec(module) is None:
                print(f"Missing dependency: {module}")
                return 2
        from data_pipeline.crawl import canonical_url, allowed
        assert canonical_url("https://Example.com/a#fragment") == "https://example.com/a"
        assert allowed("https://sub.example.com/a", {"example.com"})
        assert not allowed("https://example.org/a", {"example.com"})
        print("Smoke tests passed; no crawl or training started.")
        return 0

    if args.mode == "status":
        return show_status()
    if args.mode == "train":
        return train_chat()

    crawl = [python, "data_pipeline/crawl.py"]
    for option, value in (
        ("--max-hours", args.max_hours),
        ("--max-pages", args.max_pages),
        ("--max-disk-gb", args.max_disk_gb),
        ("--max-ram-gb", args.max_ram_gb),
        ("--seed-file", args.seed_file),
    ):
        if value is not None:
            crawl.extend([option, str(value)])
    if args.resume:
        crawl.append("--resume")
    if args.dry_run:
        crawl.append("--dry-run")

    if args.mode == "crawl":
        return run(crawl)

    process = [python, "data_pipeline/process.py"]
    if args.mode == "process":
        return run(process)

    # Pipeline: dry-run must never process or train.
    rc = run(crawl)
    if rc:
        print(f"Crawl failed with exit code {rc}.")
        return rc
    if args.dry_run:
        print("Dry run complete; skipping processing and training.")
        return 0

    raw_pages = ROOT / "Data" / "crawl" / "raw_pages.jsonl"
    if not raw_pages.exists() or raw_pages.stat().st_size == 0:
        print("No raw pages are available; skipping processing and training.")
        print("Run --mode status to inspect queue states and recorded crawl errors.")
        return 2

    rc = run(process)
    if rc:
        print(f"Processing failed with exit code {rc}.")
        return rc
    return train_chat()


if __name__ == "__main__":
    raise SystemExit(main())
