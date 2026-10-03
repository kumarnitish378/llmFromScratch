#!/usr/bin/env python3
"""Streaming cleaning and exact deduplication backed by SQLite."""
import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean(text):
    text = CONTROL_RE.sub(" ", (text or "").replace("\r\n", "\n").replace("\r", "\n"))
    text = re.sub(r"\[\s*\d+\s*\]", " ", text)
    text = URL_RE.sub(" ", text)
    out = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if len(line) < 25 or line.lower() in {
            "home", "menu", "search", "log in", "sign up", "cookie policy"
        }:
            continue
        out.append(line)
    return "\n".join(out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "data_pipeline" / "crawl_config.yaml"))
    ap.add_argument("--input")
    ap.add_argument("--output")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    base = (ROOT / cfg.get("output_dir", "Data/crawl")).resolve()
    inp = Path(args.input).resolve() if args.input else base / "raw_pages.jsonl"
    out = Path(args.output).resolve() if args.output else base / "clean_pages.jsonl"

    if not inp.exists():
        print(f"Input file not found: {inp}")
        print("Run a real crawl first, or pass --input with an existing JSONL file.")
        return 2
    if inp.stat().st_size == 0:
        print(f"Input file is empty: {inp}")
        print("No pages to process; crawl output has not been generated yet.")
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    base.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(base / "dedup.sqlite3")
    db.execute("CREATE TABLE IF NOT EXISTS hashes(sha TEXT PRIMARY KEY)")
    db.execute("DELETE FROM hashes")
    db.commit()

    corpus_path = ROOT / "Data" / "clean_training_corpus.txt"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    kept = dupes = short = malformed = 0

    try:
        with inp.open(encoding="utf-8") as src, \
             out.open("w", encoding="utf-8") as dst, \
             corpus_path.open("w", encoding="utf-8") as corpus:
            for line in src:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue

                text = clean(row.get("markdown", ""))
                if len(text) < 200:
                    short += 1
                    continue

                digest = hashlib.sha256(
                    re.sub(r"\s+", " ", text.lower()).encode("utf-8")
                ).hexdigest()
                try:
                    db.execute("INSERT INTO hashes(sha) VALUES(?)", (digest,))
                except sqlite3.IntegrityError:
                    dupes += 1
                    continue

                record = {
                    key: row.get(key)
                    for key in (
                        "url", "title", "crawled_at", "license_status",
                        "source_domain", "depth"
                    )
                }
                record["text"] = text
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                corpus.write(text + "\n\n")
                kept += 1
                if kept % 100 == 0:
                    db.commit()
                    print(f"kept={kept} duplicates={dupes} short={short}")

        db.commit()
    finally:
        db.close()

    print(
        f"Done: kept={kept}, duplicates={dupes}, "
        f"too_short={short}, malformed={malformed}, output={out}"
    )
    print(f"Plain-text training corpus: {corpus_path}")
    if kept == 0:
        print("Warning: no usable documents survived cleaning; training corpus is empty.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
