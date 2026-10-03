#!/usr/bin/env python3
"""Bounded Crawl4AI collector with persistent SQLite queue and JSONL output."""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse, parse_qsl, urlencode, urlunparse
from urllib.robotparser import RobotFileParser

import yaml
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "data_pipeline" / "crawl_config.yaml"


def canonical_url(raw):
    raw = urldefrag((raw or "").strip())[0]
    parsed = urlparse(raw)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        return None

    query = urlencode([
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in {"fbclid", "gclid"}
    ])
    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path or "/",
        "",
        query,
        "",
    ))


def allowed(url, domains):
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def print_queue_summary(db):
    print("Queue status:")
    for status, count in db.execute(
        "SELECT status, COUNT(*) FROM urls GROUP BY status ORDER BY status"
    ):
        print(f"  {status}: {count}")

    errors = db.execute(
        "SELECT url, last_error FROM urls "
        "WHERE last_error IS NOT NULL AND last_error != '' "
        "ORDER BY updated_at DESC LIMIT 10"
    ).fetchall()
    if errors:
        print("Recent URL errors:")
        for url, error in errors:
            print(f"  {url}: {error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--max-disk-gb", type=float)
    parser.add_argument("--max-ram-gb", type=float)
    parser.add_argument("--seed-file")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        raise SystemExit(f"ERROR: config file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    output_dir = (ROOT / cfg.get("output_dir", "Data/crawl")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = {
        str(item).lower().strip().rstrip(".")
        for item in cfg.get("allowed_domains", [])
        if str(item).strip()
    }
    if not domains:
        raise SystemExit("ERROR: allowed_domains is empty. Edit crawl_config.yaml.")

    seeds = list(cfg.get("seeds", []))
    if args.seed_file:
        seed_file = Path(args.seed_file)
        if not seed_file.is_absolute():
            seed_file = ROOT / seed_file
        if not seed_file.exists():
            raise SystemExit(f"ERROR: seed file not found: {seed_file}")
        seeds += [
            line.strip()
            for line in seed_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    seeds = [canonical_url(seed) for seed in seeds]
    seeds = [seed for seed in seeds if seed and allowed(seed, domains)]
    if not seeds:
        raise SystemExit("ERROR: no valid seeds within allowed_domains.")

    if args.dry_run:
        print("Dry run only. Domains:", sorted(domains))
        print("Seeds:", *seeds, sep="\n  ")
        print("Output:", output_dir)
        print("No files were crawled or changed.")
        return 0

    max_pages = max(1, args.max_pages if args.max_pages is not None else int(cfg.get("max_pages", 1000)))
    max_hours = max(0.05, args.max_hours if args.max_hours is not None else float(cfg.get("max_hours", 8)))
    max_disk_gb = max(0.1, args.max_disk_gb if args.max_disk_gb is not None else float(cfg.get("max_disk_gb", 20)))
    max_ram_gb = max(1, args.max_ram_gb if args.max_ram_gb is not None else float(cfg.get("max_ram_gb", 6)))
    delay = max(0.0, float(cfg.get("request_delay_seconds", 2.0)))
    max_depth = max(0, int(cfg.get("max_depth", 3)))
    user_agent = str(cfg.get("user_agent", "llmFromScratchResearchBot/1.0"))

    db_path = output_dir / "queue.sqlite3"
    raw_path = output_dir / "raw_pages.jsonl"
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""
        CREATE TABLE IF NOT EXISTS urls(
            url TEXT PRIMARY KEY,
            depth INTEGER,
            status TEXT DEFAULT 'pending',
            attempts INTEGER DEFAULT 0,
            last_error TEXT,
            updated_at TEXT
        )
    """)
    db.commit()

    now = datetime.now(timezone.utc).isoformat()
    for seed in seeds:
        db.execute(
            "INSERT OR IGNORE INTO urls(url, depth, status, updated_at) "
            "VALUES(?, 0, 'pending', ?)",
            (seed, now),
        )

    # Recover interrupted jobs and retry URLs that exhausted their attempts
    # during a previous run. Successful and robots-disallowed URLs stay settled.
    db.execute(
        "UPDATE urls SET status='pending', updated_at=? WHERE status='working'",
        (now,),
    )
    retry_count = db.execute("SELECT COUNT(*) FROM urls WHERE status='failed'").fetchone()[0]
    if retry_count:
        db.execute(
            "UPDATE urls SET status='pending', attempts=0, last_error=NULL, updated_at=? "
            "WHERE status='failed'",
            (now,),
        )
        print(f"Retrying {retry_count} previously failed URL(s).")
    db.commit()

    # Create the output file even when no page succeeds; downstream code can
    # then report a clear empty-input error rather than a FileNotFoundError.
    raw_path.touch(exist_ok=True)

    robots = {}

    def robot_allowed(url):
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in robots:
            rp = RobotFileParser()
            rp.set_url(origin + "/robots.txt")
            try:
                rp.read()
            except Exception as exc:
                # If robots.txt cannot be retrieved, fail closed for this host.
                print(f"[WARN] Could not read robots.txt for {origin}: {exc}")
                rp.parse(["User-agent: *", "Disallow: /"])
            robots[origin] = rp
        return robots[origin].can_fetch(user_agent, url)

    started = time.monotonic()
    pages = 0
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=30000,
        word_count_threshold=30,
    )
    browser_config = BrowserConfig(headless=True, verbose=False)

    async def crawl_loop():
        nonlocal pages
        async with AsyncWebCrawler(config=browser_config) as crawler:
            while pages < max_pages and time.monotonic() - started < max_hours * 3600:
                raw_bytes = raw_path.stat().st_size if raw_path.exists() else 0
                if raw_bytes >= max_disk_gb * 1024**3:
                    print("Disk limit reached; stopping safely.")
                    break

                try:
                    import psutil
                    rss_bytes = psutil.Process().memory_info().rss
                    if rss_bytes >= max_ram_gb * 1024**3:
                        print("RAM limit reached; stopping safely.")
                        break
                except ImportError:
                    pass

                row = db.execute(
                    "SELECT url, depth, attempts FROM urls "
                    "WHERE status='pending' ORDER BY depth, url LIMIT 1"
                ).fetchone()
                if row is None:
                    print("No pending URLs remain.")
                    break

                url, depth, attempts = row
                db.execute(
                    "UPDATE urls SET status='working', attempts=attempts+1, updated_at=? "
                    "WHERE url=?",
                    (datetime.now(timezone.utc).isoformat(), url),
                )
                db.commit()

                try:
                    if not robot_allowed(url):
                        db.execute(
                            "UPDATE urls SET status='skipped', last_error=?, updated_at=? WHERE url=?",
                            ("robots.txt disallows URL", datetime.now(timezone.utc).isoformat(), url),
                        )
                        db.commit()
                        print("[SKIP robots.txt]", url)
                        continue

                    result = await crawler.arun(url=url, config=run_config)
                    if not getattr(result, "success", False):
                        message = str(getattr(result, "error_message", "crawl failed"))
                        raise RuntimeError(message[:300])

                    markdown_value = getattr(result, "markdown", "")
                    markdown = (
                        getattr(markdown_value, "fit_markdown", None)
                        or getattr(markdown_value, "raw_markdown", None)
                        or (markdown_value if isinstance(markdown_value, str) else "")
                    )

                    links = []
                    result_links = getattr(result, "links", {}) or {}
                    for group in ("internal", "external"):
                        for item in result_links.get(group, []) or []:
                            href = item.get("href", "") if isinstance(item, dict) else ""
                            next_url = canonical_url(urljoin(url, href))
                            if next_url and allowed(next_url, domains):
                                links.append(next_url)

                    record = {
                        "url": url,
                        "title": getattr(result, "title", "") or "",
                        "markdown": markdown or "",
                        "links": sorted(set(links)),
                        "crawled_at": datetime.now(timezone.utc).isoformat(),
                        "license_status": cfg.get("default_license_status", "unknown"),
                        "source_domain": urlparse(url).hostname,
                        "depth": depth,
                    }
                    with raw_path.open("a", encoding="utf-8") as output:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")

                    if depth < max_depth:
                        timestamp = datetime.now(timezone.utc).isoformat()
                        for next_url in record["links"]:
                            db.execute(
                                "INSERT OR IGNORE INTO urls(url, depth, status, updated_at) "
                                "VALUES(?, ?, 'pending', ?)",
                                (next_url, depth + 1, timestamp),
                            )

                    db.execute(
                        "UPDATE urls SET status='done', last_error=NULL, updated_at=? WHERE url=?",
                        (datetime.now(timezone.utc).isoformat(), url),
                    )
                    db.commit()
                    pages += 1
                    print(f"[{pages}/{max_pages}] depth={depth} chars={len(markdown or '')} {url}")

                    if delay:
                        await asyncio.sleep(delay)

                except Exception as exc:
                    # The selected row's attempts value is from before this try.
                    new_status = "pending" if attempts < 2 else "failed"
                    db.execute(
                        "UPDATE urls SET status=?, last_error=?, updated_at=? WHERE url=?",
                        (
                            new_status,
                            str(exc)[:500],
                            datetime.now(timezone.utc).isoformat(),
                            url,
                        ),
                    )
                    db.commit()
                    print("[WARN]", url, str(exc)[:250])

    try:
        asyncio.run(crawl_loop())
    except Exception as exc:
        print(f"[ERROR] Crawler stopped unexpectedly: {exc}")
        print("Check that Playwright Chromium is installed and supported by your Python version.")
        return_code = 2
    else:
        return_code = 0
    finally:
        print(f"Stopped after {pages} newly crawled page(s).")
        print(f"Raw output: {raw_path}")
        print(f"Persistent queue: {db_path}")
        print_queue_summary(db)
        db.close()

    if pages == 0:
        print(
            "No new pages were crawled. Review the queue errors above. "
            "Some hosts may disallow crawling in robots.txt."
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
