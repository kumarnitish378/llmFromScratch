#!/usr/bin/env python3
"""
Simple web crawler for building text corpora.

Key controls:
- max number of output files
- max size per output file
- crawl depth
- same-domain restriction
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import html
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Deque, Iterable, List, Set, Tuple


HREF_RE = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
BLOCK_BREAK_RE = re.compile(
    r"(?i)(<br\s*/?>|</(p|div|li|h1|h2|h3|h4|h5|h6|section|article|ul|ol|pre|blockquote|tr|td|th)>)"
)
TAG_RE = re.compile(r"<[^>]+>")
LINE_SPACE_RE = re.compile(r"[ \t\f\v]+")


@dataclasses.dataclass
class CrawlConfig:
    output_dir: Path
    max_files: int
    max_file_size_bytes: int
    max_depth: int
    timeout_sec: int
    delay_sec: float
    min_text_chars: int
    same_domain_only: bool
    user_agent: str
    progress_every: int
    parallel_mode: str
    workers: int


def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def same_domain(url_a: str, url_b: str) -> bool:
    return urllib.parse.urlsplit(url_a).netloc.lower() == urllib.parse.urlsplit(url_b).netloc.lower()


def extract_links(base_url: str, html_content: str) -> List[str]:
    links: List[str] = []
    for raw in HREF_RE.findall(html_content):
        joined = urllib.parse.urljoin(base_url, raw)
        parsed = urllib.parse.urlsplit(joined)
        if parsed.scheme not in ("http", "https"):
            continue
        clean = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
        links.append(normalize_url(clean))
    return links


def html_to_text(html_content: str) -> str:
    without_scripts = SCRIPT_STYLE_RE.sub(" ", html_content)
    with_breaks = BLOCK_BREAK_RE.sub("\n", without_scripts)
    no_tags = TAG_RE.sub(" ", with_breaks)
    unescaped = html.unescape(no_tags)
    lines = []
    for raw in unescaped.splitlines():
        line = LINE_SPACE_RE.sub(" ", raw).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def fetch_html(url: str, timeout_sec: int, user_agent: str) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" not in ctype:
                return None
            data = resp.read()
            charset = "utf-8"
            if "charset=" in ctype:
                charset = ctype.split("charset=", 1)[1].split(";", 1)[0].strip()
            return data.decode(charset, errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def crawl_page_task(task: Tuple[str, int, str, str]) -> Tuple[str, int, str, str | None, List[str]]:
    url, timeout_sec, min_text_chars, user_agent = task
    html_doc = fetch_html(url, timeout_sec, user_agent)
    if html_doc is None:
        return (url, timeout_sec, user_agent, None, [])

    text = html_to_text(html_doc)
    if len(text) < min_text_chars:
        return (url, timeout_sec, user_agent, None, [])

    links = extract_links(url, html_doc)
    return (url, timeout_sec, user_agent, text, links)


def save_text(text: str, path: Path, max_bytes: int) -> int:
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        encoded = encoded[:max_bytes]
        while encoded and (encoded[-1] & 0b1100_0000) == 0b1000_0000:
            encoded = encoded[:-1]
    path.write_bytes(encoded)
    return len(encoded)


def append_text_to_file(text: str, path: Path, max_bytes: int, used_bytes: int) -> Tuple[int, int]:
    encoded = text.encode("utf-8")
    if used_bytes >= max_bytes:
        return (0, 0)

    remaining = max_bytes - used_bytes
    chunk = encoded[:remaining]
    while chunk and (chunk[-1] & 0b1100_0000) == 0b1000_0000:
        chunk = chunk[:-1]

    with path.open("ab") as f:
        f.write(chunk)
    return (len(chunk), len(encoded) - len(chunk))


def format_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"


def render_progress(
    files_written: int,
    max_files: int,
    bytes_written: int,
    max_file_size_bytes: int,
    attempted: int,
    visited: int,
    queued: int,
    start_ts: float,
) -> str:
    total_target = max_files * max_file_size_bytes
    ratio = 0.0 if total_target <= 0 else min(1.0, bytes_written / float(total_target))
    bar_width = 28
    filled = int(ratio * bar_width)
    bar = "#" * filled + "-" * (bar_width - filled)
    elapsed = max(0.001, time.time() - start_ts)
    rate = attempted / elapsed
    return (
        f"\r[progress] |{bar}| {ratio * 100:5.1f}% "
        f"files={files_written}/{max_files} "
        f"bytes={format_bytes(bytes_written)}/{format_bytes(total_target)} "
        f"pages={attempted} visited={visited} queue={queued} rate={rate:.2f}/s"
    )


def parse_seed_list(seed_args: Iterable[str]) -> List[str]:
    seeds: List[str] = []
    for item in seed_args:
        for s in item.split(","):
            s = s.strip()
            if s:
                seeds.append(normalize_url(s))
    return seeds


def parse_seed_file(seed_file_path: str) -> List[str]:
    seeds: List[str] = []
    path = Path(seed_file_path)
    if not path.exists():
        return seeds

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            for part in line.split(","):
                part = part.strip()
                if not part:
                    continue
                seeds.append(normalize_url(part))
    return seeds


def crawl(seeds: List[str], cfg: CrawlConfig) -> Tuple[int, int]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    queue: Deque[Tuple[str, int, str]] = collections.deque()
    for s in seeds:
        queue.append((s, 0, s))

    visited: Set[str] = set()
    files_written = 0
    bytes_written = 0
    pages_attempted = 0
    start_ts = time.time()
    active_file_path: Path | None = None
    active_file_bytes = 0

    def ensure_active_file() -> bool:
        nonlocal files_written, active_file_path, active_file_bytes
        if active_file_path is not None and active_file_bytes < cfg.max_file_size_bytes:
            return True
        if files_written >= cfg.max_files:
            return False
        active_file_path = cfg.output_dir / f"page_{files_written:06d}.txt"
        active_file_path.write_text("", encoding="utf-8")
        active_file_bytes = 0
        files_written += 1
        return True

    def write_page_text(url: str, text: str) -> None:
        nonlocal bytes_written, active_file_bytes, active_file_path
        payload = text + f"\n\n[source] {url}\n\n"
        remaining_text = payload
        while remaining_text:
            if not ensure_active_file():
                return
            assert active_file_path is not None
            written, pending = append_text_to_file(
                remaining_text,
                active_file_path,
                cfg.max_file_size_bytes,
                active_file_bytes,
            )
            if written <= 0:
                active_file_path = None
                active_file_bytes = 0
                continue
            active_file_bytes += written
            bytes_written += written
            if pending > 0:
                kept = len(remaining_text.encode("utf-8")) - pending
                remaining_text = remaining_text.encode("utf-8")[kept:].decode("utf-8", errors="ignore")
                active_file_path = None
                active_file_bytes = 0
            else:
                remaining_text = ""

    def print_progress(force: bool = False) -> None:
        if cfg.progress_every <= 0:
            return
        if not force and pages_attempted % cfg.progress_every != 0:
            return
        print(
            render_progress(
                files_written=files_written,
                max_files=cfg.max_files,
                bytes_written=bytes_written,
                max_file_size_bytes=cfg.max_file_size_bytes,
                attempted=pages_attempted,
                visited=len(visited),
                queued=len(queue),
                start_ts=start_ts,
            ),
            end="",
            flush=True,
        )

    while queue and files_written < cfg.max_files:
        current_depth = queue[0][1]
        level_items: List[Tuple[str, int, str]] = []
        while queue and queue[0][1] == current_depth:
            level_items.append(queue.popleft())

        pending: List[Tuple[str, str]] = []
        for url, depth, seed_root in level_items:
            if url in visited:
                continue
            visited.add(url)
            pending.append((url, seed_root))

        if not pending:
            continue

        next_level: List[Tuple[str, int, str]] = []

        if cfg.parallel_mode == "none" or cfg.workers <= 1:
            for url, seed_root in pending:
                pages_attempted += 1
                html_doc = fetch_html(url, cfg.timeout_sec, cfg.user_agent)
                print_progress()
                if html_doc is None:
                    continue

                text = html_to_text(html_doc)
                if len(text) >= cfg.min_text_chars and files_written <= cfg.max_files:
                    write_page_text(url, text)

                if current_depth < cfg.max_depth:
                    for link in extract_links(url, html_doc):
                        if cfg.same_domain_only and not same_domain(seed_root, link):
                            continue
                        if link not in visited:
                            next_level.append((link, current_depth + 1, seed_root))

                if cfg.delay_sec > 0:
                    time.sleep(cfg.delay_sec)
        else:
            executor_cls = (
                concurrent.futures.ThreadPoolExecutor
                if cfg.parallel_mode == "thread"
                else concurrent.futures.ProcessPoolExecutor
            )
            with executor_cls(max_workers=cfg.workers) as ex:
                future_to_seed: dict[concurrent.futures.Future, str] = {}
                for url, seed_root in pending:
                    task = (url, cfg.timeout_sec, cfg.min_text_chars, cfg.user_agent)
                    fut = ex.submit(crawl_page_task, task)
                    future_to_seed[fut] = seed_root

                for fut in concurrent.futures.as_completed(future_to_seed):
                    pages_attempted += 1
                    print_progress()
                    seed_root = future_to_seed[fut]
                    try:
                        url, _timeout, _ua, text, links = fut.result()
                    except Exception:
                        continue

                    if text is not None and files_written <= cfg.max_files:
                        write_page_text(url, text)

                    if current_depth < cfg.max_depth:
                        for link in links:
                            if cfg.same_domain_only and not same_domain(seed_root, link):
                                continue
                            if link not in visited:
                                next_level.append((link, current_depth + 1, seed_root))

            if cfg.delay_sec > 0:
                time.sleep(cfg.delay_sec)

        if files_written >= cfg.max_files and active_file_path is None:
            break

        # Deduplicate next frontier while preserving order.
        seen_next: Set[str] = set()
        for url, depth, seed_root in next_level:
            if url in seen_next:
                continue
            seen_next.add(url)
            queue.append((url, depth, seed_root))

    print_progress(force=True)
    print()
    return files_written, bytes_written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Crawl webpages and save cleaned text files.")
    p.add_argument("--seed", action="append", help="Seed URL. Repeatable, supports comma-separated list.")
    p.add_argument(
        "--seed-file",
        default="Tool/seeds.txt",
        help="Path to text file containing seed URLs (one per line, '#' for comments).",
    )
    p.add_argument("--output-dir", default="Data/crawled", help="Directory for output .txt files.")
    p.add_argument("--max-files", type=int, default=10, help="Maximum number of output files to save.")
    p.add_argument("--max-file-size-kb", type=int, default=20 * 1024, help="Maximum size per saved file in KB.")
    p.add_argument("--max-depth", type=int, default=50, help="Maximum crawl depth from each seed.")
    p.add_argument("--timeout-sec", type=int, default=15, help="HTTP timeout in seconds.")
    p.add_argument("--delay-sec", type=float, default=0.0, help="Delay between page fetches.")
    p.add_argument("--min-text-chars", type=int, default=200, help="Skip pages with less extracted text.")
    p.add_argument("--allow-cross-domain", action="store_true", help="Allow crawling outside seed domains.")
    p.add_argument("--user-agent", default="LLMFromScratchCrawler/1.0", help="User-Agent header.")
    p.add_argument("--progress-every", type=int, default=10, help="Print progress every N attempted pages.")
    p.add_argument("--parallel-mode", choices=["none", "thread", "process"], default="thread", help="Concurrency mode.")
    p.add_argument("--workers", type=int, default=8, help="Number of workers for parallel mode.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.max_files <= 0:
        raise SystemExit("--max-files must be > 0")
    if args.max_file_size_kb <= 0:
        raise SystemExit("--max-file-size-kb must be > 0")
    if args.max_depth < 0:
        raise SystemExit("--max-depth must be >= 0")
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be > 0")
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")

    seeds: List[str] = []
    if args.seed:
        seeds.extend(parse_seed_list(args.seed))
    if args.seed_file:
        seeds.extend(parse_seed_file(args.seed_file))

    seeds = list(dict.fromkeys(seeds))
    if not seeds:
        raise SystemExit("No valid seeds provided. Use --seed and/or --seed-file.")

    cfg = CrawlConfig(
        output_dir=Path(args.output_dir),
        max_files=args.max_files,
        max_file_size_bytes=args.max_file_size_kb * 1024,
        max_depth=args.max_depth,
        timeout_sec=args.timeout_sec,
        delay_sec=args.delay_sec,
        min_text_chars=args.min_text_chars,
        same_domain_only=not args.allow_cross_domain,
        user_agent=args.user_agent,
        progress_every=args.progress_every,
        parallel_mode=args.parallel_mode,
        workers=args.workers,
    )

    files_written, bytes_written = crawl(seeds, cfg)
    print(f"[done] files={files_written}, bytes={bytes_written}, output={cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
