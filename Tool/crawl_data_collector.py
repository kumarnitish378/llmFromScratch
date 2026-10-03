#!/usr/bin/env python3
"""
crawl_data_collector.py - Crawl4AI data scraper and corpus pipeline for NKS LLM.

Uses Crawl4AI (https://github.com/unclecode/crawl4ai.git) to crawl web pages,
extract clean markdown/text without web boilerplate, and format it for LLM training.
"""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

# Prevent Tool/compression.py from shadowing third-party/standard library compression
if sys.path and Path(sys.path[0]).name == "Tool":
    sys.path.pop(0)

from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Data"
OUTPUT_CRAWLED = DATA_DIR / "crawled_ds_corpus.txt"

# Default high-yield programming reference URLs
DEFAULT_C_URLS = [
    "https://en.wikipedia.org/wiki/Linked_list",
    "https://en.wikipedia.org/wiki/Dynamic_memory_allocation",
    "https://en.wikipedia.org/wiki/Stack_(abstract_data_type)",
    "https://en.wikipedia.org/wiki/Queue_(abstract_data_type)",
    "https://en.wikipedia.org/wiki/Binary_search_tree",
]

def clean_markdown_for_llm(raw_md: str) -> List[str]:
    """Clean markdown text, stripping citations, URLs, table formatting, and image links."""
    lines = raw_md.split("\n")
    cleaned_paragraphs = []
    
    current_para = []
    for line in lines:
        line = line.strip()
        if not line:
            if current_para:
                para_text = " ".join(current_para)
                cleaned_paragraphs.append(para_text)
                current_para = []
            continue
            
        # Skip headers, links, image tags, table borders
        if line.startswith("#") or line.startswith("!") or line.startswith("|") or line.startswith("-"):
            continue
            
        # Clean citation tags [1], [23], URLs, etc.
        line = re.sub(r"\[\s*\d+\s*\]|\(\s*\d+\s*\)", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", line) # keep link text, remove url
        line = re.sub(r"https?://\S+", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        
        if len(line) > 20:
            current_para.append(line)
            
    if current_para:
        cleaned_paragraphs.append(" ".join(current_para))
        
    # Filter for high-quality English prose paragraphs
    final_prose = []
    for para in cleaned_paragraphs:
        if len(para) < 60 or len(para) > 800:
            continue
        alpha = sum(1 for c in para if c.isalpha())
        if alpha < len(para) * 0.70:
            continue
        final_prose.append(para)
        
    return final_prose

async def crawl_urls(urls: List[str], max_pages: int = 5) -> List[str]:
    """Crawl a list of URLs using Crawl4AI."""
    try:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
    except Exception as e:
        print(f"[!] Crawl4AI import error ({e}) using python: {sys.executable}")
        return []

    results_text = []
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=20,
    )

    print(f"[*] Starting Crawl4AI crawler for {len(urls)} URLs...")
    async with AsyncWebCrawler() as crawler:
        for idx, url in enumerate(urls[:max_pages]):
            print(f"  [{idx+1}/{len(urls)}] Crawling: {url} ...")
            try:
                res = await crawler.arun(url=url, config=run_config)
                if res.success and res.markdown:
                    paragraphs = clean_markdown_for_llm(res.markdown)
                    print(f"     -> Extracted {len(paragraphs)} clean prose passages ({len(res.markdown)} chars raw markdown).")
                    results_text.extend(paragraphs)
                else:
                    print(f"     -> Crawl warning: {getattr(res, 'error_message', 'no content')}")
            except Exception as e:
                print(f"     -> Crawl error on {url}: {e}")

    return results_text

def main():
    parser = argparse.ArgumentParser(description="Crawl4AI Data Collector for NKS LLM")
    parser.add_argument("--urls", nargs="+", default=DEFAULT_C_URLS, help="List of URLs to crawl")
    parser.add_argument("--output", type=str, default=str(OUTPUT_CRAWLED), help="Output path for scraped text")
    parser.add_argument("--append-to-corpus", action="store_true", help="Directly append to clean_training_corpus.txt")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("   NKS LLM - Crawl4AI Web Data Extraction Pipeline")
    print("=" * 60)

    extracted_lines = asyncio.run(crawl_urls(args.urls))
    if not extracted_lines:
        print("[-] No content extracted. Check URL or internet access.")
        return

    with open(out_path, "w", encoding="utf-8") as f:
        for line in extracted_lines:
            f.write(line + "\n")
    print(f"\n[+] Successfully saved {len(extracted_lines)} passages to: {out_path}")

    if args.append_to_corpus:
        main_corpus = DATA_DIR / "clean_training_corpus.txt"
        with open(main_corpus, "a", encoding="utf-8") as f:
            for line in extracted_lines:
                f.write(line + "\n")
        print(f"[+] Appended to main training corpus: {main_corpus}")

if __name__ == "__main__":
    main()
