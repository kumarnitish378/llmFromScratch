# Data Download Tools

## OpenWebText2 Downloader

Script: `Tool/download_openwebtext2.py`

Source references:
- https://openwebtext2.readthedocs.io/en/latest/

### What it does
- Downloads OpenWebText2 data using official links from the docs.
- Supports variants:
  - `clean` -> `openwebtext2.jsonl.zst.tar`
  - `raw` -> `openwebtext2_raw.tar`
- Shows progress while downloading.
- Optionally extracts downloaded tar files.

### Usage

```bash
python Tool/download_openwebtext2.py --variant clean
```

```bash
python Tool/download_openwebtext2.py --variant raw --extract
```

Optional output folder:

```bash
python Tool/download_openwebtext2.py --variant clean --output-dir Data/openwebtext2 --extract
```

## Web Crawler

Script: `Tool/webcrawler.py`

### What it does
- Crawls web pages starting from seed URLs.
- Extracts plain text from HTML and saves `.txt` files.
- Enforces:
  - max number of output files
  - max size per output file
- Packs text from multiple pages to fill each output file up to size limit.
- Preserves multiline text formatting (not a single flattened line).
- Shows a live progress bar with files/bytes/pages/queue/rate.

### Usage

```bash
python Tool/webcrawler.py \
  --seed https://example.com \
  --max-files 200 \
  --max-file-size-kb 512
```

Multiple seeds:

```bash
python Tool/webcrawler.py \
  --seed https://example.com,https://example.org \
  --seed https://example.net \
  --max-files 300 \
  --max-file-size-kb 256
```

Seed URLs from file:

```bash
python Tool/webcrawler.py --seed-file Tool/seeds.txt
```

Default behavior (no seed args):

```bash
python Tool/webcrawler.py
```
This uses `Tool/seeds.txt` automatically.

`Tool/seeds.txt` format:
```text
# one URL per line
https://example.com
https://example.org
https://example.net/page,https://another.example.net/start
```

Useful flags:
- `--output-dir Data/crawled`
- `--max-depth 2`
- `--timeout-sec 15`
- `--delay-sec 0.2`
- `--progress-every 10`
- `--parallel-mode thread`
- `--workers 8`
- `--allow-cross-domain` (off by default)

Defaults:
- `--max-files 10`
- `--max-file-size-kb 20480` (20 MB per file)

Parallel crawling:
- `thread` mode is recommended for network I/O.
- `process` mode is available if needed.

Example with explicit parallel settings:
```bash
python Tool/webcrawler.py --parallel-mode thread --workers 12
```

## GitHub Repository Crawler

Script: `Tool/github_repo_crawler.py`

### What it does
- Crawls open-source GitHub repositories using the GitHub API.
- Recursively walks repository directories up to a configured depth.
- Downloads source/code/text files and packs them into bounded `.txt` corpus files.
- Enforces:
  - max number of output files
  - max size per output file
  - max size per downloaded source file
- Shows live progress with repos, directories, candidate files, downloaded files, saved files, bytes, and rate.

### Usage

Default behavior:

```bash
python Tool/github_repo_crawler.py
```

This uses `Tool/github_repos.txt` automatically.

Explicit options:

```bash
python Tool/github_repo_crawler.py \
  --repo-file Tool/github_repos.txt \
  --output-dir Data/github_crawled \
  --max-files 10 \
  --max-file-size-kb 20480 \
  --workers 8
```

Useful flags:
- `--max-source-file-kb 2048`
- `--max-depth 32`
- `--timeout-sec 20`
- `--progress-every 25`
- `--github-token %GITHUB_TOKEN%`

Notes:
- `--github-token` is optional but strongly recommended for higher API rate limits.
- Output files are packed corpus files, not one file per repository source file.

Built-in repo groups:
- `all`
- `python`
- `ml`
- `systems`
- `web`
- `data`

Examples:

```bash
python Tool/github_repo_crawler.py --repo-group python --repo-group ml
```

```bash
python Tool/github_repo_crawler.py --allow-repo-file Tool/github_repos_python.txt
```

```bash
python Tool/github_repo_crawler.py --deny-repo-file Tool/github_repos_web.txt
```

Resume support:
- enabled by default
- manifest path defaults to `Data/github_crawled/github_repo_manifest.jsonl`
- use `--no-resume` to force a fresh crawl pass

## Corpus Processor

Script: `Tool/corpus_processor.py`

### What it does
- Reads crawled `.txt` files recursively from one or more input directories.
- Normalizes whitespace while preserving multiline structure.
- Removes duplicate content using stable content hashes.
- Filters low-signal files using a minimum character threshold.
- Classifies content into:
  - `text`
  - `code`
  - `mixed`
- Packs cleaned content into bounded shard files.
- Writes a processing manifest with counts, bytes, and shard totals.

### Default inputs
- `Data/crawled`
- `Data/github_crawled`

### Default outputs
- `Data/processed/text`
- `Data/processed/code`
- `Data/processed/mixed`
- `Data/processed/manifest.json`

### Usage

Default run:

```bash
python Tool/corpus_processor.py
```

Explicit inputs:

```bash
python Tool/corpus_processor.py \
  --input-dir Data/crawled \
  --input-dir Data/github_crawled \
  --output-dir Data/processed
```

Useful flags:
- `--shard-size-mb 20`
- `--min-chars 200`
- `--progress-every 100`

## Tokenizer Evaluator

Script: `Tool/tokenizer_evaluator.py`

### What it does
- Samples processed corpus content from:
  - `Data/processed/text`
  - `Data/processed/code`
  - `Data/processed/mixed`
- Runs the built tokenizer app in BPE mode against each sample.
- Compares configured BPE model presets:
  - `raw`
  - `processed`
- Reports:
  - average token count
  - average piece count
  - average chars per token
  - average words per token
  - decode length ratio
  - lowercase match rate
  - alnum match rate
  - vocabulary size
  - per-category summaries (`text`, `code`, `mixed`)
- Saves a JSON report.

### Usage

```bash
python Tool/tokenizer_evaluator.py
```

Useful flags:
- `--samples-per-category 20`
- `--max-sample-chars 320`
- `--model raw`
- `--model processed`
- `--output Doc/tokenizer_evaluation_report.json`
