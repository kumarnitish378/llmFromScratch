# LLM quality and Crawl4AI pipeline

## Existing-answer problem

The chat path may use a corpus n-gram continuation model. N-grams predict local token continuations; they do not understand the meaning of a question and can produce unrelated text. If fewer than three tokens are generated, the current code falls back to a Transformer even if its checkpoint is absent, which means random weights. The current Transformer attention path behaves like a single super-head and ignores the supplied causal mask; embedding IDs above configured vocabulary size are wrapped modulo the vocabulary. These are implementation problems, not just sampling temperature.

## Added tools

- Recursive Crawl4AI crawler with persistent SQLite queue, retries, domain/depth limits, robots.txt checks, incremental JSONL output, and page/time/disk/RAM limits.
- Streaming cleaning and exact deduplication using SQLite-backed hashes.
- PowerShell runner with explicit modes and no automatic long-running crawl at install/build time.
- URL, timestamp, crawl depth, source domain and license-status provenance in each record.

## Important limitation

The current C++ chat trainer is a corpus n-gram continuation model, not a Transformer training loop. The added pipeline collects and prepares data; it does not claim to train a neural model. Transformer backpropagation and attention correctness require dedicated code changes and tests. A successful build or crawl does not establish that model answers are fixed.

## Before crawling

1. Edit data_pipeline/crawl_config.yaml: replace example.com seed and domain with permitted sources.
2. Check source robots.txt, terms and license. Publicly readable does not automatically mean reusable for training.
3. Run a dry run and a small bounded crawl first.
4. Inspect raw and cleaned JSONL and provenance before any model-training integration.
5. Data/ is gitignored; keep large corpora local.

The pipeline mode means crawl then process only; it does not train the Transformer.
