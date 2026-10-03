# llmFromScratch

A C++17 tokenizer project with a vocabulary-backed encoding pipeline.

## Current Features
- UTF-8-aware token splitting.
- Configurable tokenization behavior:
  - lowercase normalization
  - punctuation splitting
  - punctuation retention
  - unknown-token preservation
- Vocabulary loading from `Data/words.txt`.
- `encode` and `decode` flow with reversible unknown-token handling.
- Approximate model token estimation (`~4 chars/token` heuristic).

## Project Structure
- `main.cpp`: SRP-oriented app entry and pipeline orchestration.
- `libraries/NKS_Tokenizer/`: tokenizer library implementation.
- `Data/words.txt`: vocabulary source file.
- `Makefile`: cross-platform build into `build/`.

## Build
```bash
make
```

## Optional CUDA Backend
The default build is CPU-only and works without CUDA.

To compile the optional CUDA tensor backend, install the NVIDIA CUDA Toolkit and build with:

```bash
make rebuild USE_CUDA=1
```

When enabled, `Tensor::matmul` and last-axis `Tensor::softmax` try CUDA kernels first and fall back to CPU if no CUDA device is available.

## Run
```bash
make run
```

## Clean
```bash
make clean
```

## Design Notes
- The main pipeline is split into small single-responsibility stages.
- Compute and reporting concerns are separated to keep future GPU parallelization straightforward.


## Recursive Crawl4AI data pipeline (Windows)

The bounded collector and cleaner are in `data_pipeline/`. Before crawling, edit `data_pipeline/crawl_config.yaml`: replace the example seed/domain with websites you are allowed to crawl and use. Check robots.txt, terms, and licenses first. The pipeline records provenance and does not assume public pages are licensed for model training.

Open PowerShell in the repository folder:

```powershell
# One-time Python environment setup is performed by the runner if needed.
Set-ExecutionPolicy -Scope Process Bypass
.\run_overnight.ps1 -Mode smoke-test
.\run_overnight.ps1 -Mode crawl -MaxPages 10 -MaxHours 0.25 -DryRun
# After editing the config and reviewing the dry run:
.\run_overnight.ps1 -Mode pipeline -MaxPages 100 -MaxHours 1 -MaxRamGB 6 -MaxDiskGB 5 -Resume
.\run_overnight.ps1 -Mode status
```

The pipeline mode crawls then cleans/deduplicates. **It does not train the Transformer.** The current C++ corpus chat model is an n-gram continuation baseline; a real neural training integration and Transformer gradient/attention fixes remain separate work. The chat path no longer silently falls back to randomly initialized Transformer weights when n-gram generation is too short, and it abstains on responses with no meaningful prompt-word overlap. This lexical guard is only a safety heuristic, not semantic understanding.

Use `-Mode crawl` to resume an interrupted collection, `-Mode process` to rebuild cleaned output from saved raw JSONL, and `-Mode status` to inspect the persistent queue. Start small before leaving a long job overnight.
