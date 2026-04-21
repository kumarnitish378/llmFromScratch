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
