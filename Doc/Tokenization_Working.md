# Tokenization Working Documentation

This document explains how tokenization currently works in this project (`libraries/NKS_Tokenizer`).

## Overview

The tokenizer is a **BPE-style subword tokenizer** with:
- UTF-8-aware text handling
- Pre-tokenization (whitespace, punctuation, camelCase split)
- BPE training from `Data/words.txt`
- Greedy longest-match subword encoding
- `##` continuation markers for non-initial subwords

Core files:
- `libraries/NKS_Tokenizer/NKS_Tokenizer.h`
- `libraries/NKS_Tokenizer/NKS_Tokenizer.cpp`
- `main.cpp`

## High-Level Flow

```mermaid
flowchart TD
    A[Load vocabulary file Data/words.txt] --> B[Normalize words]
    B --> C[Build word frequency table]
    C --> D[Initialize symbol corpus at char level]
    D --> E[Run BPE merge iterations]
    E --> F[Build subword vocabulary]
    F --> G[Rebuild token-to-id maps]
    G --> H[Read user input text]
    H --> I[Pre-tokenize text]
    I --> J[Split words into subwords\n(greedy longest match)]
    J --> K[Map pieces to token IDs]
    K --> L[Decode token IDs back to text]
    L --> M[Print pieces, IDs, decoded text]
```

## Detailed Pipeline

### 1. Vocabulary Loading and Training

Entry point: `loadVocabulary(vocabularyPath)`

Steps:
1. Read lines from `Data/words.txt`.
2. Normalize tokens (trim/control-char cleanup, lowercase if enabled).
3. Keep a bounded training set (`trainingWordLimit_`).
4. Train BPE using pair-merge iterations (`bpeMergeOps_`).
5. Build subword vocabulary and max token length (`maxSubwordChars_`).
6. Rebuild token-ID maps from learned subwords.

### 2. Pre-tokenization

Entry point: `preTokenize(text)`

Responsibilities:
1. Split on whitespace.
2. Handle punctuation as separate tokens (configurable).
3. Split camelCase boundaries when enabled.

Example:
- Input: `helloMyNameIsNitishSharmaAndI`
- Pre-tokenized words: `hello`, `My`, `Name`, `Is`, `Nitish`, `Sharma`, `And`, `I`

### 3. Subword Segmentation

Entry point: `subwordTokenizeWord(word)`

Algorithm:
1. Normalize word.
2. Scan left-to-right.
3. At each position, pick longest vocabulary match.
4. Prefix non-first pieces with `##`.

Example shape:
- `nitish` -> `[n, ##itis, ##h]` (depends on learned merges)

### 4. Encoding

Entry point: `encode(text)`

Steps:
1. Call `tokenize(text)` to get subword pieces.
2. Convert each piece to token ID using `tokenToId_`.
3. If piece is unknown:
   - preserve with dynamic ID if `preserveUnknownTokens_ == true`
   - otherwise map to `<unk>`.

### 5. Decoding

Entry point: `decode(tokenIds)`

Steps:
1. Convert IDs back to tokens (`idToToken_` / dynamic map).
2. Remove `##` from continuation pieces while joining.
3. Join punctuation/connector tokens with spacing rules.

## Runtime Config Knobs

Main configurable options:
- `setLowercase(bool)`
- `setSplitOnPunctuation(bool)`
- `setKeepPunctuation(bool)`
- `setSplitCamelCase(bool)`
- `setPreserveUnknownTokens(bool)`
- `setBpeMergeOps(size_t)`
- `setTrainingWordLimit(size_t)`

Current `main.cpp` defaults:
- `setBpeMergeOps(600)`
- `setTrainingWordLimit(25000)`

## Why This Solves the Previous Issue

Previous behavior treated long concatenated words as a single token.

Now:
1. camelCase pre-tokenization splits boundaries
2. subword segmentation breaks unseen words into known fragments

Result: `helloMyNameIsNitishSharmaAndI` is no longer one giant unknown token.

## SRP and GPU-Parallelization Readiness

Current design follows SRP and keeps compute boundaries clear:
1. Pre-tokenization stage
2. Subword segmentation stage
3. ID mapping stage
4. Decoding/formatting stage

This allows future parallelization strategy such as:
- Batch pre-tokenization on CPU threads
- GPU kernel for parallel longest-match over token spans
- Parallel ID lookup with device-side hash/table structures

## Limitations and Next Improvements

1. BPE training runs at startup; large merge counts can increase latency.
2. Current implementation is BPE-style and simplified (not full GPT tokenizer parity).
3. UTF-8 handling is robust for splitting, but normalization is basic (no full Unicode normalization forms).
4. Persisting trained merges/vocab to a model file would avoid retraining every run.

## Suggested Next Steps

1. Add serialization for trained subword vocab and merge rules.
2. Add benchmark command for tokens/sec and startup latency.
3. Add unit tests for:
   - camelCase boundaries
   - punctuation spacing
   - `##` continuation reconstruction
   - unknown token preservation
