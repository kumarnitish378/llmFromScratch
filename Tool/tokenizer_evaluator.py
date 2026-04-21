#!/usr/bin/env python3
"""
Tokenizer evaluator.

Evaluates the existing tokenizer application on processed corpus samples.
Current focus is BPE raw-vs-processed comparison using the saved model presets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

APP_PATH = Path("build/app.exe")
DEFAULT_OUTPUT_PATH = Path("Doc/tokenizer_evaluation_report.json")
DEFAULT_CATEGORY_DIRS = {
    "text": Path("Data/processed/text"),
    "code": Path("Data/processed/code"),
    "mixed": Path("Data/processed/mixed"),
}
HEADER_LINE = "=" * 80
CHUNK_RE = re.compile(
    r"={80}\ncategory:\s*(?P<category>[^\n]+)\nsource:\s*(?P<source>[^\n]+)\n={80}\n(?P<body>.*?)(?=(?:\n={80}\ncategory:)|\Z)",
    re.DOTALL,
)
MODEL_PRESETS = {
    "raw": {
        "NKS_BPE_TRAINING_PATH": "Data",
        "NKS_BPE_MODEL_PATH": "Metadata/bpe_model_essay.bin",
        "NKS_MERGED_TXT_CORPUS_PATH": "Metadata/all_data_txt_corpus.txt",
    },
    "processed": {
        "NKS_BPE_TRAINING_PATH": "Data/processed",
        "NKS_BPE_MODEL_PATH": "Metadata/bpe_model_processed.bin",
        "NKS_MERGED_TXT_CORPUS_PATH": "Metadata/processed_txt_corpus.txt",
    },
}
WORD_RE = re.compile(r"\w+", re.UNICODE)
ALNUM_RE = re.compile(r"[^\w]+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


@dataclass
class Sample:
    category: str
    source: str
    text: str


@dataclass
class EvalResult:
    sample: Sample
    token_count: int
    piece_count: int
    vocabulary_size: int
    approx_token_count: int
    decoded_text: str
    lowercase_match: bool
    alnum_match: bool
    chars_per_token: float
    words_per_token: float
    decode_length_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tokenizer behavior on processed corpus samples.")
    parser.add_argument("--app-path", default=str(APP_PATH), help="Path to the built tokenizer executable.")
    parser.add_argument("--samples-per-category", type=int, default=20, help="Number of samples to evaluate per category.")
    parser.add_argument("--max-sample-chars", type=int, default=320, help="Maximum characters per sample text.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Where to write the JSON report.")
    parser.add_argument("--model", action="append", choices=sorted(MODEL_PRESETS.keys()), default=[], help="BPE model preset to evaluate. Defaults to raw and processed.")
    return parser.parse_args()


def normalize_single_line_input(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip())


def normalize_lower(text: str) -> str:
    return normalize_single_line_input(text).lower()


def normalize_alnum(text: str) -> str:
    return ALNUM_RE.sub("", normalize_single_line_input(text).lower())


def extract_chunks_from_shard(path: Path, fallback_category: str, max_chars: int) -> Iterable[Sample]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    for match in CHUNK_RE.finditer(raw):
        category = match.group("category").strip() or fallback_category
        source = match.group("source").strip() or path.as_posix()
        body = match.group("body").strip()
        if not body:
            continue
        sample_text = normalize_single_line_input(body[:max_chars])
        if sample_text:
            yield Sample(category=category, source=source, text=sample_text)


def load_samples(samples_per_category: int, max_chars: int) -> List[Sample]:
    samples: List[Sample] = []
    for category, directory in DEFAULT_CATEGORY_DIRS.items():
        count = 0
        if not directory.exists():
            continue
        for shard in sorted(directory.glob("*.txt")):
            for sample in extract_chunks_from_shard(shard, category, max_chars):
                if sample.category != category:
                    sample = Sample(category=category, source=sample.source, text=sample.text)
                samples.append(sample)
                count += 1
                if count >= samples_per_category:
                    break
            if count >= samples_per_category:
                break
    return samples


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def run_bpe_eval(app_path: Path, model_env: Dict[str, str], sample: Sample) -> EvalResult:
    env = os.environ.copy()
    env.update(model_env)
    proc = subprocess.run(
        [str(app_path)],
        input=f"bpe\n{sample.text}\n".encode("utf-8"),
        capture_output=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        stderr_text = proc.stderr.decode("utf-8", errors="replace")
        stdout_text = proc.stdout.decode("utf-8", errors="replace")
        raise RuntimeError(f"Evaluator app failed for sample from {sample.source}: {stderr_text or stdout_text}")

    output = proc.stdout.decode("utf-8", errors="replace")
    vocab_match = re.search(r"Vocabulary size:\s*(\d+)", output)
    approx_match = re.search(r"Approx model token count:\s*(\d+)", output)
    pieces_match = re.search(r"Tokenizer pieces:\s*(.*)", output)
    ids_match = re.search(r"Encoded token IDs:\s*(.*)", output)
    decoded_match = re.search(r"Decoded text:\s*(.*)", output)

    if not vocab_match or not approx_match or pieces_match is None or ids_match is None or decoded_match is None:
        raise RuntimeError(f"Failed to parse evaluator output:\n{output}")

    ids_text = ids_match.group(1).strip()
    token_ids = [part.strip() for part in ids_text.split(",") if part.strip()]
    pieces_text = pieces_match.group(1)
    piece_count = len(re.findall(r"\[[^\]]*\]", pieces_text))
    decoded_text = decoded_match.group(1).strip()
    token_count = len(token_ids)
    input_word_count = count_words(sample.text)

    return EvalResult(
        sample=sample,
        token_count=token_count,
        piece_count=piece_count,
        vocabulary_size=int(vocab_match.group(1)),
        approx_token_count=int(approx_match.group(1)),
        decoded_text=decoded_text,
        lowercase_match=normalize_lower(sample.text) == normalize_lower(decoded_text),
        alnum_match=normalize_alnum(sample.text) == normalize_alnum(decoded_text),
        chars_per_token=len(sample.text) / max(1, token_count),
        words_per_token=input_word_count / max(1, token_count),
        decode_length_ratio=len(decoded_text) / max(1, len(sample.text)),
    )


def summarize(results: Sequence[EvalResult]) -> Dict[str, object]:
    if not results:
        return {
            "samples": 0,
            "avg_token_count": 0.0,
            "avg_piece_count": 0.0,
            "avg_chars_per_token": 0.0,
            "avg_words_per_token": 0.0,
            "avg_decode_length_ratio": 0.0,
            "lowercase_match_rate": 0.0,
            "alnum_match_rate": 0.0,
            "vocabulary_size": 0,
        }

    return {
        "samples": len(results),
        "avg_token_count": sum(r.token_count for r in results) / len(results),
        "avg_piece_count": sum(r.piece_count for r in results) / len(results),
        "avg_chars_per_token": sum(r.chars_per_token for r in results) / len(results),
        "avg_words_per_token": sum(r.words_per_token for r in results) / len(results),
        "avg_decode_length_ratio": sum(r.decode_length_ratio for r in results) / len(results),
        "lowercase_match_rate": sum(1 for r in results if r.lowercase_match) / len(results),
        "alnum_match_rate": sum(1 for r in results if r.alnum_match) / len(results),
        "vocabulary_size": results[0].vocabulary_size,
    }


def build_report(model_results: Dict[str, List[EvalResult]], available_categories: Sequence[str]) -> Dict[str, object]:
    report = {"models": {}, "categories_present": list(available_categories)}
    for model_name, results in model_results.items():
        by_category: Dict[str, List[EvalResult]] = {category: [] for category in DEFAULT_CATEGORY_DIRS}
        for result in results:
            by_category.setdefault(result.sample.category, []).append(result)

        report["models"][model_name] = {
            "summary": summarize(results),
            "categories": {category: summarize(category_results) for category, category_results in by_category.items()},
            "samples": [
                {
                    "category": r.sample.category,
                    "source": r.sample.source,
                    "input": r.sample.text,
                    "decoded": r.decoded_text,
                    "token_count": r.token_count,
                    "piece_count": r.piece_count,
                    "chars_per_token": r.chars_per_token,
                    "words_per_token": r.words_per_token,
                    "decode_length_ratio": r.decode_length_ratio,
                    "lowercase_match": r.lowercase_match,
                    "alnum_match": r.alnum_match,
                }
                for r in results
            ],
        }
    return report


def print_console_summary(report: Dict[str, object], model_names: Sequence[str]) -> None:
    summary = {}
    for model_name in model_names:
        model_info = report["models"][model_name]
        summary[model_name] = {
            "summary": model_info["summary"],
            "categories": model_info["categories"],
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    app_path = Path(args.app_path)
    if not app_path.exists():
        print(f"Tokenizer app not found at {app_path}", file=sys.stderr)
        return 1

    samples = load_samples(args.samples_per_category, args.max_sample_chars)
    if not samples:
        print("No evaluation samples found under Data/processed/*", file=sys.stderr)
        return 1

    models = args.model or ["raw", "processed"]
    model_results: Dict[str, List[EvalResult]] = {}
    for model_name in models:
        env = MODEL_PRESETS[model_name]
        model_results[model_name] = [run_bpe_eval(app_path, env, sample) for sample in samples]

    available_categories = sorted({sample.category for sample in samples})
    report = build_report(model_results, available_categories)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print_console_summary(report, models)
    print(f"Saved report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
