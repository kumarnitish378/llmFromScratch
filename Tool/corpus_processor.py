#!/usr/bin/env python3
"""
Corpus processor for turning raw crawled files into cleaned tokenizer training datasets.

Responsibilities:
- collect .txt inputs recursively from one or more source directories
- normalize text and preserve multiline structure
- remove duplicate content using stable hashes
- filter low-signal inputs
- classify content into text / code / mixed buckets
- pack cleaned outputs into bounded shard files
- write a manifest with processing statistics
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_INPUT_DIRS = [Path("Data/crawled"), Path("Data/github_crawled")]
DEFAULT_OUTPUT_DIR = Path("Data/processed")
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "manifest.json"
DEFAULT_SHARD_SIZE_MB = 20
DEFAULT_MIN_CHARS = 200
DEFAULT_PROGRESS_EVERY = 100

CODE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
    ".py", ".pyi", ".java", ".kt", ".kts", ".scala", ".go", ".rs",
    ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".php", ".rb",
    ".swift", ".m", ".mm", ".cs", ".fs", ".lua", ".sh", ".bash",
    ".zsh", ".ps1", ".bat", ".cmd", ".sql", ".html", ".htm", ".css",
    ".scss", ".less", ".xml", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".proto", ".cmake", ".gradle",
}

CODE_KEYWORDS = {
    "#include", "public static void", "def ", "class ", "import ", "from ",
    "function ", "const ", "let ", "var ", "return ", "if (", "for (",
    "while (", "template<", "std::", "using namespace", "package ", "fn ",
    "println!", "console.log", "SELECT ", "INSERT ", "CREATE TABLE",
}

TEXT_HINTS = {
    " the ", " and ", " of ", " to ", " in ", " that ", " is ", " for ",
    " with ", " on ", " as ", " by ", " this ", " from ", " are ", " was ",
}

WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")
DIGIT_HEAVY_RE = re.compile(r"\d")
PUNCT_HEAVY_RE = re.compile(r"[{}();<>\[\]=:+\-/*%#@`~]")


@dataclasses.dataclass
class ProcessorConfig:
    input_dirs: List[Path]
    output_dir: Path
    manifest_path: Path
    shard_size_bytes: int
    min_chars: int
    progress_every: int


@dataclasses.dataclass
class CategoryStats:
    files: int = 0
    bytes: int = 0
    shards: int = 0


@dataclasses.dataclass
class ProcessorStats:
    files_scanned: int = 0
    files_processed: int = 0
    files_skipped_too_short: int = 0
    files_skipped_duplicate: int = 0
    files_skipped_empty: int = 0
    bytes_input: int = 0
    bytes_output: int = 0
    category_stats: Dict[str, CategoryStats] = dataclasses.field(
        default_factory=lambda: {
            "text": CategoryStats(),
            "code": CategoryStats(),
            "mixed": CategoryStats(),
        }
    )
    started_at: float = dataclasses.field(default_factory=time.time)

    def elapsed_seconds(self) -> float:
        return max(0.001, time.time() - self.started_at)


class ShardWriter:
    def __init__(self, category: str, output_root: Path, shard_size_bytes: int) -> None:
        self.category = category
        self.output_root = output_root / category
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.shard_size_bytes = shard_size_bytes
        self.current_size = 0
        self.current_index = 0
        self.current_path: Optional[Path] = None

    def _ensure_file(self) -> None:
        if self.current_path is not None:
            return
        self.current_index += 1
        self.current_path = self.output_root / f"{self.category}_{self.current_index:04d}.txt"
        self.current_size = 0

    def write_chunk(self, text: str) -> int:
        encoded = text.encode("utf-8", errors="replace")
        if not encoded:
            return 0

        written_total = 0
        offset = 0
        while offset < len(encoded):
            self._ensure_file()
            remaining = self.shard_size_bytes - self.current_size
            piece = encoded[offset: offset + remaining]
            while piece and (piece[-1] & 0b1100_0000) == 0b1000_0000:
                piece = piece[:-1]
            if not piece:
                break

            with self.current_path.open("ab") as handle:
                handle.write(piece)

            self.current_size += len(piece)
            written_total += len(piece)
            offset += len(piece)

            if self.current_size >= self.shard_size_bytes:
                self.current_path = None
                self.current_size = 0

        return written_total

    @property
    def shard_count(self) -> int:
        return self.current_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean, deduplicate, classify, and shard crawled corpus files.")
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help="Input directory to scan recursively for .txt files. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Processed corpus output directory.")
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH), help="Manifest JSON output path.")
    parser.add_argument("--shard-size-mb", type=int, default=DEFAULT_SHARD_SIZE_MB, help="Maximum size of each output shard in MB.")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS, help="Minimum cleaned character count to keep a file.")
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY, help="Print progress after every N scanned files.")
    return parser.parse_args()


def validate_config(args: argparse.Namespace) -> ProcessorConfig:
    input_dirs = [Path(path) for path in args.input_dir] if args.input_dir else list(DEFAULT_INPUT_DIRS)
    return ProcessorConfig(
        input_dirs=input_dirs,
        output_dir=Path(args.output_dir),
        manifest_path=Path(args.manifest_path),
        shard_size_bytes=max(1, args.shard_size_mb) * 1024 * 1024,
        min_chars=max(1, args.min_chars),
        progress_every=max(1, args.progress_every),
    )


def iter_input_files(input_dirs: Sequence[Path]) -> Iterable[Path]:
    seen = set()
    for directory in input_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.txt")):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            yield path


def normalize_text(raw_text: str) -> str:
    lines: List[str] = []
    for raw_line in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = WHITESPACE_RE.sub(" ", raw_line).strip()
        if line:
            lines.append(line)
        else:
            lines.append("")

    normalized = "\n".join(lines)
    normalized = BLANK_LINES_RE.sub("\n\n", normalized)
    return normalized.strip()


def classify_content(text: str, source_path: Path) -> str:
    suffix = source_path.suffix.lower()
    if suffix in CODE_EXTENSIONS:
        return "code"

    lowered = f" {text.lower()} "
    text_score = sum(1 for hint in TEXT_HINTS if hint in lowered)
    code_score = sum(1 for keyword in CODE_KEYWORDS if keyword.lower() in lowered)

    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        punctuation_density = len(PUNCT_HEAVY_RE.findall(text)) / max(1, len(text))
        digit_density = len(DIGIT_HEAVY_RE.findall(text)) / max(1, len(text))
    else:
        avg_line_length = 0.0
        punctuation_density = 0.0
        digit_density = 0.0

    if code_score >= max(2, text_score + 1):
        return "code"
    if text_score >= max(2, code_score + 1) and punctuation_density < 0.08:
        return "text"
    if avg_line_length < 120 and punctuation_density < 0.06 and digit_density < 0.08:
        return "text"
    return "mixed"


def build_chunk(category: str, source_path: Path, text: str) -> str:
    header = [
        "=" * 80,
        f"category: {category}",
        f"source: {source_path.as_posix()}",
        "=" * 80,
        "",
    ]
    return "\n".join(header) + text.rstrip() + "\n\n"


def print_progress(stats: ProcessorStats) -> None:
    rate = stats.files_processed / stats.elapsed_seconds()
    sys.stdout.write(
        "\r"
        f"[Corpus] scanned {stats.files_scanned} | processed {stats.files_processed}"
        f" | dup {stats.files_skipped_duplicate} | short {stats.files_skipped_too_short}"
        f" | empty {stats.files_skipped_empty} | out_bytes {stats.bytes_output}"
        f" | rate {rate:5.2f} files/s"
    )
    sys.stdout.flush()


def write_manifest(config: ProcessorConfig, stats: ProcessorStats, writers: Dict[str, ShardWriter]) -> None:
    for category, writer in writers.items():
        stats.category_stats[category].shards = writer.shard_count

    manifest = {
        "created_at": int(time.time()),
        "input_dirs": [str(path) for path in config.input_dirs],
        "output_dir": str(config.output_dir),
        "shard_size_bytes": config.shard_size_bytes,
        "min_chars": config.min_chars,
        "stats": {
            "files_scanned": stats.files_scanned,
            "files_processed": stats.files_processed,
            "files_skipped_duplicate": stats.files_skipped_duplicate,
            "files_skipped_too_short": stats.files_skipped_too_short,
            "files_skipped_empty": stats.files_skipped_empty,
            "bytes_input": stats.bytes_input,
            "bytes_output": stats.bytes_output,
            "elapsed_seconds": stats.elapsed_seconds(),
        },
        "categories": {
            category: {
                "files": cat.files,
                "bytes": cat.bytes,
                "shards": cat.shards,
            }
            for category, cat in stats.category_stats.items()
        },
    }

    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def process_corpus(config: ProcessorConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "text": ShardWriter("text", config.output_dir, config.shard_size_bytes),
        "code": ShardWriter("code", config.output_dir, config.shard_size_bytes),
        "mixed": ShardWriter("mixed", config.output_dir, config.shard_size_bytes),
    }
    stats = ProcessorStats()
    seen_hashes = set()

    for source_path in iter_input_files(config.input_dirs):
        stats.files_scanned += 1

        try:
            raw_text = source_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            stats.files_skipped_empty += 1
            continue

        stats.bytes_input += len(raw_text.encode("utf-8", errors="replace"))
        normalized = normalize_text(raw_text)
        if not normalized:
            stats.files_skipped_empty += 1
            continue
        if len(normalized) < config.min_chars:
            stats.files_skipped_too_short += 1
            continue

        content_hash = hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()
        if content_hash in seen_hashes:
            stats.files_skipped_duplicate += 1
            continue
        seen_hashes.add(content_hash)

        category = classify_content(normalized, source_path)
        chunk = build_chunk(category, source_path, normalized)
        written = writers[category].write_chunk(chunk)
        if written <= 0:
            continue

        stats.files_processed += 1
        stats.bytes_output += written
        stats.category_stats[category].files += 1
        stats.category_stats[category].bytes += written

        if config.progress_every > 0 and stats.files_scanned % config.progress_every == 0:
            print_progress(stats)

    print_progress(stats)
    sys.stdout.write("\n")
    write_manifest(config, stats, writers)
    return 0


def main() -> int:
    return process_corpus(validate_config(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
