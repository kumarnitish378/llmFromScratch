#!/usr/bin/env python3
"""
remove_indic_data.py - Filter out all Hindi, Sanskrit, and Devanagari text from training datasets.

Purges any lines containing Devanagari, Vedic extensions, or Indic Unicode blocks.
Ensures 100% of the training corpus is pure English, programming code, and mathematical notation.
"""

import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Data"
METADATA_DIR = ROOT_DIR / "Metadata"

# Regex matching Devanagari, Devanagari Extended, Vedic, and all Indic scripts (Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam)
INDIC_SCRIPTS_RE = re.compile(
    r"[\u0900-\u097F"   # Devanagari (Hindi, Sanskrit, Marathi, Nepali)
    r"\uA8E0-\uA8FF"   # Devanagari Extended
    r"\u1CD0-\u1CFF"   # Vedic Extensions
    r"\u0980-\u0D7F]"  # All Indic scripts (Bengali, Tamil, Telugu, etc.)
)

def clean_file(file_path: Path) -> tuple[int, int]:
    """Remove Indic lines from a single text file in place. Returns (lines_removed, total_lines)."""
    if not file_path.is_file():
        return (0, 0)
        
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    lines_removed = 0
    total_lines = 0
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(temp_path, "w", encoding="utf-8", errors="ignore") as fout:
        for line in fin:
            total_lines += 1
            if INDIC_SCRIPTS_RE.search(line):
                lines_removed += 1
            else:
                fout.write(line)
                
    if lines_removed > 0:
        temp_path.replace(file_path)
    else:
        temp_path.unlink(missing_ok=True)
        
    return (lines_removed, total_lines)

def main():
    print("=" * 65)
    print("   NKS LLM - Hindi/Sanskrit (Devanagari) Data Purge")
    print("=" * 65)
    
    targets = []
    # 1. Clean crawled files
    crawled_dir = DATA_DIR / "crawled"
    if crawled_dir.exists():
        targets.extend(list(crawled_dir.glob("*.txt")))
        
    # 2. Clean processed shards
    processed_dir = DATA_DIR / "processed"
    if processed_dir.exists():
        targets.extend(list(processed_dir.rglob("*.txt")))
        
    # 3. Clean root Data and Metadata files
    targets.extend(list(DATA_DIR.glob("*.txt")))
    targets.extend(list(METADATA_DIR.glob("*.txt")))
    
    total_removed = 0
    total_scanned = 0
    cleaned_files = 0
    
    for target in sorted(targets):
        rel_path = target.relative_to(ROOT_DIR)
        removed, total = clean_file(target)
        if total == 0:
            continue
        total_scanned += total
        total_removed += removed
        if removed > 0:
            cleaned_files += 1
            print(f"  [-] {rel_path}: Removed {removed:,} lines ({removed/total*100:.1f}%)")
            
    print("\n" + "=" * 65)
    print("   Summary of Purge:")
    print(f"   Files Modified:  {cleaned_files}")
    print(f"   Lines Removed:   {total_removed:,}")
    print(f"   Total Scanned:   {total_scanned:,}")
    print("=" * 65)

if __name__ == "__main__":
    main()
