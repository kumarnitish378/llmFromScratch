"""
=============================================================
  Hybrid Text Compression: LZ77 + Huffman Coding Pipeline
=============================================================

PIPELINE:
  Compress:   Input Text → LZ77 Tokenizer → Huffman Encoder → Bit Stream
  Decompress: Bit Stream → Huffman Decoder → LZ77 Reconstructor → Text

STAGE 1 — LZ77:
  Scans input for repeated substrings using a sliding window.
  Emits either:
    - Literal token:    ('L', char)
    - Back-reference:   ('R', offset, length)

STAGE 2 — Huffman Coding:
  Builds a frequency-optimal prefix-free binary tree.
  Encodes LZ77 tokens into a compact bitstream.
  Stores the tree in the header for decompression.
"""

import heapq
import json
import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
#  SECTION 1: LZ77 TOKENIZER
# ─────────────────────────────────────────────

@dataclass
class Token:
    """
    LZ77 output token.
      kind='L' → literal character
      kind='R' → back-reference (offset from current pos, match length)
    """
    kind: str           # 'L' or 'R'
    value: str = ''     # for 'L': the character
    offset: int = 0     # for 'R': how far back to look
    length: int = 0     # for 'R': how many chars to copy

    def __repr__(self):
        if self.kind == 'L':
            return f"Lit({repr(self.value)})"
        return f"Ref(offset={self.offset}, len={self.length})"


def lz77_compress(text: str, window_size: int = 255, max_match: int = 255) -> list[Token]:
    """
    LZ77 forward pass.
    
    Sliding window search:
      - Search window: the last `window_size` characters (already processed)
      - Lookahead buffer: next `max_match` characters (to be processed)
      
    For each position, find the longest match in the window.
    If match length >= 3, emit a back-reference (saves bytes).
    Otherwise emit a literal.
    """
    tokens = []
    i = 0
    n = len(text)

    while i < n:
        # Define search window boundaries
        win_start = max(0, i - window_size)
        window = text[win_start:i]

        best_offset = 0
        best_length = 0

        # Try every possible match start in the window
        lookahead = text[i: i + max_match]

        for j in range(len(window)):
            length = 0
            # Extend match as far as possible (can wrap into lookahead)
            while (length < len(lookahead) and
                   window[j + length % (len(window) - j)] == lookahead[length]):
                length += 1
                if j + length % (len(window) - j) >= len(window):
                    break

            if length > best_length:
                best_length = length
                best_offset = len(window) - j  # distance back from current pos

        # Only use a back-reference if it saves space (min 3 chars to be worth encoding overhead)
        if best_length >= 3:
            tokens.append(Token(kind='R', offset=best_offset, length=best_length))
            i += best_length
        else:
            tokens.append(Token(kind='L', value=text[i]))
            i += 1

    return tokens


def lz77_decompress(tokens: list[Token]) -> str:
    """
    LZ77 inverse pass.
    Reconstructs original string by replaying literal and back-reference tokens.
    """
    output = []

    for tok in tokens:
        if tok.kind == 'L':
            output.append(tok.value)
        else:
            # Back-reference: copy `length` chars starting `offset` positions back
            start = len(output) - tok.offset
            for k in range(tok.length):
                output.append(output[start + k])

    return ''.join(output)


# ─────────────────────────────────────────────
#  SECTION 2: HUFFMAN ENCODER
# ─────────────────────────────────────────────

@dataclass(order=True)
class HuffNode:
    """
    A node in the Huffman binary tree.
    freq: frequency count (used for heap ordering)
    symbol: leaf symbol (None for internal nodes)
    left/right: children
    """
    freq: int
    symbol: Optional[str] = field(default=None, compare=False)
    left: Optional['HuffNode'] = field(default=None, compare=False)
    right: Optional['HuffNode'] = field(default=None, compare=False)


def build_huffman_tree(symbols: list[str]) -> HuffNode:
    """
    Build optimal prefix-free Huffman tree from a list of symbols.
    
    Algorithm:
      1. Count symbol frequencies
      2. Push all into a min-heap
      3. Repeatedly merge the two lowest-frequency nodes
      4. Result is the tree root
    """
    freq = Counter(symbols)

    # Edge case: only one unique symbol
    if len(freq) == 1:
        sym, count = next(iter(freq.items()))
        root = HuffNode(freq=count * 2)
        root.left = HuffNode(freq=count, symbol=sym)
        return root

    heap = [HuffNode(freq=f, symbol=s) for s, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        merged = HuffNode(freq=a.freq + b.freq, left=a, right=b)
        heapq.heappush(heap, merged)

    return heap[0]


def build_codebook(node: HuffNode, prefix: str = '', codebook: dict = None) -> dict[str, str]:
    """
    Traverse Huffman tree to assign binary codes.
    Left edge = '0', right edge = '1'.
    Only leaf nodes have symbols.
    """
    if codebook is None:
        codebook = {}

    if node.symbol is not None:
        codebook[node.symbol] = prefix or '0'  # handle single-symbol edge case
        return codebook

    if node.left:
        build_codebook(node.left, prefix + '0', codebook)
    if node.right:
        build_codebook(node.right, prefix + '1', codebook)

    return codebook


def huffman_encode(symbols: list[str]) -> tuple[bytes, dict, int]:
    """
    Encode a list of symbols into a compact byte array.
    Returns:
      - encoded bytes
      - codebook (needed for decoding, stored in header)
      - number of padding bits added to last byte
    """
    tree = build_huffman_tree(symbols)
    codebook = build_codebook(tree)

    # Concatenate all binary codes
    bitstring = ''.join(codebook[s] for s in symbols)

    # Pad to a full byte boundary
    padding = (8 - len(bitstring) % 8) % 8
    bitstring += '0' * padding

    # Convert bitstring to bytes
    encoded = bytes(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))

    return encoded, codebook, padding


def huffman_decode(encoded: bytes, codebook: dict, padding: int, num_symbols: int) -> list[str]:
    """
    Decode bytes back to symbols using the stored codebook.
    Builds a reverse lookup trie for efficient decoding.
    """
    # Rebuild bitstring, strip padding from last byte
    bitstring = ''.join(f'{byte:08b}' for byte in encoded)
    if padding > 0:
        bitstring = bitstring[:-padding]

    # Reverse codebook: binary string → symbol
    reverse = {v: k for k, v in codebook.items()}

    symbols = []
    current = ''
    for bit in bitstring:
        current += bit
        if current in reverse:
            symbols.append(reverse[current])
            current = ''
            if len(symbols) == num_symbols:
                break

    return symbols


# ─────────────────────────────────────────────
#  SECTION 3: TOKEN SERIALIZATION
# ─────────────────────────────────────────────

def tokens_to_symbols(tokens: list[Token]) -> list[str]:
    """
    Flatten LZ77 tokens into a symbol stream for Huffman encoding.
    
    Encoding scheme:
      Literal:      'L' + char         → 2 symbols
      Back-ref:     'R' + offset_byte + length_byte  → 3 symbols
    
    All values treated as single-character symbols in the Huffman alphabet.
    """
    symbols = []
    for tok in tokens:
        if tok.kind == 'L':
            symbols.append('L')
            symbols.append(tok.value)
        else:
            symbols.append('R')
            symbols.append(chr(tok.offset))
            symbols.append(chr(tok.length))
    return symbols


def symbols_to_tokens(symbols: list[str]) -> list[Token]:
    """Reconstruct LZ77 tokens from flat symbol stream."""
    tokens = []
    i = 0
    while i < len(symbols):
        kind = symbols[i]
        if kind == 'L':
            tokens.append(Token(kind='L', value=symbols[i+1]))
            i += 2
        elif kind == 'R':
            offset = ord(symbols[i+1])
            length = ord(symbols[i+2])
            tokens.append(Token(kind='R', offset=offset, length=length))
            i += 3
    return tokens


# ─────────────────────────────────────────────
#  SECTION 4: TOP-LEVEL COMPRESS / DECOMPRESS
# ─────────────────────────────────────────────

def compress(text: str) -> bytes:
    """
    Full compression pipeline: text → bytes
    
    Output format (binary):
      [4 bytes] original text length
      [4 bytes] number of LZ77 tokens
      [4 bytes] number of symbols
      [4 bytes] huffman padding bits
      [N bytes] JSON-encoded codebook (length-prefixed)
      [M bytes] huffman-encoded payload
    """
    # Stage 1: LZ77
    tokens = lz77_compress(text)

    # Stage 2: Serialize tokens to symbol stream
    symbols = tokens_to_symbols(tokens)

    # Stage 3: Huffman encode
    encoded_bytes, codebook, padding = huffman_encode(symbols)

    # Serialize codebook to JSON bytes
    # Convert non-printable keys to escape sequences for JSON safety
    safe_codebook = {repr(k): v for k, v in codebook.items()}
    codebook_json = json.dumps(safe_codebook).encode('utf-8')

    # Build header
    header = struct.pack(
        '>IIII',
        len(text),          # original length
        len(tokens),        # token count
        len(symbols),       # symbol count
        padding             # huffman padding
    )

    # Length-prefix the codebook
    codebook_block = struct.pack('>I', len(codebook_json)) + codebook_json

    return header + codebook_block + encoded_bytes


def decompress(data: bytes) -> str:
    """
    Full decompression pipeline: bytes → text
    """
    offset = 0

    # Parse header
    orig_len, token_count, symbol_count, padding = struct.unpack_from('>IIII', data, offset)
    offset += 16

    # Parse codebook
    cb_len = struct.unpack_from('>I', data, offset)[0]
    offset += 4
    codebook_json = data[offset:offset + cb_len].decode('utf-8')
    offset += cb_len

    # Restore codebook (reverse repr() escaping)
    safe_codebook = json.loads(codebook_json)
    codebook = {eval(k): v for k, v in safe_codebook.items()}

    # Remaining bytes = huffman payload
    encoded_bytes = data[offset:]

    # Stage 1: Huffman decode
    symbols = huffman_decode(encoded_bytes, codebook, padding, symbol_count)

    # Stage 2: Symbols → tokens
    tokens = symbols_to_tokens(symbols)

    # Stage 3: LZ77 decompress
    text = lz77_decompress(tokens)

    return text


# ─────────────────────────────────────────────
#  SECTION 5: DEMO & ANALYSIS
# ─────────────────────────────────────────────

def compression_stats(label: str, original: str, compressed: bytes):
    orig_bytes = len(original.encode('utf-8'))
    comp_bytes = len(compressed)
    ratio = comp_bytes / orig_bytes if orig_bytes > 0 else 1
    saving = (1 - ratio) * 100

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Original size  : {orig_bytes:>8} bytes")
    print(f"  Compressed size: {comp_bytes:>8} bytes")
    print(f"  Ratio          : {ratio:>8.4f}  ({saving:+.1f}% {'saved' if saving > 0 else 'overhead'})")


def run_builtin_tests():
    """Run the original built-in test suite."""
    test_cases = {
        "Short repeated text": "abababababababababab",
        "English prose": (
            "The quick brown fox jumps over the lazy dog. "
            "The dog barked and the fox ran away quickly. "
            "A quick brown dog outpaces a lazy fox."
        ),
        "Structured/code-like": (
            "def foo(x):\n    return x * x\n"
            "def bar(x):\n    return x + x\n"
            "def baz(x):\n    return foo(x) + bar(x)\n"
        ),
        "Highly repetitive": "AAAA" * 50 + "BBBB" * 50,
        "Lorem Ipsum": (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Duis aute irure dolor in reprehenderit in voluptate velit esse."
        ),
    }

    print("=" * 55)
    print("  HYBRID LZ77 + HUFFMAN COMPRESSION RESULTS")
    print("=" * 55)

    all_passed = True
    for label, text in test_cases.items():
        compressed = compress(text)
        recovered = decompress(compressed)
        passed = recovered == text
        if not passed:
            print(f"\n  ✗ ROUNDTRIP FAILED: {label}")
            all_passed = False
        else:
            compression_stats(label, text, compressed)
            print(f"  Roundtrip        : ✓ Perfect reconstruction")

    print(f"\n{'='*55}")
    print("  ALL TESTS PASSED ✓" if all_passed else "  SOME TESTS FAILED ✗")
    print("=" * 55)


def run_file_compression(input_path: str):
    """
    Compress a file and save output as <name>_compressed.<ext>.
    Also verifies roundtrip integrity.
    """
    import os
    import time

    # ── Read input ────────────────────────────────────────
    if not os.path.exists(input_path):
        print(f"  ✗ File not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    if not text:
        print("  ✗ File is empty.")
        return

    # ── Build output filename ─────────────────────────────
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_compressed{ext}"

    # ── Compress ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  LZ77 + HUFFMAN FILE COMPRESSION")
    print(f"{'='*60}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"{'─'*60}")

    t0 = time.perf_counter()
    compressed = compress(text)
    t_compress = time.perf_counter() - t0

    # ── Write compressed binary ───────────────────────────
    with open(output_path, 'wb') as f:
        f.write(compressed)

    # ── Roundtrip verification ────────────────────────────
    t0 = time.perf_counter()
    recovered = decompress(compressed)
    t_decompress = time.perf_counter() - t0
    roundtrip_ok = (recovered == text)

    # ── Stats ─────────────────────────────────────────────
    orig_bytes  = len(text.encode('utf-8'))
    comp_bytes  = len(compressed)
    ratio       = comp_bytes / orig_bytes if orig_bytes > 0 else 1.0
    saving_pct  = (1.0 - ratio) * 100.0
    lz77_tokens = lz77_compress(text)
    ref_count   = sum(1 for t in lz77_tokens if t.kind == 'R')
    lit_count   = sum(1 for t in lz77_tokens if t.kind == 'L')

    print(f"  Original size    : {orig_bytes:>12,} bytes  ({orig_bytes/1024:.2f} KB)")
    print(f"  Compressed size  : {comp_bytes:>12,} bytes  ({comp_bytes/1024:.2f} KB)")
    print(f"  {'─'*40}")
    print(f"  Compression ratio: {ratio:>12.4f}x")
    print(f"  Space saving     : {saving_pct:>11.1f}%  {'✓ reduced' if saving_pct > 0 else '✗ overhead (file too small)'}")
    print(f"  {'─'*40}")
    print(f"  LZ77 literals    : {lit_count:>12,} tokens")
    print(f"  LZ77 back-refs   : {ref_count:>12,} tokens  (repeated patterns found)")
    print(f"  Compress time    : {t_compress*1000:>11.2f} ms")
    print(f"  Decompress time  : {t_decompress*1000:>11.2f} ms")
    print(f"  {'─'*40}")
    print(f"  Roundtrip check  : {'✓ PERFECT — byte-for-byte identical' if roundtrip_ok else '✗ MISMATCH — data corrupted!'}")
    print(f"{'='*60}\n")

    if roundtrip_ok:
        print(f"  ✓ Saved compressed file → {output_path}")
    else:
        print(f"  ✗ Roundtrip failed — compressed file NOT saved.")
        os.remove(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # No argument: run built-in tests
        print("Usage: python compression.py <filename>")
        print("       python compression.py --test\n")
        print("No file given — running built-in test suite...\n")
        run_builtin_tests()

    elif sys.argv[1] == '--test':
        run_builtin_tests()

    else:
        run_file_compression(sys.argv[1])
