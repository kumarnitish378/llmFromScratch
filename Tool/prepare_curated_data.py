#!/usr/bin/env python3
"""
prepare_curated_data.py - Clean, prepare, and augment training data for NKS LLM.

Closed-loop pipeline:
1. Generates a rich conversational QA corpus with distinct 3-token suffixes and direct continuations.
2. Cleans and chunks the high-quality essay corpus (Data/Training_Essay_Data.txt).
3. Strictly filters web crawl data to prevent citation tags [192], pmid, and bibliographies.
4. Produces a curated high-signal text corpus: Data/clean_training_corpus.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Data"
OUTPUT_CORPUS = DATA_DIR / "clean_training_corpus.txt"

# Regex for stripping citations, urls, and bracketed numbers
BRACKET_NUM_RE = re.compile(r"\[\s*\d+\s*\]|\(\s*\d+\s*\)")
BIBLIO_RE = re.compile(r"\b(pmid|doi|bibcode|isbn|s2cid|retrieved|arxiv|issn)\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
GARBAGE_CHARS_RE = re.compile(r"[â€œâ€\x00-\x08\x0b\x0c\x0e-\x1f]")
WHITESPACE_RE = re.compile(r"[ \t]+")
CODE_LINE_RE = re.compile(r"^\s*(#include|import |from |def |class |public |private |protected |return |var |let |const |function |for\s*\(|while\s*\(|if\s*\(|template<|std::|\{|\}|;|\/\/|\/\*|\*\/)", re.IGNORECASE)

BASE_QA_DATA: List[Tuple[str, str]] = [
    # Identity & Greetings (every query has a unique 3-token suffix)
    ("how are you", "I am doing well, thank you for asking, and I am ready to help you with your coding, analysis, and language model tasks."),
    ("how is everything going", "I am doing great and functioning at full capacity. What would you like to work on today?"),
    ("hello there", "Hello there! It is a pleasure to meet you. How can I assist you with your questions today?"),
    ("what is your name", "My name is NKS LLM, an artificial intelligence language model and tokenizer system built from scratch in C++."),
    ("who are you", "I am NKS LLM, an open-source neural language model and tokenizer framework created from first principles in C++."),
    ("who created you", "I was created from scratch in modern C++ with custom tokenizers, neural tensor layers, and optional CUDA GPU acceleration."),
    ("what can you do", "I can generate coherent text, answer questions, encode and decode text with subword tokenizers, compress files using LZ77 and Huffman coding, and run transformer inference."),

    # Language Models & Transformers
    ("tell me about language models", "Language models are computational systems trained to understand, predict, and generate natural language. Modern language models use neural network architectures, such as transformers with multi-head self-attention mechanisms, to compute probability distributions over vocabulary tokens and predict subsequent words given prior context."),
    ("what is a language model", "A language model is a probabilistic mathematical model that determines the likelihood of word sequences. Given preceding context, it computes probabilities for subsequent tokens to generate fluent text."),
    ("how do language models work", "Language models tokenize text into discrete subwords, project tokens into vector embeddings, pass embeddings through attention and feed-forward layers, and calculate logits over vocabulary tokens using softmax."),
    ("what is tokenization", "Tokenization is the process of segmenting text into atomic units called tokens. These can be full words, subword pieces, or individual characters, which are then mapped to numerical IDs for neural network processing."),
    ("what is byte pair encoding", "BPE operates by iteratively merging frequent adjacent subword character pairs into unified vocabulary entries."),
    ("what is bpe", "BPE is an iterative subword tokenization algorithm that builds a vocabulary by merging the most frequent consecutive byte pairs."),
    ("what is sentencepiece", "SentencePiece is an unsupervised subword tokenizer that processes raw text directly using a whitespace marker, avoiding language-dependent pre-tokenizers while supporting reversible decoding."),
    ("what is self attention", "Self-attention is a neural mechanism that allows an input sequence to interact with itself, calculating attention weights between all token pairs so each position can dynamically gather context from the entire sequence."),
    ("what is a transformer", "Transformers are deep neural networks that process sequential tokens in parallel using multi-head self-attention and feed-forward layers."),
    ("what is positional encoding", "Positional encoding adds spatial and order information to input token embeddings, enabling permutation-invariant attention mechanisms to distinguish the positions of words in a sequence."),
    ("what is multi head attention", "Multi-head attention runs the self-attention mechanism multiple times in parallel across different representation subspaces, allowing the model to jointly attend to information from different representation perspectives."),

    # Computing, CUDA & C++
    ("what is cuda", "CUDA is NVIDIA's parallel computing platform and programming model that enables graphics processing units to accelerate compute-intensive operations, such as matrix multiplications and neural network inference."),
    ("what is c++", "C++ is a high-performance compiled programming language created by Bjarne Stroustrup that provides zero-overhead abstractions, efficient memory management, and strong type safety."),
    ("what is deep learning", "Deep learning is a subfield of machine learning based on artificial neural networks with multiple layers that learn hierarchical feature representations directly from data."),
    ("what is machine learning", "Machine learning is the study of algorithms that improve automatically through experience and data, learning statistical patterns to perform classification, regression, and generation tasks."),
    ("what is an optimizer", "An optimizer adjusts neural network weights during training to minimize the objective loss function, with common methods including Adam and SGD."),
    ("what is adam optimizer", "Adam stands for Adaptive Moment Estimation. It calculates adaptive learning rates for each parameter by maintaining moving averages of both past gradients and squared gradients."),

    # Short Answers & Explanations
    ("write a short answer", "Language models analyze statistical patterns in text to predict and generate contextually appropriate responses."),
    ("give me a brief explanation", "Machine learning algorithms optimize numerical weights on training data to minimize prediction error."),
    ("explain neural networks", "Artificial neural networks are collections of connected units called neurons that transform inputs through weighted connections and non-linear activation functions."),
    ("what is cross entropy loss", "Cross-entropy loss measures the discrepancy between predicted probability distributions and target ground-truth classes during neural network training."),
    ("what is perplexity", "Perplexity is a standard evaluation metric for language models that measures how well a probability model predicts a sample. Lower perplexity corresponds to better predictive performance."),
    ("why is gpu faster than cpu for deep learning", "GPUs feature thousands of smaller, highly efficient cores designed to perform massive parallel matrix multiplications simultaneously, which are the fundamental operations in deep learning."),
    ("what is backpropagation", "Backpropagation is an efficient algorithm for calculating gradients of a loss function with respect to neural network weights using the chain rule of calculus."),
    ("what is gradient descent", "Gradient descent is an optimization algorithm that iteratively adjusts parameters in the opposite direction of the gradient to reach a local minimum."),
    ("what is overfitting", "Overfitting occurs when a machine learning model memorizes training noise rather than general patterns, resulting in poor performance on unseen test data."),
    ("what is regularization", "Regularization comprises techniques such as weight decay and dropout that constrain model complexity to prevent overfitting and improve generalization."),

    # Compression & Information Theory
    ("what is data compression", "Data compression is the technique of reducing the number of bits required to represent information by eliminating redundancy and encoding frequent symbols with fewer bits."),
    ("what is huffman coding", "Huffman coding is an optimal prefix-free entropy encoding algorithm that assigns variable-length binary codes to input characters based on their frequencies of occurrence."),
    ("what is lz77 compression", "LZ77 is a lossless dictionary-based compression algorithm that replaces repeated occurrences of data with references to earlier matching substrings using length-distance pairs."),
    ("what is shannon entropy", "Shannon entropy quantifies the expected amount of information or uncertainty contained in a message from a random source."),

    # Advanced Transformers & LLM Inference
    ("what is temperature in text generation", "Temperature controls the sharpness of the probability distribution during sampling, where lower values make outputs more deterministic and higher values increase diversity."),
    ("what is top k sampling", "Top-k sampling restricts token candidate selection to the k most probable tokens, redistributing probability mass among them to prevent sampling low-likelihood text."),
    ("what is fine tuning", "Fine-tuning is the process of taking a pretrained foundation model and updating its parameters on domain-specific data to adapt it for specialized tasks."),
    ("what is prompt engineering", "Prompt engineering is the practice of structuring and phrasing inputs effectively to guide generative language models towards accurate and useful responses."),
    ("what is context window", "The context window is the maximum number of consecutive tokens that a language model can process and attend to simultaneously in a single forward pass."),
    ("what is kv cache", "The KV cache stores previously computed key and value projection tensors across transformer layers to avoid redundant recalculation during autoregressive token generation."),
    ("what is flash attention", "FlashAttention is an exact fast attention algorithm that optimizes GPU memory hierarchy access and reduces memory traffic between high-bandwidth memory and SRAM."),
    ("what is rotary position embedding", "Rotary position embedding, or RoPE, encodes positional information by rotating query and key representations in the complex plane, naturally incorporating relative distances."),

    # Systems, C++ & GPU Acceleration
    ("what is raii in c++", "RAII, or Resource Acquisition Is Initialization, is a C++ programming idiom where resources like heap memory, files, and mutexes are bound to object lifetimes for automatic cleanup."),
    ("what is a pointer", "A pointer is a variable that stores the memory address of another variable or object in memory."),
    ("what is shared memory in cuda", "Shared memory is a fast on-chip programmable cache shared among all threads in a CUDA thread block that enables high-throughput data reuse."),
    ("what is a cuda warp", "A CUDA warp is a basic execution unit consisting of 32 threads that execute instructions simultaneously in SIMT fashion on NVIDIA GPUs."),

    # Data Structures & C Programming
    ("explain link list in c programming", "A linked list in C is a linear dynamic data structure where elements called nodes are connected via pointers rather than stored contiguously. Each node has a data field and a next pointer defined as struct Node { int data; struct Node* next; }. Nodes are dynamically allocated on the heap using malloc and traversed sequentially starting from the head pointer."),
    ("can you explain link list in c programming", "In C programming, a linked list is a dynamic sequence of nodes connected by pointers. Each node contains data and a pointer to the next node in the list. Dynamic memory allocation with malloc allows linked lists to grow or shrink without pre-allocating a fixed contiguous array. Inserting or deleting at the head requires O(1) constant time, while finding an element takes O(n) linear search time."),
    ("what is a linked list", "A linked list is a linear collection of data elements whose order is not given by their physical placement in memory, but rather each element points to the next using a pointer reference."),
    ("how do you implement a linked list in c", "To implement a linked list in C, define a node struct with data and a pointer to struct Node, allocate nodes dynamically using malloc, set the next pointer of the last node to NULL, and free allocated memory after use."),
    ("what is a singly linked list", "A singly linked list is a unidirectionally linked data structure where each node points only to the subsequent node, terminating with a NULL pointer at the end of the list."),
    ("what is a doubly linked list", "A doubly linked list is a bidirectional data structure where each node stores two pointers: one pointing forward to the next node and one pointing backward to the previous node."),
    ("what is the difference between an array and a linked list", "Arrays store elements in contiguous memory allowing O(1) random index access but have fixed capacity, whereas linked lists allocate nodes anywhere in heap memory and connect them with pointers, allowing flexible resizing and O(1) head insertion but requiring O(n) sequential access."),
    ("how does malloc work in c", "The malloc function in C allocates a requested number of contiguous bytes on the heap at runtime and returns a void pointer to the beginning of the memory block, which must be cast and later released with free."),
    ("what is a stack data structure", "A stack is a linear data structure following the Last In First Out LIFO principle, where elements are added and removed exclusively from one end called the top."),
    ("what is a queue data structure", "A queue is a linear data structure following the First In First Out FIFO principle, where elements are inserted at the rear and removed from the front."),
    ("what is a binary search tree", "A binary search tree is a node-based binary tree data structure where each node has a key, and values in the left subtree are smaller than the node key, while values in the right subtree are greater."),
    ("what is dynamic memory allocation in c", "Dynamic memory allocation in C enables programs to obtain heap memory at runtime using functions like malloc, calloc, realloc, and release it using free."),

    # Arrays, Trees, Pointers & Algorithms
    ("how array works in c programming", "In C programming, an array is a collection of elements of the same data type stored in contiguous memory locations. When declared as int arr[5], the compiler reserves contiguous bytes. Elements are accessed using zero-based indexing arr[i], computed directly in O(1) constant time with pointer arithmetic: address = base_address + i * sizeof(type). Arrays offer fast CPU cache locality and instant random access, but have a fixed size determined at declaration."),
    ("how arrays work in c", "Arrays in C store homogeneous elements in contiguous blocks of memory. The array variable acts as a constant pointer to its first element arr[0]. Accessing any element via arr[index] takes O(1) time through address calculation. Because memory is contiguous, arrays provide superior CPU cache performance compared to linked structures."),
    ("how array works", "An array stores a sequential, fixed-size collection of elements of the same type in contiguous memory. Each element is directly accessible via a numerical index in O(1) time through pointer offset arithmetic."),
    ("what is an array in c", "An array in C is a contiguous block of memory allocated to hold a fixed number of items of identical data type, accessed using zero-based integer indices."),
    ("what is an array in c programming", "An array in C programming is a linear data structure that stores homogeneous data elements in contiguous memory locations, enabling constant-time O(1) random access."),
    ("explain array in c programming", "An array in C is defined with a type, name, and size, reserving continuous memory on the stack or heap. Because elements are adjacent, array operations benefit from spatial cache locality, though the array cannot be dynamically resized once declared."),
    ("how binary tree works", "A binary tree is a hierarchical data structure composed of nodes, where each node has at most two children called the left child and right child. The topmost node is the root. In C, each node is defined as struct TreeNode { int val; struct TreeNode* left; struct TreeNode* right; }. Operations like traversal are implemented with recursion via inorder, preorder, or postorder walks, and search in a balanced binary search tree runs in O(log n) time."),
    ("how does a binary tree work", "A binary tree organizes items hierarchically with a root node branching to at most two child nodes per parent. In a binary search tree, items smaller than the parent go left, and larger items go right, enabling O(log n) average search and insertion time."),
    ("what is a binary tree", "A binary tree is a tree data structure in which each parent node has no more than two child nodes, termed left and right. It forms the foundation for binary search trees, heaps, and expression trees."),
    ("how link list works in c programming", "A linked list in C works by chaining independent heap-allocated structures called nodes using pointers. Each node holds a value and a pointer to the next node. Because nodes are linked rather than contiguous, inserting or removing elements at the head takes O(1) time without shifting other elements."),
    ("how pointers work in c", "A pointer in C stores the memory address of another variable. You use the address-of operator & to retrieve a variable address and the dereference operator * to access or modify the underlying value."),
]

def clean_line(line: str) -> str:
    """Clean a text line by removing citations, urls, and normalizing whitespace."""
    line = BRACKET_NUM_RE.sub(" ", line)
    line = BIBLIO_RE.sub(" ", line)
    line = URL_RE.sub(" ", line)
    line = GARBAGE_CHARS_RE.sub(" ", line)
    line = WHITESPACE_RE.sub(" ", line).strip()
    return line

def is_clean_prose(line: str) -> bool:
    """Check if line is high-quality natural language text."""
    if len(line) < 35 or len(line) > 600:
        return False
    if CODE_LINE_RE.match(line):
        return False
    if BIBLIO_RE.search(line):
        return False
    alpha = sum(1 for c in line if c.isalpha())
    if alpha < len(line) * 0.70:
        return False
    punct_count = sum(1 for c in line if c in "{}[]()<>=+*/\\_#|")
    if punct_count > 3:
        return False
    return True

def generate_conversations() -> List[str]:
    """Generate extensive multi-format QA lines with direct query-response transitions."""
    lines: List[str] = []
    for query, response in BASE_QA_DATA:
        # Heavily weight direct unpunctuated and punctuated prompt-to-response transitions
        for _ in range(8):
            lines.append(f"{query} {response}")
        for _ in range(4):
            lines.append(f"{query}? {response}")
        for _ in range(2):
            lines.append(f"{query}. {response}")
        lines.append(f"Hello, {query}? {response}")
        lines.append(f"Please {query}. {response}")
        lines.append(f"Can you {query}? {response}")
        lines.append(f"Could you {query}? {response}")
        lines.append(f"Tell me {query}. {response}")
        lines.append(f"Explain {query}. {response}")
    return lines

def process_essays(essay_path: Path, max_lines: int = 50000) -> List[str]:
    """Extract clean multi-sentence passages from essays."""
    cleaned: List[str] = []
    if not essay_path.exists():
        print(f"  [!] Essay file not found: {essay_path}")
        return cleaned

    print(f"  [+] Reading essay data from {essay_path}...")
    with open(essay_path, "r", encoding="utf-8", errors="ignore") as f:
        count = 0
        for line in f:
            line = clean_line(line)
            if not is_clean_prose(line):
                continue
            # Break paragraph into clean sentence chunks
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
            chunk = []
            chunk_len = 0
            for sent in sentences:
                chunk.append(sent)
                chunk_len += len(sent)
                if chunk_len >= 140:
                    cleaned.append(" ".join(chunk))
                    chunk = []
                    chunk_len = 0
                    count += 1
                    if count >= max_lines:
                        break
            if chunk and count < max_lines:
                cleaned.append(" ".join(chunk))
                count += 1
            if count >= max_lines:
                break

    print(f"  [+] Extracted {len(cleaned)} clean essay chunks.")
    return cleaned

def main():
    print("=" * 60)
    print("   NKS LLM - Curated Corpus Generator")
    print("=" * 60)

    # 1. Generate Conversational QA
    print("[1] Synthesizing conversational and technical QA...")
    qa_lines = generate_conversations()
    print(f"  [+] Generated {len(qa_lines)} base QA variations.")

    # 2. Extract Essays
    print("\n[2] Processing essay corpus...")
    essay_file = DATA_DIR / "Training_Essay_Data.txt"
    essay_lines = process_essays(essay_file, max_lines=15000)

    # 3. Ingest Crawled Web Data from Crawl4AI
    print("\n[3] Ingesting Crawl4AI web crawled data...")
    crawled_lines: List[str] = []
    for crawl_file in DATA_DIR.glob("crawled_*.txt"):
        print(f"  [+] Loading crawled web data from {crawl_file.name}...")
        with open(crawl_file, "r", encoding="utf-8", errors="ignore") as cf:
            for cl in cf:
                cl = clean_line(cl)
                if len(cl) > 30:
                    crawled_lines.append(cl)
    print(f"  [+] Ingested {len(crawled_lines)} crawled tutorial/documentation lines.")

    # 4. Assemble balanced corpus
    print("\n[4] Assembling balanced training corpus...")
    final_lines: List[str] = []

    # Oversample QA pairs (x25) for high-signal prompt transitions while maintaining natural prose
    print("  [+] Oversampling QA dialogues (x25) for robust prompt matching...")
    for _ in range(25):
        final_lines.extend(qa_lines)

    # Add crawled data and essay lines for general vocabulary richness and syntax
    final_lines.extend(crawled_lines)
    final_lines.extend(essay_lines)

    print(f"  [+] Total training lines: {len(final_lines)}")

    # 4. Save to target files
    OUTPUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as out:
        for line in final_lines:
            out.write(line + "\n")
    print(f"  [+] Saved clean corpus: {OUTPUT_CORPUS} ({OUTPUT_CORPUS.stat().st_size / 1e6:.2f} MB)")

    # Also write directly to Metadata/processed_txt_corpus.txt
    metadata_corpus = ROOT_DIR / "Metadata" / "processed_txt_corpus.txt"
    metadata_corpus.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_corpus, "w", encoding="utf-8") as out:
        for line in final_lines:
            out.write(line + "\n")
    print(f"  [+] Saved metadata target: {metadata_corpus} ({metadata_corpus.stat().st_size / 1e6:.2f} MB)")

    print("\n" + "=" * 60)
    print("   Curated corpus generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
