"""
Hands-on AI Concepts: Tokens, Context Windows, Temperature, Top-p, Embeddings
================================================================================
Everything here runs offline with just numpy. Instead of calling a black-box
tokenizer/model API, we build tiny versions of each mechanism from scratch so
you can see exactly what's happening at every step.

Run this file section by section (or all at once):
    python ai_concepts_hands_on.py
"""

import re
import numpy as np
from collections import Counter

np.set_printoptions(precision=3, suppress=True)


# ============================================================================
# SECTION 1: TOKENS — building a tiny Byte-Pair Encoding (BPE) tokenizer
# ============================================================================
# Real tokenizers (GPT's tiktoken, Claude's tokenizer, etc.) use the same
# core idea: start with characters, then repeatedly merge the most frequent
# adjacent pair into a new token. This is exactly how "ization" ends up as
# its own token — it showed up often enough in training data to earn one.

print("=" * 70)
print("SECTION 1: TOKENS")
print("=" * 70)

def get_pair_counts(word_freqs):
    """Count frequency of every adjacent symbol pair across the vocabulary."""
    pairs = Counter()
    for word, freq in word_freqs.items():
        symbols = word.split()
        
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_pair(pair, word_freqs):
    """Merge every occurrence of `pair` into a single symbol."""
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    # print(pattern)
    new_word_freqs = {}
    for word, freq in word_freqs.items():
        new_word = pattern.sub("".join(pair), word)
        new_word_freqs[new_word] = freq
        # print(f"Merging {pair} into {new_word}")
    # print(new_word_freqs)
    return new_word_freqs

def train_bpe(corpus, num_merges=15):
    """Train a mini BPE tokenizer on a small corpus. Returns the merge rules."""
    # Start every word split into characters, with </w> marking word end
    word_freqs = Counter()
    for word in corpus.split():
        spaced = " ".join(list(word)) + " </w>"
        word_freqs[spaced] += 1
    merges = []
    for _ in range(num_merges):
        pairs = get_pair_counts(word_freqs)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        word_freqs = merge_pair(best_pair, word_freqs)
        merges.append(best_pair)
    return merges

def apply_bpe(text, merges):
    """Tokenize new text using learned merge rules."""
    tokens = []
    for word in text.split():
        symbols = list(word) + ["</w>"]
        symbols_str = " ".join(symbols)
        for pair in merges:
            bigram = re.escape(" ".join(pair))
            pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
            symbols_str = pattern.sub("".join(pair), symbols_str)
            # print(f"After merging {pair} with pattern {pattern}: {symbols_str}")
        tokens.extend(symbols_str.split())
        # print(tokens)
    return [t.replace("</w>", "") for t in tokens]

# Train on a small corpus so "tokenization" and "token" share sub-word pieces
corpus = (
    "tokenization token tokens tokenize tokenizer "
    "organization organize organized organizer organizes "
    "the quick brown fox jumps over the lazy dog "
) * 3

# print(corpus.split())
merges = train_bpe(corpus, num_merges=40)
# print(merges)
sample = "The tokenizer organizes tokens."
cleaned = re.sub(r"([.!?,'])", r" \1 ", sample.lower())
cleaned = re.sub(r"\s+", " ", cleaned).strip()
tokens = [t for t in apply_bpe(cleaned, merges) if t]
print(f"Input text: {sample!r}")
print(f"Tokens ({len(tokens)}): {tokens}")
print("Notice: 'organizes' splits into 'organize' + 's' — the model learned")
print("'organize' as a whole chunk (it saw it a lot) but 's' stayed separate.\n")


# ============================================================================
# SECTION 2: CONTEXT WINDOW — a sliding token buffer
# ============================================================================
print("=" * 70)
print("SECTION 2: CONTEXT WINDOW")
print("=" * 70)

def simulate_context_window(conversation_tokens, max_context=20):
    """
    Simulate what happens when a conversation exceeds the context window.
    Oldest tokens get dropped first (simplest truncation strategy).
    """
    total = sum(len(t) for t in conversation_tokens)
    kept = []
    running = 0
    # Walk backwards from the newest message, keep what fits
    for turn in reversed(conversation_tokens):
        if running + len(turn) > max_context:
            break
        kept.insert(0, turn)
        running += len(turn)
    return kept, total, running

conversation = [
    ["hello", "there", "!"],                                   # turn 1
    ["what", "is", "the", "capital", "of", "France", "?"],      # turn 2
    ["Paris", "is", "the", "capital", "of", "France", "."],     # turn 3
    ["and", "what", "is", "its", "population", "?"],            # turn 4
]

kept, total, running = simulate_context_window(conversation, max_context=15)
print(f"Full conversation: {total} tokens")
print(f"Context window limit: 15 tokens")
print(f"Tokens that survive: {running} -> kept turns: {kept}")
print("The earliest turn(s) got dropped once the budget ran out — this is")
print("why a model can 'forget' the start of a long chat.\n")


# ============================================================================
# SECTION 3: TEMPERATURE & TOP-P — controlling next-token sampling
# ============================================================================
print("=" * 70)
print("SECTION 3: TEMPERATURE & TOP-P")
print("=" * 70)

# Pretend these are raw model scores (logits) for the next word after
# "The weather today is"
vocab = ["sunny", "cloudy", "rainy", "cold", "purple", "velociraptor"]
logits = np.array([4.0, 3.0, 2.7, 2.3, -1.0, -1.5])

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def apply_temperature(logits, temperature):
    """Lower temperature sharpens the distribution; higher flattens it."""
    return softmax(logits / max(temperature, 1e-6))

def apply_top_p(probs, p):
    """Keep only the smallest set of top tokens whose probs sum to >= p."""
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative, p) + 1
    mask = np.zeros_like(probs, dtype=bool)
    mask[order[:cutoff]] = True
    filtered = np.where(mask, probs, 0.0)
    return filtered / filtered.sum()

print(f"Vocabulary: {vocab}")
print(f"Raw logits: {logits}\n")

for temp in [0.2, 0.7, 1.3]:
    probs = apply_temperature(logits, temp)
    print(f"Temperature={temp}:")
    for word, p in zip(vocab, probs):
        print(f"    {word:14s} {p:.3f}")
    print()

print("Now applying top-p=0.7 on top of temperature=0.7:")
probs_t07 = apply_temperature(logits, 0.7)
probs_top_p = apply_top_p(probs_t07, p=0.7)
for word, p in zip(vocab, probs_top_p):
    marker = "  <- excluded" if p == 0 else ""
    print(f"    {word:14s} {p:.3f}{marker}")
print("\ntop-p hard-excludes the unlikely tail ('purple', 'velociraptor')")
print("no matter how temperature reshapes the rest of the distribution.\n")


# ============================================================================
# SECTION 4: EMBEDDINGS — vectors, cosine similarity, and analogies
# ============================================================================
print("=" * 70)
print("SECTION 4: EMBEDDINGS")
print("=" * 70)

# Hand-crafted 4D toy embeddings (real ones are 768-4096 dims, trained on
# huge corpora) — dimensions loosely mean: [royalty, gender, food, size]
embeddings = {
    "king":    np.array([0.90, 0.80, 0.05, 0.50]),
    "queen":   np.array([0.88, 0.10, 0.05, 0.45]),
    "man":     np.array([0.10, 0.85, 0.05, 0.55]),
    "woman":   np.array([0.08, 0.12, 0.05, 0.50]),
    "banana":  np.array([0.02, 0.05, 0.90, 0.20]),
    "apple":   np.array([0.03, 0.04, 0.88, 0.15]),
}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Pairwise cosine similarities:")
words = list(embeddings.keys())
for i in range(len(words)):
    for j in range(i + 1, len(words)):
        w1, w2 = words[i], words[j]
        sim = cosine_similarity(embeddings[w1], embeddings[w2])
        print(f"    {w1:8s} <-> {w2:8s}  {sim:.3f}")

# The classic analogy: king - man + woman ~= queen
analogy_vector = embeddings["king"] - embeddings["man"] + embeddings["woman"]
print(f"\nking - man + woman = {analogy_vector}")

best_word, best_sim = None, -1
for word, vec in embeddings.items():
    if word == "king":
        continue
    sim = cosine_similarity(analogy_vector, vec)
    if sim > best_sim:
        best_word, best_sim = word, sim
print(f"Closest match: '{best_word}' (similarity={best_sim:.3f}) — as expected, 'queen'")
print("\nThis is the same principle real semantic search and RAG systems use:")
print("embed a query, embed a document corpus, retrieve by nearest vectors.")