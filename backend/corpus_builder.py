"""
corpus_builder.py
=================
Downloads public-domain medical texts from Project Gutenberg, cleans them,
and produces three artefacts used by the Flask server:

    data/corpus.txt          – cleaned raw text  (for reference / retraining)
    data/frequency_dict.txt  – SymSpell word-frequency list  (word<TAB>count)
    data/bigrams.json        – bigram counts  { "word_a word_b": count, ... }

Run ONCE before starting the server:
    python corpus_builder.py

Requires: nltk  (pip install nltk)
"""

import os
import re
import json
import urllib.request
from collections import Counter, defaultdict

import nltk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CORPUS_PATH    = os.path.join(DATA_DIR, "corpus.txt")
FREQ_DICT_PATH = os.path.join(DATA_DIR, "frequency_dict.txt")
BIGRAMS_PATH   = os.path.join(DATA_DIR, "bigrams.json")

# Public-domain medical texts from Project Gutenberg
GUTENBERG_URLS = [
    # Gray's Anatomy (~430 000 words)
    "https://www.gutenberg.org/cache/epub/40726/pg40726.txt",
    # The Merck Manual 1899 edition
    "https://www.gutenberg.org/cache/epub/4483/pg4483.txt",
    # Materia Medica
    "https://www.gutenberg.org/cache/epub/17911/pg17911.txt",
    # A System of Practical Medicine
    "https://www.gutenberg.org/cache/epub/12224/pg12224.txt",
    # Anatomy of the Human Body (Cunningham)
    "https://www.gutenberg.org/cache/epub/4141/pg4141.txt",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_text(url: str) -> str:
    """Download a plain-text file and return its content as a string."""
    print(f"  Downloading {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read().decode("utf-8", errors="ignore")
        # Strip Project Gutenberg header/footer boilerplate
        start = raw.find("*** START OF")
        end   = raw.find("*** END OF")
        if start != -1:
            raw = raw[start + 50:]
        if end != -1:
            raw = raw[:end]
        return raw
    except Exception as exc:
        print(f"  WARNING: could not download {url}: {exc}")
        return ""


def clean_text(text: str) -> str:
    """Lower-case, remove non-alpha characters, normalise whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenise(text: str):
    """Simple whitespace tokeniser; returns a list of lowercase word strings."""
    return [w for w in text.split() if len(w) >= 2]


# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------

def build_corpus():
    print("=== Corpus Builder ===")

    # 1. Download all sources
    raw_parts = []
    for url in GUTENBERG_URLS:
        raw_parts.append(download_text(url))

    combined_raw = "\n".join(raw_parts)

    # 2. Also pull NLTK's built-in medical/general corpora as supplement
    print("  Loading NLTK corpora ...")
    try:
        nltk.download("gutenberg",  quiet=True)
        nltk.download("brown",      quiet=True)
        nltk.download("reuters",    quiet=True)
        nltk.download("punkt",      quiet=True)

        from nltk.corpus import gutenberg, brown, reuters
        # Reuters has a lot of factual prose; Brown has diverse domain text
        nltk_text = " ".join(
            gutenberg.raw().split()[:200_000]
            + brown.raw().split()[:200_000]
            + reuters.raw().split()[:100_000]
        )
        combined_raw += "\n" + nltk_text
        print("  NLTK corpora loaded.")
    except Exception as exc:
        print(f"  WARNING: NLTK corpora unavailable: {exc}")

    # 3. Clean
    print("  Cleaning text ...")
    cleaned = clean_text(combined_raw)
    tokens  = tokenise(cleaned)
    print(f"  Total tokens: {len(tokens):,}")

    # 4. Save corpus.txt
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        f.write(cleaned)
    print(f"  Saved corpus.txt  ({os.path.getsize(CORPUS_PATH) // 1024} KB)")

    # 5. Build unigram frequency dict  (SymSpell format: word<TAB>count)
    print("  Building frequency dictionary ...")
    freq: Counter = Counter(tokens)
    # Keep only words that appear at least twice (reduces noise)
    freq = {w: c for w, c in freq.items() if c >= 2}

    with open(FREQ_DICT_PATH, "w", encoding="utf-8") as f:
        for word, count in sorted(freq.items(), key=lambda x: -x[1]):
            f.write(f"{word}\t{count}\n")
    print(f"  Saved frequency_dict.txt  ({len(freq):,} unique words)")

    # 6. Build bigram counts
    print("  Building bigram counts ...")
    bigram_counts: Counter = Counter()
    for i in range(len(tokens) - 1):
        bigram_counts[(tokens[i], tokens[i + 1])] += 1

    # Keep only bigrams seen at least twice
    bigrams_dict = {
        f"{a} {b}": count
        for (a, b), count in bigram_counts.items()
        if count >= 2
    }

    with open(BIGRAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(bigrams_dict, f)
    print(f"  Saved bigrams.json  ({len(bigrams_dict):,} bigrams)")

    print("\n=== Build complete. You can now start the server. ===")


if __name__ == "__main__":
    build_corpus()
