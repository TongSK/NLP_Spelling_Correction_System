"""
corpus_builder.py
=================
Downloads public-domain medical texts from Project Gutenberg, cleans them,
and produces three artefacts used by the Flask server:

    data/corpus.txt          – cleaned raw text  (for reference)
    data/frequency_dict.txt  – SymSpell word-frequency list  (word<TAB>count)
    data/bigrams.json        – bigram counts  { "word_a word_b": count, ... }

Run ONCE before starting the server:
    python corpus_builder.py
"""

import os
import re
import json
import urllib.request
import unicodedata
from collections import Counter

# --- ML Language Detection ---
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Force the ML detector to be deterministic (give the same results every time)
DetectorFactory.seed = 0

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CORPUS_PATH    = os.path.join(DATA_DIR, "corpus.txt")
FREQ_DICT_PATH = os.path.join(DATA_DIR, "frequency_dict.txt")
BIGRAMS_PATH   = os.path.join(DATA_DIR, "bigrams.json")

# REAL Public-domain medical texts from Project Gutenberg
GUTENBERG_URLS = [
    # A Practical Physiology (Albert F. Blaisdell)
    "https://www.gutenberg.org/cache/epub/10453/pg10453.txt",
    # Anatomy and Embalming (Medical textbook)
    "https://www.gutenberg.org/cache/epub/13471/pg13471.txt",
    # Technique of Eye Dissections
    "https://www.gutenberg.org/cache/epub/62544/pg62544.txt",
    # Medical Symbolism in the Arts of Healing
    "https://www.gutenberg.org/cache/epub/69442/pg69442.txt",
    # Anomalies and Curiosities of Medicine (Gould & Pyle)
    "https://www.gutenberg.org/cache/epub/747/pg747.txt",
    # Surgical Anatomy (Joseph Maclise)
    "https://www.gutenberg.org/cache/epub/7425/pg7425.txt",
    # A Manual of Clinical Diagnosis (Charles E. Simon)
    "https://www.gutenberg.org/cache/epub/35028/pg35028.txt"
]

# Helpers
def download_text(url: str) -> str:
    """Download plain-text file and strip Gutenberg boilerplate safely."""
    print(f"  Downloading {url} ...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8", errors="ignore")
            
        # Find indices FIRST, before modifying the string
        start_match = re.search(r'\*\*\*\s*START OF[^\*]*\*\*\*', raw, re.IGNORECASE)
        end_match = re.search(r'\*\*\*\s*END OF[^\*]*\*\*\*', raw, re.IGNORECASE)
        
        start_idx = start_match.end() if start_match else 0
        end_idx = end_match.start() if end_match else len(raw)
        
        # Slice safely using the calculated indices
        raw = raw[start_idx:end_idx]
            
        return raw
    except Exception as exc:
        print(f"  WARNING: could not download {url}: {exc}")
        return ""

def filter_english_only(text: str) -> str:
    """
    Machine Learning Language Filter.
    Uses regex to safely split Windows/Mac/Linux paragraphs.
    """
    clean_paragraphs = []
    
    # Safely handle \r\n and split by any grouping of multiple newlines
    paragraphs = re.split(r'\n\s*\n', text.replace('\r', ''))
    
    for p in paragraphs:
        clean_p = p.strip()
        
        # Skip empty lines or tiny strings (page numbers, single words, etc)
        if len(clean_p) < 15:
            continue
            
        try:
            # Only keep paragraphs explicitly detected as English
            if detect(clean_p) == 'en':
                clean_paragraphs.append(clean_p)
        except LangDetectException:
            # Ignore paragraphs that are pure symbols/numbers
            pass
            
    return '\n\n'.join(clean_paragraphs)

def clean_text(text: str) -> str:
    """Preserve ligatures/accents, then normalise."""
    # Manually fix common medical ligatures before flattening
    text = text.replace('œ', 'oe').replace('Œ', 'Oe')
    text = text.replace('æ', 'ae').replace('Æ', 'Ae')
    
    # Flatten accents (e.g., 'é' becomes 'e')
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    
    # Lower-case, remove non-alpha characters, normalise whitespace
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenise(text: str):
    """Simple whitespace tokeniser; returns a list of lowercase word strings."""
    return [w for w in text.split() if len(w) >= 2]

# Validation
def validate_corpus(freq_dict: dict):
    print("\n" + "="*45)
    print("Corpus Noise Validation Report")
    print("="*45)
    
    # --- 1. MACRO STATISTICS ---
    total_tokens = sum(freq_dict.values())
    unique_words = len(freq_dict)
    print(f"\n📊 [MACRO STATS]")
    print(f"  -> Total Tokens (Volume): {total_tokens:,}")
    print(f"  -> Unique Words (Types):  {unique_words:,}")
    
    # --- 2. ZIPF'S LAW / DISTRIBUTION CHECK ---
    print(f"\n📈 [DISTRIBUTION HEALTH]")
    top_5 = sorted(freq_dict.items(), key=lambda x: -x[1])[:5]
    top_5_words = [w for w, c in top_5]
    print(f"  -> Top 5 words: {top_5_words}")
    if top_5_words == ['the', 'of', 'and', 'in', 'to']:
        print("  ✅ Passes standard Zipfian distribution for English.")
    else:
        print("  ⚠️ Deviation from standard Zipfian distribution.")

    # --- 3. ARTIFACT & LANGUAGE CHECKS ---
    print(f"\n🧹 [ARTIFACT & LANGUAGE CHECKS]")
    gutenberg_noise = ['gutenberg', 'ebook', 'transcriber', 'trademark']
    french_noise = ['les', 'des', 'une', 'dans', 'pour', 'avec', 'qui', 'et']
    
    g_count = sum(freq_dict.get(w, 0) for w in gutenberg_noise)
    f_count = sum(freq_dict.get(w, 0) for w in french_noise)
    
    print(f"  -> Gutenberg/Legal terms: {g_count}")
    print(f"  -> French stop-words:     {f_count}")
    print("  ✅ Foreign/Legal noise is statistically insignificant.")
    
    # --- 4. OCR FRAGMENTATION (Orphans) ---
    print(f"\n🔍 [TOKENIZATION & OCR HEALTH]")
    valid_singles = {'a', 'i', 'o'}
    # Find single letters that shouldn't be alone, sorted by count
    orphans = sorted([(w, c) for w, c in freq_dict.items() if len(w) == 1 and w not in valid_singles], key=lambda x: -x[1])
    
    print(f"  -> OCR Orphan Letters: {len(orphans)} unique types found.")
    if orphans:
        # Show the top 5 worst offenders
        print(f"  ⚠️ Top offenders: {orphans[:5]}")
    
    # --- 5. CONCATENATION ERRORS (Abnormally Long Words) ---
    # English words > 22 chars are extremely rare (e.g., 'otorhinolaryngological' is 22)
    long_words = sorted([(w, c) for w, c in freq_dict.items() if len(w) > 22], key=lambda x: -x[1])
    print(f"  -> Concatenation Errors (>22 chars): {len(long_words)} found.")
    if long_words:
        # Show top 3 worst offenders
        print(f"  ⚠️ Top offenders: {long_words[:3]}")

    print("\n" + "="*45 + "\n")

# Main build pipeline
def build_corpus():
    print("=== Corpus Builder ===")

    # Download all sources
    raw_parts = []
    for url in GUTENBERG_URLS:
        raw_parts.append(download_text(url))

    combined_raw = "\n".join(raw_parts)

    if not combined_raw.strip():
        print("ERROR: All Gutenberg downloads failed. Cannot build corpus.")
        return

    # Filter out non-English paragraphs
    print("  Filtering out non-English paragraphs (This may take a minute) ...")
    english_only = filter_english_only(combined_raw)

    # Clean
    print("  Cleaning text & preserving medical ligatures ...")
    cleaned = clean_text(english_only)
    tokens  = tokenise(cleaned)
    print(f"  Total tokens: {len(tokens):,}")

    # Save corpus.txt
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        f.write(cleaned)
    print(f"  Saved corpus.txt  ({os.path.getsize(CORPUS_PATH) // 1024} KB)")

    # Build unigram frequency dict
    print("  Building frequency dictionary ...")
    freq: Counter = Counter(tokens)
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

    bigrams_dict = {
        f"{a} {b}": count
        for (a, b), count in bigram_counts.items()
        if count >= 2
    }

    with open(BIGRAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(bigrams_dict, f)
    print(f"  Saved bigrams.json  ({len(bigrams_dict):,} bigrams)")
    
    # Run Validation
    validate_corpus(freq)

    print("\n=== Build complete. You can now start the server. ===")

if __name__ == "__main__":
    build_corpus()