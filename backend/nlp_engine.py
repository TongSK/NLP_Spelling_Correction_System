"""
nlp_engine.py
=============
Core NLP engine for the Spelling Correction System.

Techniques implemented
----------------------
1. Hybrid Dictionary  – merges SymSpell (specialized medical corpus, O(1) lookup) 
                        with PySpellChecker (standard English) for comprehensive coverage.
2. Bidirectional LM   – context scoring utilizing both Left (w_{i-1} -> w_i) and 
                        Right (w_i -> w_{i+1}) bigram log-probabilities.
3. Lemmatization      – uses NLTK WordNetLemmatizer to reduce words to their morphological 
                        roots, eliminating false positive context errors on plurals.
4. Noisy-Channel      – custom log-linear scoring model: 
                        Score = log(Freq) - (10 * EditDistance) + BigramLogProb + DomainBonus
5. Levenshtein        – dynamic programming minimum edit distance for UI annotation.

Public API
----------
    engine = NLPEngine()          # call once at startup
    result = engine.check(text)   # returns structured JSON-serialisable dict
"""



import os
import re
import json
import math
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from spellchecker import SpellChecker
import numpy as np
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

log = logging.getLogger(__name__)

# Paths
BASE_DIR  = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE_DIR, "data")
FREQ_PATH = os.path.join(DATA_DIR, "frequency_dict.txt")
BG_PATH   = os.path.join(DATA_DIR, "bigrams.json")

# 1. Levenshtein (edit distance)
def levenshtein(a: str, b: str) -> int:
    """

    Standard dynamic-programming Levenshtein distance.

    Counts insertions, deletions, and substitutions (each costs 1).

    """
    m, n = len(a), len(b)
    if a == b:      return 0
    if m == 0:      return n
    if n == 0:      return m

    # Only keep two rows to save memory
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j]     + 1,      # deletion
                curr[j - 1] + 1,      # insertion
                prev[j - 1] + cost,   # substitution
            )
        prev, curr = curr, prev

    return prev[n]

# 2. Bigram language model
class BigramModel:
    """
    Simple bigram language model with add-k smoothing.

    Loaded from data/bigrams.json produced by corpus_builder.py.

    """

    def __init__(self, path: str, k: float = 0.5):

        self.k = k
        self.counts: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.vocab_size = 0

        if not os.path.exists(path):
            log.warning("bigrams.json not found — bigram model disabled.")
            return

        with open(path, encoding="utf-8") as f:
            raw: Dict[str, int] = json.load(f)

        vocab = set()

        for pair, count in raw.items():
            parts = pair.split(" ", 1)
            if len(parts) != 2:
                continue
            a, b = parts
            self.counts[a][b] = count
            vocab.update([a, b])



        self.vocab_size = len(vocab)
        log.info("BigramModel: loaded %d bigrams, vocab=%d", len(raw), self.vocab_size)



    def prob(self, prev: str, word: str) -> float:
        """P(word | prev) with add-k smoothing."""
        if not self.counts or not prev:
            return 1e-6

        context = self.counts.get(prev, {})
        total   = sum(context.values()) + self.k * max(self.vocab_size, 1)
        count   = context.get(word, 0) + self.k
        return count / total



    def log_prob(self, prev: str, word: str) -> float:
        return math.log(self.prob(prev, word) + 1e-12)

# 3. SymSpell wrapper
class SymSpellWrapper:
    """
    Wraps the symspellpy library for fast non-word candidate generation.

    Falls back to a simple brute-force search if symspellpy is not installed.
    """

    def __init__(self, freq_path: str, max_edit: int = 2):

        self.max_edit   = max_edit
        self.sym_spell  = None
        self._freq: Dict[str, int] = {}



        # Load frequency dict regardless (needed for fallback + scoring)
        if os.path.exists(freq_path):
            with open(freq_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip().split("\t")
                    if len(parts) == 2:
                        self._freq[parts[0]] = int(parts[1])

        else:
            log.warning("frequency_dict.txt not found — run corpus_builder.py first.")



        # Try to load symspellpy
        try:
            from symspellpy import SymSpell, Verbosity
            self._Verbosity = Verbosity
            ss = SymSpell(max_dictionary_edit_distance=max_edit, prefix_length=7)

            if os.path.exists(freq_path):
                ss.load_dictionary(freq_path, term_index=0, count_index=1, separator="\t")
                self.sym_spell = ss
                log.info("SymSpell loaded: %d entries", len(self._freq))
            else:
                log.warning("SymSpell: no frequency dict — using fallback.")

        except ImportError:
            log.warning("symspellpy not installed — using brute-force fallback.")



    @property
    def vocab(self) -> set:
        return set(self._freq.keys())

    def frequency(self, word: str) -> int:
        return self._freq.get(word, 0)

    def lookup(self, word: str, max_edit: Optional[int] = None, all_candidates: bool = False) -> List[Dict]:
        """
        Return a list of candidate dicts:

            { word, edit_distance, frequency }

        Sorted by edit_distance ASC, frequency DESC.
        """

        ed = max_edit if max_edit is not None else self.max_edit

        if self.sym_spell:
            from symspellpy import Verbosity

            # Force SymSpell to find all neighbours, not just the distance-0 exact match
            v = Verbosity.ALL if all_candidates else Verbosity.CLOSEST

            suggestions = self.sym_spell.lookup(word, v, ed)

            return [
                {
                    "word"          : s.term,
                    "edit_distance" : s.distance,
                    "frequency"     : s.count,
                }
                for s in suggestions
                ]



        # ---- Brute-force fallback ----
        candidates = []

        for vocab_word in self._freq:
            d = levenshtein(word, vocab_word)

            if 0 < d <= ed:
                candidates.append({
                    "word"          : vocab_word,
                    "edit_distance" : d,
                    "frequency"     : self._freq[vocab_word],
                })

        candidates.sort(key=lambda x: (x["edit_distance"], -x["frequency"]))

        return candidates[:10]

# Real-word confusables table
CONFUSABLES: Dict[str, List[str]] = {
    "affect"     : ["effect"],
    "effect"     : ["affect"],
    "their"      : ["there", "they're"],
    "there"      : ["their", "they're"],
    "dose"       : ["does"],
    "does"       : ["dose"],
    "liver"      : ["lifer"],
    "principal"  : ["principle"],
    "principle"  : ["principal"],
    "compliment" : ["complement"],
    "complement" : ["compliment"],
    "discrete"   : ["discreet"],
    "discreet"   : ["discrete"],
    "elicit"     : ["illicit"],
    "illicit"    : ["elicit"],
    "eminent"    : ["imminent"],
    "imminent"   : ["eminent"],
    "precede"    : ["proceed"],
    "proceed"    : ["precede"],
    "oral"       : ["aural"],
    "ileum"      : ["ilium"],
    "mucus"      : ["mucous"],
    "mucous"     : ["mucus"],
    "pore"       : ["pour"],
    "site"       : ["sight", "cite"],
    "cite"       : ["site", "sight"],
    "colon"      : ["cologne"],
    "patient"    : ["patience"],
    "course"     : ["coarse"],
    "plain"      : ["plane"],
    "right"      : ["write", "rite"],
    "than"       : ["then"],
    "then"       : ["than"],
    "accept"     : ["except"],
    "except"     : ["accept"],
    "lose"       : ["loose"],
    "loose"      : ["lose"],
    "advice"     : ["advise"],
    "advise"     : ["advice"],
    "breath"     : ["breathe"],
    "breathe"    : ["breath"],
    "conscious"  : ["conscience"],
    "conscience" : ["conscious"],
}

# Main NLP Engine
class NLPEngine:
    """
    Orchestrates all NLP components into a single spell-check pipeline.

    check(text) → {
        tokens : [ { word, index, type, candidates } ],
        stats  : { total, nonword, realword, correct }
    }
    """

    def __init__(self):

        log.info("Initialising NLP Engine ...")

        self.symspell = SymSpellWrapper(FREQ_PATH)
        self.bigram   = BigramModel(BG_PATH)
        self.checker  = SpellChecker() #Changed
        self.vocab    = self.symspell.vocab

        self.lemmatizer = WordNetLemmatizer()

        log.info("NLP Engine ready. vocab=%d, Statistical Model Active", len(self.vocab))

    # Public API
    def check(self, text: str, ignored_indices: Optional[List[int]] = None) -> Dict:
        """
        Full pipeline: tokenise → detect errors → generate suggestions.
        Returns a JSON-serialisable dict.
        """

        # ignored_indices: A list of token_indexes that the user explicitly approved.
        if ignored_indices is None:
            ignored_indices = []

        tokens      = self._tokenise(text)
        word_tokens = [(i, t) for i, t in enumerate(tokens) if re.search(r"[a-zA-Z]", t)]

        results = []

        stats   = {"total": len(word_tokens), "nonword": 0, "realword": 0, "correct": 0}

        for pos, (tok_idx, tok) in enumerate(word_tokens):
            # Ignorer logic
            # If the user explicitly approved this word's index, leave it alone!
            if tok_idx in ignored_indices:
                stats["correct"] += 1
                continue

            clean_word = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", tok).lower()

            if not clean_word or len(clean_word) < 2:
                stats["correct"] += 1
                continue

            prev_word = word_tokens[pos - 1][1].lower() if pos > 0 else None
            next_word = word_tokens[pos + 1][1].lower() if pos < len(word_tokens) - 1 else None

            if clean_word not in self.vocab:
                # Before we declare it an error, check if it's a valid standard English word!
                # (PySpellChecker allows you to check if a word is in its dictionary using 'in')
                if clean_word in self.checker:
                    stats["correct"] += 1
                    continue  # Skip to the next word, it's spelled correctly!

                candidates = self._nonword_candidates(clean_word, prev_word, text, pos)

                results.append({
                    "token_index" : tok_idx,
                    "word"        : tok,
                    "clean"       : clean_word,
                    "type"        : "nonword",
                    "candidates"  : candidates,
                })

                stats["nonword"] += 1

            # ---- Real-word error ----
            elif self._is_realword_error(clean_word, prev_word, next_word, text, pos):
                candidates = self._realword_candidates(clean_word, prev_word, next_word, text, pos)

                results.append({
                    "token_index" : tok_idx,
                    "word"        : tok,
                    "clean"       : clean_word,
                    "type"        : "realword",
                    "candidates"  : candidates,
                })

                stats["realword"] += 1

            else:
                stats["correct"] += 1

        return {
            "tokens"  : self._tokenise(text),  # raw token list for reconstruction
            "errors"  : results,
            "stats"   : stats,
            "bert_on" : False, # Hardcoded to False so the frontend gracefully shows the statistical fallback badge
        }


    # Non-word pipeline
    def _nonword_candidates(
        self,
        word: str,
        prev_word: Optional[str],
        sentence: str,
        word_pos: int,
    ) -> List[Dict]:
        """
        Generate and rank candidates for a non-word error.
        Ranking: Merges SymSpell (Medical) and PySpellChecker (Standard English).
        """

        sym_candidates = self.symspell.lookup(word, max_edit=2)

        # Store unique candidates to avoid duplicates between dictionaries
        unique_cands = {}

        # Add SymSpell candidates (from medical corpus)
        for c in sym_candidates:
            unique_cands[c["word"]] = {
                "word": c["word"],
                "edit_distance": c["edit_distance"],
                "frequency": c["frequency"] or 1,
                "is_corpus": True,  # Tag as a trusted medical word

            }

        # Add PySpellChecker candidates (from standard English dictionary)
        english_cands = self.checker.candidates(word)

        if english_cands:
            for ew in english_cands:
                if ew not in unique_cands:
                    # Tap into PySpellChecker's internal frequency dictionary to break ties!
                    # If it can't find it, it defaults to 5.
                    real_freq = self.checker.word_frequency.dictionary.get(ew, 5)

                    unique_cands[ew] = {
                        "word": ew,
                        "edit_distance": levenshtein(word, ew),
                        "frequency": real_freq,
                        "is_corpus": False, # Tag as an external word
                    }

        if not unique_cands:
            return []

        scored = []

        for cw, data in unique_cands.items():
            dist = data["edit_distance"]
            freq = data["frequency"]
            is_corpus = data["is_corpus"] # Retrieve the tag

            # Base Word Frequency (Positive number: usually 1 to 8)
            base_freq_score = math.log(freq + 1)

            # Edit Distance Penalty (Massive negative weight: -10 per typo)
            # A 1-typo word loses 10 points. A 2-typo word loses 20 points.
            dist_penalty = 10.0 * dist

            # Bigram Context (Negative number: usually -2 to -15)
            bg_score = self.bigram.log_prob(prev_word, cw) if prev_word else 0.0

            # Domain Prioritization Bonus (Changed logic)
            # Give a massive +5 point boost if the word belongs to our medical corpus
            domain_bonus = 5.0 if is_corpus else 0.0

            # Final Combined Log-Linear Score
            final = base_freq_score - dist_penalty + (0.5 * bg_score) + domain_bonus

            scored.append({
                "word"          : cw,
                "edit_distance" : dist,
                "frequency"     : freq,
                "score"         : round(final, 4),
            })

        scored.sort(key=lambda x: -x["score"])
        return scored[:8]


    # Real-word pipeline
    def _is_realword_error(
        self,
        word: str,
        prev_word: Optional[str],
        next_word: Optional[str],
        sentence: str,
        word_pos: int,
    ) -> bool:
        """
        Returns True if the word is a valid vocabulary word but likely wrong in context.
        Strategy: Bigram probability comparison against known confusables.
        """

        # Start with known hardcoded confusables as a Set to avoid duplicates
        confusable_alts = set(CONFUSABLES.get(word, []))

        # Dynamically add all valid corpus words within 1 edit distance
        for c in self.symspell.lookup(word, max_edit=1, all_candidates=True):
            alt_word = c["word"]
            if alt_word != word and alt_word in self.vocab:
                confusable_alts.add(alt_word)

        # Primary Statistical Check: Bigram context
        if prev_word or next_word:
            # Calculate the combined score of the user's original word
            # If a neighbor doesn't exist, we just multiply by 1.0 (neutral)
            self_prob_prev = self.bigram.prob(prev_word, word) if prev_word else 1.0
            self_prob_next = self.bigram.prob(word, next_word) if next_word else 1.0
            self_total_prob = self_prob_prev * self_prob_next

            # Lemmatize the original word
            lemma_word = self.lemmatizer.lemmatize(word)

            for alt in confusable_alts:
                if alt in self.vocab:
                    # Lemmatize the alternative word
                    lemma_alt = self.lemmatizer.lemmatize(alt)

                    # STRATEGY APPLIED: If roots are the same (e.g. kidney == kidneys), ignore!
                    if lemma_word == lemma_alt:
                        continue

                    # Calculate the combined score of the alternative word
                    alt_prob_prev = self.bigram.prob(prev_word, alt) if prev_word else 1.0
                    alt_prob_next = self.bigram.prob(alt, next_word) if next_word else 1.0
                    alt_total_prob = alt_prob_prev * alt_prob_next

                    if alt_total_prob > self_total_prob * 10.0: # Margin for bigram threshold
                        return True

        return False

    def _realword_candidates(
        self,
        word: str,
        prev_word: Optional[str],
        next_word: Optional[str], # changed logic
        sentence: str,
        word_pos: int,
    ) -> List[Dict]:
        """
        Candidates for a real-word error.
        Combines known confusables + SymSpell neighbours, re-ranked by Bigram score.
        """

        alts = set(CONFUSABLES.get(word, []))

        # Add close edit-distance neighbours from SymSpell
        for c in self.symspell.lookup(word, max_edit=1, all_candidates=True):
            alts.add(c["word"])

        alts.discard(word)

        scored = []

        for alt in alts:
            if alt not in self.vocab:
                continue

            dist  = levenshtein(word, alt)

            freq  = self.symspell.frequency(alt) or 1

            # BIDIRECTIONAL Bigram context
            # We use log_prob here, so we ADD them together instead of multiplying
            bg_score_prev = self.bigram.log_prob(prev_word, alt) if prev_word else 0.0
            bg_score_next = self.bigram.log_prob(alt, next_word) if next_word else 0.0
            bg_score = bg_score_prev + bg_score_next

            # Combined score
            final = math.log(freq + 1) + 0.5 * bg_score - (1.0 * dist)

            scored.append({
                "word"          : alt,
                "edit_distance" : dist,
                "frequency"     : freq,
                "score"         : round(final, 4),
            })

        scored.sort(key=lambda x: -x["score"])

        return scored[:8]

    # Tokeniser
    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """
        Split text into alternating word / non-word tokens.
        Preserves original spacing and punctuation for reconstruction.
        """

        return re.findall(r"[a-zA-Z''\-]+|[^a-zA-Z''\-]+", text) or []