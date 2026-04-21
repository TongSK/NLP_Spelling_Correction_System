"""
nlp_engine.py
=============
Core NLP engine for the Spelling Correction System.

Techniques implemented
----------------------
1. SymSpell         – ultra-fast candidate generation for non-word errors
                      (delete-only pre-computation, O(1) average lookup)
2. Levenshtein      – minimum edit distance, used to annotate every suggestion
3. BERT MLM         – masked language model (bert-base-uncased) for
                      real-word error detection and context-aware re-ranking
4. Bigram LM        – fallback context scoring when BERT is slow / unavailable
5. Noisy-channel    – final candidate score = P(word) × P(error|word)

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

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE_DIR, "data")
FREQ_PATH = os.path.join(DATA_DIR, "frequency_dict.txt")
BG_PATH   = os.path.join(DATA_DIR, "bigrams.json")

# ---------------------------------------------------------------------------
# 1. Levenshtein (edit distance)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Bigram language model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. SymSpell wrapper
# ---------------------------------------------------------------------------

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
                ss.load_dictionary(freq_path, term_index=0, count_index=1)
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

    def lookup(self, word: str, max_edit: Optional[int] = None) -> List[Dict]:
        """
        Return a list of candidate dicts:
            { word, edit_distance, frequency }
        Sorted by edit_distance ASC, frequency DESC.
        """
        ed = max_edit if max_edit is not None else self.max_edit

        if self.sym_spell:
            from symspellpy import Verbosity
            suggestions = self.sym_spell.lookup(
                word,
                Verbosity.CLOSEST,
                max_edit,
            )
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


# ---------------------------------------------------------------------------
# 4. BERT masked language model
# ---------------------------------------------------------------------------

class BERTScorer:
    """
    Uses bert-base-uncased as a masked language model to:
      (a) score how likely a word is in a given sentence context
      (b) generate top fill-mask predictions for a [MASK] position

    This gives context-aware re-ranking far beyond what a bigram model can do.
    """

    MODEL_NAME = "bert-base-uncased"

    def __init__(self):
        self.pipeline = None
        self.tokenizer = None
        self.model = None

        try:
            from transformers import pipeline, BertTokenizer, BertForMaskedLM
            import torch

            log.info("Loading BERT model '%s' ...", self.MODEL_NAME)
            self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
            self.model     = BertForMaskedLM.from_pretrained(self.MODEL_NAME)
            self.model.eval()
            self._torch = torch
            log.info("BERT model loaded.")
        except Exception as exc:
            log.warning("BERT unavailable (%s) — falling back to bigram LM.", exc)

    @property
    def available(self) -> bool:
        return self.model is not None

    def score_word_in_context(self, sentence: str, target_word: str, word_index: int) -> float:
        """
        Replace target_word with [MASK] and return the log-probability
        that BERT predicts target_word at that position.

        Parameters
        ----------
        sentence    : full sentence string
        target_word : the word whose probability we want
        word_index  : 0-based index of target_word in the word list

        Returns
        -------
        log-probability (float, higher = more likely)
        """
        if not self.available:
            return -999.0

        try:
            words   = sentence.split()
            masked  = words[:]
            masked[word_index] = "[MASK]"
            masked_sentence = " ".join(masked)

            inputs = self.tokenizer(masked_sentence, return_tensors="pt")
            with self._torch.no_grad():
                outputs = self.model(**inputs)

            logits      = outputs.logits  # (1, seq_len, vocab)
            # Find position of [MASK] in token sequence
            mask_pos    = (inputs["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_pos) == 0:
                return -999.0

            mask_logits = logits[0, mask_pos[0], :]   # (vocab_size,)
            log_probs   = self._torch.nn.functional.log_softmax(mask_logits, dim=-1)

            token_ids = self.tokenizer.encode(target_word, add_special_tokens=False)
            if not token_ids:
                return -999.0
            # For single-token words use direct lookup; multi-token: sum log-probs
            score = float(log_probs[token_ids[0]])
            return score

        except Exception as exc:
            log.debug("BERT scoring error: %s", exc)
            return -999.0

    def top_predictions(self, sentence: str, word_index: int, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Return top-k BERT predictions for [MASK] at word_index.
        Used to re-rank SymSpell candidates.
        """
        if not self.available:
            return []

        try:
            words  = sentence.split()
            masked = words[:]
            masked[word_index] = "[MASK]"
            masked_sentence = " ".join(masked)

            inputs = self.tokenizer(masked_sentence, return_tensors="pt")
            with self._torch.no_grad():
                outputs = self.model(**inputs)

            logits   = outputs.logits
            mask_pos = (inputs["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_pos) == 0:
                return []

            mask_logits = logits[0, mask_pos[0], :]
            log_probs   = self._torch.nn.functional.log_softmax(mask_logits, dim=-1)

            top_ids     = self._torch.topk(log_probs, top_k * 5).indices.tolist()
            results     = []
            for tid in top_ids:
                token = self.tokenizer.decode([tid]).strip()
                if re.match(r"^[a-z]+$", token):
                    results.append((token, float(log_probs[tid])))
                if len(results) >= top_k:
                    break
            return results

        except Exception as exc:
            log.debug("BERT top_predictions error: %s", exc)
            return []


# ---------------------------------------------------------------------------
# 5. Real-word confusables table
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 6. Main NLP Engine
# ---------------------------------------------------------------------------

class NLPEngine:
    """
    Orchestrates all NLP components into a single spell-check pipeline.

    check(text) → {
        tokens : [ { word, index, type, candidates } ],
        stats  : { total, nonword, realword, correct }
    }
    """

    REAL_WORD_THRESHOLD = -2.0   # BERT log-prob below this → potential real-word error

    def __init__(self):
        log.info("Initialising NLP Engine ...")
        self.symspell = SymSpellWrapper(FREQ_PATH)
        self.bigram   = BigramModel(BG_PATH)
        self.bert     = BERTScorer()
        self.vocab    = self.symspell.vocab
        log.info("NLP Engine ready. vocab=%d, BERT=%s",
                 len(self.vocab), "ON" if self.bert.available else "OFF")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, text: str) -> Dict:
        """
        Full pipeline: tokenise → detect errors → generate suggestions.
        Returns a JSON-serialisable dict.
        """
        tokens      = self._tokenise(text)
        word_tokens = [(i, t) for i, t in enumerate(tokens) if re.search(r"[a-zA-Z]", t)]

        results = []
        stats   = {"total": len(word_tokens), "nonword": 0, "realword": 0, "correct": 0}

        for pos, (tok_idx, tok) in enumerate(word_tokens):
            clean_word = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", tok).lower()
            if not clean_word or len(clean_word) < 2:
                stats["correct"] += 1
                continue

            prev_word = word_tokens[pos - 1][1].lower() if pos > 0 else None
            next_word = word_tokens[pos + 1][1].lower() if pos < len(word_tokens) - 1 else None

            # ---- Non-word error ----
            if clean_word not in self.vocab:
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
                candidates = self._realword_candidates(clean_word, prev_word, text, pos)
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
            "bert_on" : self.bert.available,
        }

    # ------------------------------------------------------------------
    # Non-word pipeline
    # ------------------------------------------------------------------

    def _nonword_candidates(
        self,
        word: str,
        prev_word: Optional[str],
        sentence: str,
        word_pos: int,
    ) -> List[Dict]:
        """
        Generate and rank candidates for a non-word error.
        Ranking: SymSpell (edit distance + freq) × BERT context score.
        """
        sym_candidates = self.symspell.lookup(word, max_edit=2)

        if not sym_candidates:
            return []

        # BERT top predictions for this masked position
        bert_top = {}
        if self.bert.available:
            preds = self.bert.top_predictions(sentence, word_pos, top_k=20)
            bert_top = {w: score for w, score in preds}

        scored = []
        for c in sym_candidates:
            cw   = c["word"]
            dist = c["edit_distance"]
            freq = c["frequency"] or 1

            # Noisy-channel prior
            noisy_score = math.log(freq + 1) / (dist + 1)

            # Bigram context
            bg_score = self.bigram.log_prob(prev_word, cw) if prev_word else 0.0

            # BERT context (if available)
            bert_score = bert_top.get(cw, -10.0) if bert_top else 0.0

            # Combined score
            final = noisy_score + 0.5 * bg_score + (1.0 * bert_score if self.bert.available else 0.0)

            scored.append({
                "word"          : cw,
                "edit_distance" : dist,
                "frequency"     : freq,
                "bert_score"    : round(bert_score, 4),
                "score"         : round(final, 4),
            })

        scored.sort(key=lambda x: -x["score"])
        return scored[:8]

    # ------------------------------------------------------------------
    # Real-word pipeline
    # ------------------------------------------------------------------

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

        Strategy:
          1. Check BERT log-probability; if very low AND a confusable alternative
             scores higher → flag as real-word error.
          2. Fallback: bigram probability comparison against known confusables.
        """
        # Check confusables table
        confusable_alts = CONFUSABLES.get(word, [])

        if self.bert.available:
            word_score = self.bert.score_word_in_context(sentence, word, word_pos)
            # If BERT confidence is very low, check whether an alternative scores better
            if word_score < self.REAL_WORD_THRESHOLD:
                for alt in confusable_alts:
                    if alt in self.vocab:
                        alt_score = self.bert.score_word_in_context(sentence, alt, word_pos)
                        if alt_score > word_score + 1.0:   # alt is meaningfully better
                            return True
            return False

        # Bigram fallback
        if prev_word:
            self_prob = self.bigram.prob(prev_word, word)
            for alt in confusable_alts:
                if alt in self.vocab:
                    alt_prob = self.bigram.prob(prev_word, alt)
                    if alt_prob > self_prob * 2.0:
                        return True
        return False

    def _realword_candidates(
        self,
        word: str,
        prev_word: Optional[str],
        sentence: str,
        word_pos: int,
    ) -> List[Dict]:
        """
        Candidates for a real-word error.
        Combines known confusables + SymSpell neighbours,
        re-ranked by BERT context score.
        """
        alts = set(CONFUSABLES.get(word, []))
        # Also add close edit-distance neighbours from SymSpell
        for c in self.symspell.lookup(word, max_edit=1):
            alts.add(c["word"])
        alts.discard(word)

        scored = []
        for alt in alts:
            if alt not in self.vocab:
                continue
            dist  = levenshtein(word, alt)
            freq  = self.symspell.frequency(alt) or 1

            bg_score = self.bigram.log_prob(prev_word, alt) if prev_word else 0.0

            bert_score = 0.0
            if self.bert.available:
                bert_score = self.bert.score_word_in_context(sentence, alt, word_pos)

            final = math.log(freq + 1) + 0.5 * bg_score + (1.5 * bert_score if self.bert.available else 0.0)

            scored.append({
                "word"          : alt,
                "edit_distance" : dist,
                "frequency"     : freq,
                "bert_score"    : round(bert_score, 4),
                "score"         : round(final, 4),
            })

        scored.sort(key=lambda x: -x["score"])
        return scored[:8]

    # ------------------------------------------------------------------
    # Tokeniser
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """
        Split text into alternating word / non-word tokens.
        Preserves original spacing and punctuation for reconstruction.
        """
        return re.findall(r"[a-zA-Z''\-]+|[^a-zA-Z''\-]+", text) or []
