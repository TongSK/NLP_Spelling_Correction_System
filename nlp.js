/**
 * nlp.js
 * Core NLP engine for the Probabilistic Spelling Correction System.
 *
 * Implements:
 *   1. Minimum Edit Distance (Levenshtein) — dynamic programming
 *   2. Non-word error detection — out-of-vocabulary lookup
 *   3. Real-word error detection — bigram language model context
 *   4. Noisy channel model — P(correction) × P(error|correction)
 *   5. Candidate generation and ranking
 */

/* ================================================================
   1. MINIMUM EDIT DISTANCE (LEVENSHTEIN)
   ================================================================ */

/**
 * Compute the Levenshtein edit distance between two strings.
 * Operations: insertion, deletion, substitution — each costs 1.
 *
 * @param {string} a - Source string
 * @param {string} b - Target string
 * @returns {number} Edit distance
 */
function levenshtein(a, b) {
  const m = a.length;
  const n = b.length;

  // Early exit optimisations
  if (a === b) return 0;
  if (m === 0) return n;
  if (n === 0) return m;
  if (Math.abs(m - n) > 3) return Math.abs(m - n); // prune obviously far candidates

  // Allocate DP table
  const dp = [];
  for (let i = 0; i <= m; i++) {
    dp[i] = new Array(n + 1).fill(0);
    dp[i][0] = i;
  }
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];               // match — no cost
      } else {
        dp[i][j] = 1 + Math.min(
          dp[i - 1][j],       // deletion
          dp[i][j - 1],       // insertion
          dp[i - 1][j - 1]    // substitution
        );
      }
    }
  }
  return dp[m][n];
}

/* ================================================================
   2. CANDIDATE GENERATION
   ================================================================ */

/**
 * Generate correction candidates for a misspelled word.
 * Candidates are all vocabulary words within edit distance ≤ maxDist.
 * Ranked by a combined score from the noisy channel model:
 *   score = freq(candidate) × (1 + bigram_score × 10) / dist²
 *
 * @param {string} word         - The misspelled word (lowercase)
 * @param {string|null} prevWord - Previous word (for bigram context)
 * @param {string|null} nextWord - Next word (optional future context)
 * @param {number} maxDist       - Maximum edit distance to consider (default 2)
 * @returns {Array<{word, dist, score, freq}>} Sorted candidate list
 */
function getCandidates(word, prevWord = null, nextWord = null, maxDist = 2) {
  const w = word.toLowerCase();
  const candidates = [];

  for (const vocabWord of VOCAB) {
    const dist = levenshtein(w, vocabWord);
    if (dist === 0 || dist > maxDist) continue;

    const freq     = WORD_FREQ[vocabWord] || 1;
    const bgScore  = prevWord ? getBigramProb(prevWord, vocabWord) : 0;

    // Noisy channel model score
    const score = freq * (1 + bgScore * 10) / (dist * dist);

    candidates.push({ word: vocabWord, dist, score, freq });
  }

  // Sort: primary = score descending; secondary = dist ascending
  candidates.sort((a, b) => b.score - a.score || a.dist - b.dist);
  return candidates.slice(0, 8); // return top-8 suggestions
}

/* ================================================================
   3. TOKENISATION
   ================================================================ */

/**
 * Tokenise text into alternating word / non-word tokens.
 * Preserves punctuation, spaces, and newlines as separate tokens
 * so the original text can be reconstructed exactly.
 *
 * @param {string} text
 * @returns {string[]} Token array
 */
function tokenize(text) {
  return text.match(/[a-zA-Z''-]+|[^a-zA-Z''-]+/g) || [];
}

/* ================================================================
   4. SPELL CHECKING
   ================================================================ */

/**
 * Run the full spell-check pipeline on an input text.
 *
 * Returns an object mapping token index → error descriptor.
 * Each error descriptor: { type: 'nonword'|'realword', word, candidates }
 *
 * @param {string} text - Input text
 * @returns {{ tokens: string[], errors: Object }} Result
 */
function spellCheck(text) {
  const tokens    = tokenize(text);
  const errors    = {};

  // Collect only word tokens with their indices
  const wordTokens = [];
  tokens.forEach((tok, i) => {
    if (/[a-zA-Z]/.test(tok)) wordTokens.push({ tok, i });
  });

  wordTokens.forEach(({ tok, i }, wi) => {
    // Strip leading/trailing punctuation for lookup
    const w = tok.replace(/^[^a-zA-Z]+|[^a-zA-Z]+$/g, '').toLowerCase();
    if (!w || w.length < 2) return;

    const prevWord = wi > 0 ? wordTokens[wi - 1].tok.toLowerCase() : null;
    const nextWord = wi < wordTokens.length - 1 ? wordTokens[wi + 1].tok.toLowerCase() : null;

    /* ---- Non-word detection ---- */
    if (!VOCAB.has(w)) {
      const candidates = getCandidates(w, prevWord, nextWord);
      errors[i] = { type: 'nonword', word: tok, candidates };
      return;
    }

    /* ---- Real-word detection (context-sensitive) ---- */
    const confusable = REAL_WORD_CONFUSABLES[w];
    if (confusable && VOCAB.has(confusable)) {
      const probSelf = prevWord ? getBigramProb(prevWord, w)          : 0.05;
      const probAlt  = prevWord ? getBigramProb(prevWord, confusable) : 0.10;

      // Flag if the alternative fits the context significantly better
      if (probAlt > probSelf * 1.5) {
        const candidates = getCandidates(w, prevWord, nextWord);
        // Ensure the most likely confusable is in the list
        const alreadyIncluded = candidates.some(c => c.word === confusable);
        if (!alreadyIncluded) {
          candidates.unshift({
            word  : confusable,
            dist  : levenshtein(w, confusable),
            score : WORD_FREQ[confusable] || 5,
            freq  : WORD_FREQ[confusable] || 5,
          });
        }
        errors[i] = { type: 'realword', word: tok, candidates };
      }
    }
  });

  return { tokens, errors };
}

/* ================================================================
   5. UTILITIES
   ================================================================ */

/**
 * Escape HTML special characters.
 * @param {string} s
 * @returns {string}
 */
function escHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Build a flat sorted array of bigrams for the UI panel.
 * @returns {Array<{pair, a, b, prob, cnt}>}
 */
function buildBigramTable() {
  const rows = [];
  Object.entries(BIGRAMS).forEach(([a, bMap]) => {
    const total = Object.values(bMap).reduce((s, v) => s + v, 0) || 1;
    Object.entries(bMap).forEach(([b, cnt]) => {
      rows.push({ pair: `${a} ${b}`, a, b, prob: (cnt / total).toFixed(4), cnt });
    });
  });
  rows.sort((a, b) => parseFloat(b.prob) - parseFloat(a.prob));
  return rows;
}
