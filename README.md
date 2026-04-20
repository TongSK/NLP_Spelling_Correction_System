# Probabilistic Spelling Correction System

A browser-based NLP spelling corrector built on a **medical science corpus** (~100,000+ word tokens).
Detects both **non-word errors** and **real-word (context) errors** using bigrams, Levenshtein edit
distance, and the noisy channel model.

---

## File Structure

```
spelling_corrector/
├── index.html    — Main HTML page and layout
├── style.css     — Full stylesheet (light + responsive)
├── corpus.js     — Medical corpus, vocabulary, bigram model
├── nlp.js        — Core NLP engine (edit distance, candidate generation, spell-check pipeline)
├── app.js        — UI controller (rendering, events, tab panels)
└── README.md     — This file
```

---

## How to Run

1. Open `index.html` directly in any modern browser — no server or build step required.
2. Type or paste medical text into the editor (up to 500 characters).
3. Click **Check Spelling** to analyse the text.

---

## NLP Techniques

### 1. Non-word Detection
Any token not found in the corpus vocabulary is flagged as a non-word error (red underline).

### 2. Real-word Detection (Context-sensitive)
Words that exist in the vocabulary but are used in the wrong context are flagged as real-word
errors (amber underline). Detection uses:
- A curated confusable-pairs table (e.g. `dose` / `does`, `affect` / `effect`)
- Bigram probability comparison: if P(correct_word | prev_word) >> P(typed_word | prev_word),
  the word is flagged.

### 3. Minimum Edit Distance — Levenshtein
Dynamic programming computes the minimum number of single-character operations (insert, delete,
substitute) to transform the misspelled word into each vocabulary candidate.

```
levenshtein("pashent", "patient") = 2   (sh→ti, swap)
levenshtein("diabetis", "diabetes") = 1  (insertion)
```

Only candidates with distance ≤ 2 are shown as suggestions.

### 4. Noisy Channel Model
Candidates are scored by:

```
score = P(candidate) × P(error | candidate) / distance²
      ≈ corpus_frequency × (1 + bigram_context_score × 10) / dist²
```

This combines:
- **Language model prior** — how common is the candidate word?
- **Error model** — how likely is this typo given this word?
- **Context score** — does the candidate fit the bigram context?

### 5. Bigram Language Model
Co-occurrence counts P(w_i | w_{i-1}) are estimated from the corpus with add-0.5 smoothing
(Laplace-like). Used for:
- Real-word error detection (comparing bigram probabilities)
- Re-ranking correction candidates by contextual fit

---

## Corpus

Domain: **Medical Science** (~100,000+ word tokens, ~500 unique vocabulary types)

Coverage:
- Clinical medicine (diagnosis, treatment, symptoms)
- Pharmacology (drugs, dosage, pharmacokinetics)
- Pathology (histology, biopsy, necrosis)
- Anatomy and physiology (organs, cells, tissues)
- Immunology (antibodies, cytokines, inflammation)
- Genetics and molecular biology (gene, mutation, expression)
- Epidemiology (incidence, prevalence, outbreak)

---

## GUI Features

| Feature | Description |
|---|---|
| Text editor | 500-character editor with live counter |
| Spell check | Analyses text on demand |
| Stats bar | Shows total words, non-word errors, real-word errors, correct count |
| Highlighted view | Red = non-word, Amber = real-word; click any error for suggestions |
| Suggestion chips | Sorted by edit distance + frequency + context; click to apply |
| Corpus panel | Sorted vocabulary list with frequency; searchable |
| Bigrams panel | Top bigram pairs with probability bars; searchable |
| Methods panel | Explanation of each NLP technique used |

---

## Browser Compatibility

Tested on Chrome 120+, Firefox 121+, Safari 17+, Edge 120+.
No dependencies — pure HTML/CSS/JavaScript.
