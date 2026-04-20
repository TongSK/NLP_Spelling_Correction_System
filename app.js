/**
 * app.js
 * Application controller for the Spelling Correction System.
 * Handles UI events, rendering, and state management.
 *
 * Depends on: corpus.js, nlp.js
 */

/* ================================================================
   STATE
   ================================================================ */

let currentTokens = [];   // Tokenised version of the editor text
let currentErrors = {};   // { tokenIndex: errorDescriptor }

/* ================================================================
   INITIALISATION
   ================================================================ */

(function init() {
  renderWordList();
  renderBigramList();
  // Pre-load a sample text so the user can try immediately
  loadSample();
})();

/* ================================================================
   EDITOR
   ================================================================ */

/**
 * Update the character counter below the editor.
 */
function updateCount() {
  const len = document.getElementById('editor').value.length;
  const el  = document.getElementById('charCount');
  el.textContent = `${len} / 500`;
  el.style.color = len > 470 ? 'var(--danger)' : 'var(--text-faint)';
}

/**
 * Load a sample medical text demonstrating both error types.
 */
function loadSample() {
  document.getElementById('editor').value =
    'The pashent has diabetis and hypertenshon. The doctar prescribed ' +
    'medcine for the conditon. Liver disease can affect the kidny. ' +
    'Blood presure was measured carefully. The dose of the drug was ' +
    'reccomended by the specalist.';
  updateCount();
}

/**
 * Clear the editor and all results.
 */
function clearAll() {
  document.getElementById('editor').value = '';
  document.getElementById('highlighted').innerHTML =
    '<span class="placeholder">Highlighted text will appear here after checking...</span>';
  document.getElementById('suggestionBox').innerHTML =
    '<div class="suggestion-header">Click a highlighted word to see correction suggestions with edit distance scores.</div>';
  document.getElementById('activeWordLabel').textContent = '';
  document.getElementById('statsRow').style.display = 'none';
  document.getElementById('charCount').textContent = '0 / 500';
  currentTokens = [];
  currentErrors = {};
}

/* ================================================================
   SPELL-CHECK PIPELINE
   ================================================================ */

/**
 * Run the spell-check pipeline and update the UI.
 */
function runCheck() {
  const text = document.getElementById('editor').value.trim();
  if (!text) { alert('Please enter some text first.'); return; }

  const result = spellCheck(text);
  currentTokens = result.tokens;
  currentErrors = result.errors;

  // Compute stats
  const wordTokens = currentTokens.filter(t => /[a-zA-Z]/.test(t));
  const total      = wordTokens.length;
  let nonword = 0, realword = 0;
  Object.values(currentErrors).forEach(e => {
    if (e.type === 'nonword')  nonword++;
    if (e.type === 'realword') realword++;
  });
  const ok = total - nonword - realword;

  // Update stats bar
  document.getElementById('sTotal').textContent   = total;
  document.getElementById('sNonword').textContent = nonword;
  document.getElementById('sRealword').textContent = realword;
  document.getElementById('sOk').textContent      = ok;
  document.getElementById('statsRow').style.display = 'grid';

  renderHighlighted();

  // Reset suggestion panel
  document.getElementById('suggestionBox').innerHTML =
    '<div class="suggestion-header">Click a highlighted word above to see correction suggestions.</div>';
  document.getElementById('activeWordLabel').textContent = '';
}

/* ================================================================
   RENDERING
   ================================================================ */

/**
 * Render the highlighted text view from currentTokens / currentErrors.
 */
function renderHighlighted() {
  const container = document.getElementById('highlighted');
  let html = '';

  currentTokens.forEach((tok, i) => {
    if (currentErrors[i]) {
      const e     = currentErrors[i];
      const cls   = e.type === 'nonword' ? 'err-nonword' : 'err-realword';
      const title = e.type === 'nonword' ? 'Non-word error — click for suggestions'
                                         : 'Real-word (context) error — click for suggestions';
      html += `<span class="${cls}" onclick="showSuggestions(${i})" title="${title}">${escHtml(tok)}</span>`;
    } else {
      html += escHtml(tok);
    }
  });

  container.innerHTML = html ||
    '<span class="placeholder">No text checked yet.</span>';
}

/**
 * Show correction suggestions for the clicked error token.
 * @param {number} tokenIdx - Index in currentTokens
 */
function showSuggestions(tokenIdx) {
  const e = currentErrors[tokenIdx];
  if (!e) return;

  const box   = document.getElementById('suggestionBox');
  const label = document.getElementById('activeWordLabel');

  // Label above suggestion box
  const typeTag = e.type === 'nonword'
    ? '<span class="err-tag tag-nw">non-word</span>'
    : '<span class="err-tag tag-rw">real-word</span>';
  label.innerHTML = `"${escHtml(e.word)}" ${typeTag}`;

  if (!e.candidates || e.candidates.length === 0) {
    box.innerHTML = '<div class="suggestion-header">No close matches found in corpus vocabulary.</div>';
    return;
  }

  let html = '<div class="suggestion-header">Ranked by edit distance · corpus frequency · bigram context score:</div>';
  html += '<div class="sugg-list">';
  e.candidates.forEach(c => {
    html += `<span class="sugg-chip" onclick="applySuggestion(${tokenIdx}, '${escHtml(c.word)}')">`
          + `${escHtml(c.word)}`
          + `<span class="dist">d=${c.dist}</span>`
          + `</span>`;
  });
  html += '</div>';

  box.innerHTML = html;

  // Highlight the clicked token
  document.querySelectorAll('.err-nonword, .err-realword').forEach(el => {
    el.style.outline = '';
  });
  const spans = document.querySelectorAll(`[onclick="showSuggestions(${tokenIdx})"]`);
  spans.forEach(s => s.style.outline = '2px solid var(--accent)');
}

/**
 * Apply a chosen suggestion: replace the token and re-run the checker.
 * @param {number} tokenIdx  - Token index to replace
 * @param {string} newWord   - Replacement word
 */
function applySuggestion(tokenIdx, newWord) {
  currentTokens[tokenIdx] = newWord;
  delete currentErrors[tokenIdx];

  // Sync the editor textarea
  document.getElementById('editor').value = currentTokens.join('');
  updateCount();

  // Re-run for fresh results
  runCheck();
}

/* ================================================================
   SIDE PANEL — TAB SWITCHING
   ================================================================ */

/**
 * Switch between Corpus / Bigrams / Methods tabs.
 * @param {string} name - 'corpus' | 'bigrams' | 'methods'
 */
function switchTab(name) {
  ['corpus', 'bigrams', 'methods'].forEach(t => {
    document.getElementById('side-' + t).style.display = t === name ? 'flex' : 'none';
    document.getElementById('tab-' + t).classList.toggle('active', t === name);
  });
}

/* ================================================================
   CORPUS WORD LIST
   ================================================================ */

let filteredWords = SORTED_WORDS;

/**
 * Filter the corpus word list by search query.
 * @param {string} query
 */
function filterWords(query) {
  const q = query.toLowerCase().trim();
  filteredWords = q
    ? SORTED_WORDS.filter(([w]) => w.includes(q))
    : SORTED_WORDS;
  renderWordList();
}

/**
 * Render the corpus word list (max 150 items for performance).
 */
function renderWordList() {
  const ul    = document.getElementById('wordList');
  const label = document.getElementById('wordCountLabel');

  label.textContent = `Showing ${Math.min(filteredWords.length, 150).toLocaleString()} of ${filteredWords.length.toLocaleString()} words`;

  ul.innerHTML = filteredWords.slice(0, 150).map(([w, f]) =>
    `<li class="word-item">
       <span>${escHtml(w)}</span>
       <span class="word-freq">${f.toLocaleString()}</span>
     </li>`
  ).join('');
}

/* ================================================================
   BIGRAM TABLE
   ================================================================ */

const ALL_BIGRAMS = buildBigramTable();
let filteredBigrams = ALL_BIGRAMS;

/**
 * Filter the bigram table by search query.
 * @param {string} query
 */
function filterBigrams(query) {
  const q = query.toLowerCase().trim();
  filteredBigrams = q
    ? ALL_BIGRAMS.filter(bg => bg.pair.includes(q))
    : ALL_BIGRAMS;
  renderBigramList();
}

/**
 * Render the bigram probability table (max 100 items).
 */
function renderBigramList() {
  const div   = document.getElementById('bigramList');
  const label = document.getElementById('bigramCountLabel');

  label.textContent = `${filteredBigrams.length.toLocaleString()} bigrams`;

  const maxProb = filteredBigrams.length
    ? parseFloat(filteredBigrams[0].prob)
    : 1;

  div.innerHTML = filteredBigrams.slice(0, 100).map(bg => {
    const pct = maxProb > 0
      ? Math.round((parseFloat(bg.prob) / maxProb) * 100)
      : 0;
    return `<div class="bigram-row">
              <span class="bigram-pair">${escHtml(bg.pair)}</span>
              <div>
                <div class="bigram-prob">${bg.prob}</div>
                <div class="prob-bar-wrap">
                  <div class="prob-bar" style="width:${pct}%"></div>
                </div>
              </div>
            </div>`;
  }).join('');
}
