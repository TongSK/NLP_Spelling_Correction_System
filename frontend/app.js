/**
 * app.js  —  Frontend controller for the Spelling Correction System.
 *
 * All NLP work is done by the Python/Flask backend (http://localhost:5000).
 * This file handles:
 *   - Calling the REST API
 *   - Rendering highlighted text, suggestions, corpus word list, bigrams
 *   - UI interactions (tab switching, applying suggestions, pagination)
 */

"use strict";

// ── Configuration ────────────────────────────────────────────
const API_BASE = "http://localhost:5000/api";

// ── State ────────────────────────────────────────────────────
let currentTokens = [];    // raw token array from last /api/check response
let currentErrors  = {};   // { token_index: errorDescriptor }

let ignoredIndices = [];   // remembers which words the user accepted or ignored

let wordOffset    = 0;
let wordQuery     = "";
let bigramQuery   = "";

// ── Initialisation ───────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  checkServerStatus();
  loadCorpusWords();
  loadBigrams();
  loadSample();
});

// ── Server health check ───────────────────────────────────────
async function checkServerStatus() {
  const el = document.getElementById("serverStatus");
  try {
    const res  = await fetch(`${API_BASE}/status`);
    const data = await res.json();

    el.textContent = "Server connected";
    el.className   = "server-status ok";
  } catch {
    el.textContent = "Server offline — start app.py";
    el.className   = "server-status err";
  }
}

// ── Editor helpers ────────────────────────────────────────────
/*
function updateCount() {
  const len = document.getElementById("editor").value.length;
  const el  = document.getElementById("charCount");
  el.textContent = `${len} / 500`;
  el.className   = len > 470 ? "char-count warn" : "char-count";
}
  */
 // ── Editor helpers ────────────────────────────────────────────
function updateCount() {
  const editor = document.getElementById("editor");
  const text = editor.value;
  const len = text.length;
  const el  = document.getElementById("charCount");
  el.textContent = `${len} / 500`;
  el.className   = len > 470 ? "char-count warn" : "char-count";
  
  const cursorPos = editor.selectionStart;

  const textBeforeCursor = text.substring(0, cursorPos);

  // Count how many valid words are in that text.
  // We use a regex that matches the Python backend to count the words.
  const wordsBeforeCursor = (textBeforeCursor.match(/[a-zA-Z]+/g) || []).length;

  // The user is currently editing at (or just after) this word index.
  // Because they are typing here, any word AFTER this point might shift its position.
  // We calculate a "Safe Limit" (subtracting 1 to be safe about the current word being edited).
  const safeIndexLimit = Math.max(0, wordsBeforeCursor - 1);

  // 5. Filter the ignore list! Keep only the indices that are safely BEFORE the edit.
  ignoredIndices = ignoredIndices.filter(index => index < safeIndexLimit);
}


// ── Editor helpers ────────────────────────────────────────────
function loadSample() {
  ignoredIndices = [];
  
  // A curated list of 10 clinical scenarios.
  const sampleParagraphs = [
    "A pashent arrived at the clinic complaining of severe head paint in these few days. The doctar examined his spiral cord but found zero abnormalities.",
    
    "She experienced muscle weekness in her left led for several days. The ultrasound revealed a blocked vain of the knee joint. The surgeon will operate to restore normal blood flow. She requires immediate medcine to prevent clotting.",
    
    "The surgeon successfully clamped the lacerated artary during the operation. This quick action prevented any further lose of blood. The anesthesiologist monitored her hert rate carefully. It will take a long coarse to recover.",
    
    "A specalist recommended physical straining after the plaster was removed. It simply cannot support the wait of the arm right now. The fractured coller bone requires a specialized sling to heal properly. Rehabilitation is a slow root to recovery.",
    
    "My friend felt a sudden weekness in her left site and the doctar ordered an Magnetic Resonance Imaging to check for a brain bleed. The results showed a small clott near the frontal lobe. It will by a long course of treatment.",

    "Damage to the optick nerve can lead to permanent visual impairment. The patient reported a sudden loss of site in their right eye. They could knot see clearly in dim light. Urgent examinatns is required immediately.",
    
    "Her attending physcian noted that the fraxture was quite complicated. The bone was broken completely away from its bass. It will require surgery add extensive rehabilitation. The patient must rest for six weeks.",
    
    "My child's sudden feaver was accompanied by chills and sweating. Pediatricians an the nurse administered fluids intravenously to manage the severity if this illness. They monitored his temperture every hour. His breathing sounded like a hoarse whisper.",
    
    "The nurse recorded the vitals an the existence of chest paint or other problems.  Doctars will check the patient's Electrocardiogram results shortly. There is a small clott in the left ventricle. We must monitor patients closely tonight.",
    
    "She does have a weekness in the valves of her left vain. The blood pressure is too high an needs medication. Therefore, the presciption must be taken twice daily."
  ];

  // Pick a random number between 0 and 9
  const randomIndex = Math.floor(Math.random() * sampleParagraphs.length);

  // Set the editor value to the randomly selected paragraph
  document.getElementById("editor").value = sampleParagraphs[randomIndex];
  
  updateCount();
}

function clearAll() {
  ignoredIndices = [];
  document.getElementById("editor").value = "";
  document.getElementById("highlighted").innerHTML =
    '<span class="placeholder">Highlighted text will appear here after checking...</span>';
  document.getElementById("suggestionBox").innerHTML =
    '<p class="hint">Click a highlighted word above to see correction suggestions.</p>';
  document.getElementById("activeWordLabel").textContent = "";
  document.getElementById("statsRow").style.display = "none";
  updateCount();
  currentTokens = [];
  currentErrors = {};
}

// ── Main spell-check call ─────────────────────────────────────
async function runCheck() {
  const text = document.getElementById("editor").value.trim();
  if (!text) { alert("Please enter some text first."); return; }

  // Show spinner
  document.getElementById("spinner").style.display = "";
  document.getElementById("checkBtn").disabled = true;

  try {
    const res = await fetch(`${API_BASE}/check`, {
      method  : "POST",
      headers : { "Content-Type": "application/json" },
      body    : JSON.stringify({ 
        text: text,
        ignored_indices: ignoredIndices
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Server error: ${err.error || res.statusText}`);
      return;
    }

    const data = await res.json();
    processResult(data);

  } catch (e) {
    alert("Could not reach the server. Make sure app.py is running.");
    console.error(e);
  } finally {
    document.getElementById("spinner").style.display = "none";
    document.getElementById("checkBtn").disabled = false;
  }
}

function processResult(data) {
  currentTokens = data.tokens || [];
  currentErrors = {};

  (data.errors || []).forEach(e => {
    currentErrors[e.token_index] = e;
  });

  const s = data.stats || {};
  document.getElementById("sTotal").textContent   = s.total   || 0;
  document.getElementById("sNonword").textContent = s.nonword || 0;
  document.getElementById("sRealword").textContent= s.realword|| 0;
  document.getElementById("sOk").textContent      = s.correct || 0;
  document.getElementById("statsRow").style.display = "grid";

  renderHighlighted();

  document.getElementById("suggestionBox").innerHTML =
    '<p class="hint">Click a highlighted word above to see correction suggestions.</p>';
  document.getElementById("activeWordLabel").textContent = "";
}

// ── Render highlighted text ───────────────────────────────────
function renderHighlighted() {
  const container = document.getElementById("highlighted");
  if (!currentTokens.length) {
    container.innerHTML = '<span class="placeholder">No text to display.</span>';
    return;
  }

  let html = "";
  currentTokens.forEach((tok, i) => {
    const e = currentErrors[i];
    if (e) {
      const cls   = e.type === "nonword" ? "err-nonword" : "err-realword";
      const title = e.type === "nonword"
        ? "Non-word error — click for suggestions"
        : "Real-word (context) error — click for suggestions";
      html += `<span class="${cls}" onclick="showSuggestions(${i})" title="${title}">${escHtml(tok)}</span>`;
    } else {
      html += escHtml(tok);
    }
  });
  container.innerHTML = html;
}

// ── Show suggestions for a clicked error ─────────────────────
function showSuggestions(tokenIdx) {
  const e = currentErrors[tokenIdx];
  if (!e) return;

  const box   = document.getElementById("suggestionBox");
  const label = document.getElementById("activeWordLabel");

  const typeTag = e.type === "nonword"
    ? '<span class="err-tag tag-nw">non-word</span>'
    : '<span class="err-tag tag-rw">real-word</span>';
  label.innerHTML = `"${escHtml(e.word)}" ${typeTag}`;

  if (!e.candidates || e.candidates.length === 0) {
    box.innerHTML = '<p class="hint">No close matches found in corpus vocabulary.</p>';
    return;
  }

  let html = '<p class="hint" style="margin-bottom:4px">Ranked by edit distance · corpus frequency · bidirectional bigram score:</p>';
  html += '<div class="sugg-list">';

  const textBefore = currentTokens.slice(0, tokenIdx).join("").trim();

  const isLeadingWord = textBefore === "" || /[.!?]$/.test(textBefore);

  e.candidates.forEach(c => {
    let finalWord = c.word;

    if (isLeadingWord) {
      finalWord = finalWord.charAt(0).toUpperCase() + finalWord.slice(1);
    }

    const safeWord = escHtml(finalWord);
    html +=
      `<span class="sugg-chip" onclick="applySuggestion(${tokenIdx},'${safeWord}')">` +
      `${safeWord}` +
      `<span class="meta">d=${c.edit_distance}</span>` +
      `</span>`;
  });
  html += "</div>";

  html += `<div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center;">` +
            `<span class="hint" style="margin:0;">Word spelled correctly?</span>` +
            `<button style="background: transparent; color: #64748b; border: 1px solid #cbd5e1; padding: 4px 10px; font-size: 0.85rem; border-radius: 4px; cursor: pointer; transition: all 0.2s;" ` +
            `onmouseover="this.style.background='#f8fafc'; this.style.color='#0f172a';" ` +
            `onmouseout="this.style.background='transparent'; this.style.color='#64748b';" ` +
            `onclick="ignoreError(${tokenIdx})">` +
            `&#10006; Ignore Error</button>` +
          `</div>`;

  box.innerHTML = html;

  // Highlight the selected token
  document.querySelectorAll(".err-nonword, .err-realword").forEach(el => el.style.outline = "");
  document.querySelectorAll(`[onclick="showSuggestions(${tokenIdx})"]`)
    .forEach(el => el.style.outline = "2px solid #185FA5");
}

// ── Ignore an error without changing the text ─────────────────
function ignoreError(tokenIdx) {
  const errorToIgnore = currentErrors[tokenIdx];
  if (!errorToIgnore) return;

  const totalEl = document.getElementById("sTotal");
  const okEl = document.getElementById("sOk");

  if (totalEl) totalEl.textContent = Math.max(0, parseInt(totalEl.textContent) - 1);
  if (okEl) okEl.textContent = parseInt(okEl.textContent) + 1;

  if (errorToIgnore.type === "nonword") {
    const nwEl = document.getElementById("sNonword");
    if (nwEl) nwEl.textContent = Math.max(0, parseInt(nwEl.textContent) - 1);
  } else {
    const rwEl = document.getElementById("sRealword");
    if (rwEl) rwEl.textContent = Math.max(0, parseInt(rwEl.textContent) - 1);
  }

  if (!ignoredIndices.includes(tokenIdx)) {
    ignoredIndices.push(tokenIdx);
  }
  
  delete currentErrors[tokenIdx];
  
  renderHighlighted();
  
  document.getElementById("suggestionBox").innerHTML = '<p class="hint">Error ignored and statistics updated.</p>';
  document.getElementById("activeWordLabel").textContent = "";
}

function applySuggestion(tokenIdx, newWord) {
  currentTokens[tokenIdx] = newWord;
  delete currentErrors[tokenIdx];

  if (!ignoredIndices.includes(tokenIdx)) {
    ignoredIndices.push(tokenIdx);
  }

  document.getElementById("editor").value = currentTokens.join("");
  updateCount();
  runCheck();
}

// ── Tab switching ─────────────────────────────────────────────
function switchTab(name) {
  ["corpus", "bigrams", "methods"].forEach(t => {
    document.getElementById("side-" + t).style.display  = t === name ? "flex" : "none";
    document.getElementById("tab-" + t).classList.toggle("active", t === name);
  });
}

// ── Corpus word list ──────────────────────────────────────────
async function loadCorpusWords(reset = true) {
  if (reset) wordOffset = 0;
  const url = `${API_BASE}/corpus/words?limit=100&offset=${wordOffset}&q=${encodeURIComponent(wordQuery)}`;
  try {
    const res  = await fetch(url);
    const data = await res.json();

    document.getElementById("wordCountLabel").textContent =
      `${data.total.toLocaleString()} words available in corpus`;

    const ul = document.getElementById("wordList");
    if (reset) ul.innerHTML = "";

    data.words.forEach(({ word, frequency }) => {
      const li = document.createElement("li");
      li.className = "word-item";
      li.innerHTML = `<span>${escHtml(word)}</span><span class="word-freq">${frequency.toLocaleString()}</span>`;
      ul.appendChild(li);
    });

    wordOffset += data.words.length;
    const loadMore = document.getElementById("wordLoadMore");
    loadMore.style.display = wordOffset < data.total ? "" : "none";

  } catch {
    document.getElementById("wordCountLabel").textContent = "Corpus unavailable — start app.py";
  }
}

function loadMoreWords() {
  loadCorpusWords(false);
}

function filterWords(q) {
  wordQuery = q;
  loadCorpusWords(true);
}

// ── Bigrams ───────────────────────────────────────────────────
async function loadBigrams(reset = true) {
  const url = `${API_BASE}/corpus/bigrams?limit=100&q=${encodeURIComponent(bigramQuery)}`;
  try {
    const res  = await fetch(url);
    const data = await res.json();

    document.getElementById("bigramCountLabel").textContent =
      `${data.total.toLocaleString()} available bigrams`;

    const maxCount = data.bigrams.length ? data.bigrams[0].count : 1;
    const div = document.getElementById("bigramList");
    div.innerHTML = data.bigrams.map(b => {
      const pct = Math.round((b.count / maxCount) * 100);
      return `<div class="bigram-row">
                <span>${escHtml(b.pair)}</span>
                <div class="bigram-meta">
                  <div class="bigram-count">${b.count.toLocaleString()}</div>
                  <div class="prob-bar-wrap">
                    <div class="prob-bar" style="width:${pct}%"></div>
                  </div>
                </div>
              </div>`;
    }).join("");

  } catch {
    document.getElementById("bigramCountLabel").textContent = "Unavailable — start app.py";
  }
}

function filterBigrams(q) {
  bigramQuery = q;
  loadBigrams(true);
}

// ── Utility ───────────────────────────────────────────────────
function escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
