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

    // BERT badge
    const bertBadge = document.getElementById("bertBadge");
    bertBadge.style.display = "";
    if (data.bert_on) {
      bertBadge.textContent = "BERT ON";
      bertBadge.className   = "badge bert-on";
    } else {
      bertBadge.textContent = "BERT OFF (bigram fallback)";
      bertBadge.className   = "badge bert-off";
    }

    // Vocab badge
    const vocabBadge = document.getElementById("vocabBadge");
    vocabBadge.style.display = "";
    vocabBadge.textContent   = `${Number(data.vocab).toLocaleString()} words`;
    vocabBadge.className     = "badge vocab";

  } catch {
    el.textContent = "Server offline — start app.py";
    el.className   = "server-status err";
  }
}

// ── Editor helpers ────────────────────────────────────────────
function updateCount() {
  const len = document.getElementById("editor").value.length;
  const el  = document.getElementById("charCount");
  el.textContent = `${len} / 500`;
  el.className   = len > 470 ? "char-count warn" : "char-count";
}

function loadSample() {
  document.getElementById("editor").value =
    "The pashent has diabetis and hypertenshon. The doctar prescribed " +
    "medcine for the conditon. Liver disease can affect the kidny. " +
    "Blood presure was carefully monitered by the specalist.";
  updateCount();
}

function clearAll() {
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
      body    : JSON.stringify({ text }),
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

  let html = '<p class="hint" style="margin-bottom:4px">Ranked by edit distance · corpus frequency · BERT context score:</p>';
  html += '<div class="sugg-list">';
  e.candidates.forEach(c => {
    const safeWord = escHtml(c.word);
    html +=
      `<span class="sugg-chip" onclick="applySuggestion(${tokenIdx},'${safeWord}')">` +
      `${safeWord}` +
      `<span class="meta">d=${c.edit_distance}</span>` +
      `</span>`;
  });
  html += "</div>";
  box.innerHTML = html;

  // Highlight the selected token
  document.querySelectorAll(".err-nonword, .err-realword").forEach(el => el.style.outline = "");
  document.querySelectorAll(`[onclick="showSuggestions(${tokenIdx})"]`)
    .forEach(el => el.style.outline = "2px solid #185FA5");
}

// ── Apply a chosen suggestion ─────────────────────────────────
function applySuggestion(tokenIdx, newWord) {
  currentTokens[tokenIdx] = newWord;
  delete currentErrors[tokenIdx];

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
      `${data.total.toLocaleString()} words in corpus`;

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
      `${data.total.toLocaleString()} bigrams`;

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
