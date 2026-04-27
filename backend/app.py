"""
app.py
======
Flask REST API server for the Spelling Correction System.

Endpoints
---------
POST /api/check          – spell-check a piece of text
GET  /api/corpus/words   – sorted vocabulary list (with optional ?q= search)
GET  /api/corpus/bigrams – top bigrams (with optional ?q= filter)
GET  /api/status         – server health / model status

Start the server
----------------
    python app.py

The server listens on http://localhost:5000 by default.
The frontend (index.html) should be opened in a browser; it talks to this API.
"""

import logging
import os
import json

from flask import Flask, jsonify, request
from flask_cors import CORS

from nlp_engine import NLPEngine, levenshtein

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)   # allow requests from the frontend running on file:// or a different port

# ---------------------------------------------------------------------------
# Initialise NLP engine (done once at startup, not per-request)
# ---------------------------------------------------------------------------
log.info("Starting NLP engine initialisation ...")
engine = NLPEngine()
log.info("Server ready.")

# ---------------------------------------------------------------------------
# Helper: load corpus data for /api/corpus/* endpoints
# ---------------------------------------------------------------------------
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
FREQ_PATH = os.path.join(DATA_DIR, "frequency_dict.txt")
BG_PATH   = os.path.join(DATA_DIR, "bigrams.json")


def _load_freq_list():
    """Return sorted list of (word, count) tuples from frequency_dict.txt."""
    words = []
    if os.path.exists(FREQ_PATH):
        with open(FREQ_PATH, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split("\t")
                if len(parts) == 2:
                    try:
                        words.append((parts[0], int(parts[1])))
                    except ValueError:
                        pass
        words.sort(key=lambda x: -x[1])
    return words


def _load_bigrams():
    """Return sorted list of bigram dicts from bigrams.json."""
    if not os.path.exists(BG_PATH):
        return []
    with open(BG_PATH, encoding="utf-8") as f:
        raw: dict = json.load(f)

    rows = []
    for pair, count in raw.items():
        parts = pair.split(" ", 1)
        if len(parts) == 2:
            rows.append({"pair": pair, "a": parts[0], "b": parts[1], "count": count})

    # Sort by count descending
    rows.sort(key=lambda x: -x["count"])
    return rows


# Cache corpus data in memory (loaded lazily on first request)
_freq_cache   = None
_bigram_cache = None


def get_freq_list():
    global _freq_cache
    if _freq_cache is None:
        _freq_cache = _load_freq_list()
    return _freq_cache


def get_bigram_list():
    global _bigram_cache
    if _bigram_cache is None:
        _bigram_cache = _load_bigrams()
    return _bigram_cache


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def status():
    """Health check — returns model availability flags."""
    return jsonify({
        "ok"       : True,
    })


@app.route("/api/check", methods=["POST"])
def check_spelling():
    """
    Spell-check the submitted text.

    Request body (JSON):
        { "text": "The pashent has diabetis ..." }

    Response (JSON):
        {
            "tokens"  : ["The", " ", "pashent", " ", ...],
            "errors"  : [
                {
                    "token_index" : 2,
                    "word"        : "pashent",
                    "clean"       : "pashent",
                    "type"        : "nonword",
                    "candidates"  : [
                        { "word": "patient", "edit_distance": 2, "frequency": 8432,
                          "bert_score": -1.23, "score": 4.56 },
                        ...
                    ]
                },
                ...
            ],
            "stats"   : { "total": 12, "nonword": 2, "realword": 1, "correct": 9 },
            "bert_on" : true
        }
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text is empty."}), 400

    if len(text) > 5000:
        return jsonify({"error": "Text exceeds 5 000 character limit."}), 400

    # Extract the ignored indices array from the JavaScript ---
    # If it's not in the request, default to an empty list []
    ignored_indices = data.get("ignored_indices", [])

    try:
        result = engine.check(text, ignored_indices=ignored_indices)
        return jsonify(result)
    except Exception as exc:
        log.exception("Error during spell-check: %s", exc)
        return jsonify({"error": "Internal server error.", "detail": str(exc)}), 500


@app.route("/api/corpus/words", methods=["GET"])
def corpus_words():
    """
    Return corpus vocabulary sorted by frequency.

    Query params:
        q      (str)  – filter words containing this substring
        limit  (int)  – max results (default 200)
        offset (int)  – pagination offset (default 0)
    """
    q      = request.args.get("q", "").strip().lower()
    limit  = min(int(request.args.get("limit",  200)), 1000)
    offset = max(int(request.args.get("offset", 0)),   0)

    words = get_freq_list()

    if q:
        words = [(w, c) for w, c in words if q in w]

    total   = len(words)
    page    = words[offset: offset + limit]

    return jsonify({
        "total"  : total,
        "offset" : offset,
        "limit"  : limit,
        "words"  : [{"word": w, "frequency": c} for w, c in page],
    })


@app.route("/api/corpus/bigrams", methods=["GET"])
def corpus_bigrams():
    """
    Return top bigrams from the corpus.

    Query params:
        q      (str)  – filter bigrams containing this substring
        limit  (int)  – max results (default 100)
        offset (int)  – pagination offset (default 0)
    """
    q      = request.args.get("q", "").strip().lower()
    limit  = min(int(request.args.get("limit",  100)), 500)
    offset = max(int(request.args.get("offset", 0)),   0)

    bigrams = get_bigram_list()

    if q:
        bigrams = [b for b in bigrams if q in b["pair"]]

    total = len(bigrams)
    page  = bigrams[offset: offset + limit]

    return jsonify({
        "total"   : total,
        "offset"  : offset,
        "limit"   : limit,
        "bigrams" : page,
    })


@app.route("/api/edit-distance", methods=["POST"])
def edit_distance():
    """
    Compute Levenshtein distance between two words.

    Request body (JSON):
        { "a": "pashent", "b": "patient" }

    Response:
        { "a": "pashent", "b": "patient", "distance": 2 }
    """
    data = request.get_json(silent=True)
    if not data or "a" not in data or "b" not in data:
        return jsonify({"error": "Provide both 'a' and 'b' fields."}), 400

    a = data["a"].strip().lower()
    b = data["b"].strip().lower()
    return jsonify({"a": a, "b": b, "distance": levenshtein(a, b)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
