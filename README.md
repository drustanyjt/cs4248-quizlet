# CS4248 NLP Quizlet

A two-mode exam-prep web app for NUS CS4248 (Natural Language Processing).

- **Drill tab** — keyboard-driven MCQ/MRQ/FITB deck for active recall.
- **Learn tab** — tutorial chunks per lecture, with comprehension MCQs and an "Ask Claude" Q&A box.

Pure-stdlib Python server (no dependencies) + single-page HTML/JS frontend.

---

## Run

```bash
python3 server.py
```

Then open <http://127.0.0.1:8001/> in a browser.

Default port is `8001`. Override with `--port` / `--host`:

```bash
python3 server.py --port 9000 --host 0.0.0.0
```

The page renders math via MathJax (CDN) and markdown via marked.js (CDN), so an internet connection is needed for those to load — but the app works offline-ish if you've cached them.

---

## Tabs / URLs

| URL | Shows |
|---|---|
| `#drill` (default) | Keyboard quiz |
| `#drill` + `?lecture=N` | Quiz filtered to a single lecture |
| `#learn` | 12-lecture grid |
| `#learn/L7` | Chunk list for Lecture 7 |
| `#learn/L7/L7-3` | Specific chunk |

### Drill keyboard controls

- `1`–`9` — select option (MCQ replaces, MRQ toggles)
- `Enter` — submit / advance
- `Tab` — skip current (counts as wrong)
- `S` — skip remaining
- `R` — restart after deck is done

---

## Q&A workflow ("Ask Claude")

The app has no API key built in. The Ask button just persists your question to disk:

1. Click **💬 Ask Claude about this chunk** on any Learn-tab chunk.
2. Type your question, submit. It's saved to `discussions.json` with `status: pending`.
3. In your Claude Code session, say something like *"answer my pending discussions"*. Claude reads the file, fills in answers, sets `status: answered`.
4. Refresh the chunk page — answers appear inline under the chunk.

This means Q&A only works while you have a Claude Code session pointed at the project root. The chunk-reading and MCQ-checking parts work standalone.

---

## File map

| File / dir | What it is |
|---|---|
| `server.py` | stdlib HTTP server |
| `index.html` | single-page UI for both tabs |
| `questions.json` | drill deck (~111 MRQ/MCQ/FITB questions) |
| `chunks/lecture*.json` | Learn-tab content (58 chunks across 12 lectures, 166 MCQs) |
| `study_guides/lecture*.md` | skim-readable study guides per lecture (source of the chunks) |
| `results.json` | (gitignored) drill attempt history |
| `discussions.json` | (gitignored) your Q&A with Claude |
| `WEAK_AREAS.md` | (gitignored) auto-generated drill weak-area snapshot |

`results.json` and `discussions.json` are per-device — they're in `.gitignore` so each device has its own progress.

---

## Multi-device usage

Each device starts with a clean slate (no drill history, no past Q&A). The app state is local-only. To resume work:

1. `git pull` on the new device
2. `python3 server.py`
3. Open the URL, switch to the Learn tab, pick up where you mentally left off

---

## Adding content

Drill questions go into `questions.json` (schema: `{id, type, lecture, topic, difficulty, q, options/correct/explanation}`).

Learn chunks go into `chunks/lecture<NN>.json` as a JSON array of objects with shape:

```json
{
  "lecture": "5",
  "chunk_id": "L5-1",
  "order": 1,
  "title": "...",
  "content_md": "<markdown body>",
  "checks": [
    {"q": "...", "options": [{"id":"1","text":"..."}], "correct": "2", "explanation": "..."}
  ]
}
```

The server merges all `chunks/lecture*.json` on each `/chunks` request, so you can drop in new files without restarting.
