#!/usr/bin/env python3
"""
CS4248 NLP Quizlet — keyboard-driven, stdlib only.

Endpoints:
  GET  /                  index.html
  GET  /questions         all drill questions
  GET  /results           past drill attempts
  POST /answer            record drill attempt
  GET  /stats             aggregate drill stats
  GET  /weak-areas        questions user got wrong
  GET  /chunks            all Learn-tab chunks (merged from chunks/*.json)
  GET  /discussions       all user clarifying questions + Claude's answers
  POST /discussions       submit a clarifying question (auto id, pending status)
  GET  /healthz           {ok}
"""
import argparse
import json
import sys
import threading
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).parent.resolve()
QUESTIONS_FILE = ROOT / "questions.json"
RESULTS_FILE = ROOT / "results.json"
INDEX_FILE = ROOT / "index.html"
WEAK_AREAS_MD = ROOT / "WEAK_AREAS.md"
CHUNKS_DIR = ROOT / "chunks"
DISCUSSIONS_FILE = ROOT / "discussions.json"

LOCK = threading.Lock()


def load_chunks():
    """Merge all chunks/lectureNN.json files into a single ordered list."""
    if not CHUNKS_DIR.exists():
        return []
    chunks = []
    for p in sorted(CHUNKS_DIR.glob("lecture*.json")):
        try:
            data = json.loads(p.read_text())
            if isinstance(data, list):
                chunks.extend(data)
        except Exception as e:
            sys.stderr.write(f"[chunks] skip {p.name}: {e}\n")
    chunks.sort(key=lambda c: (int(c.get("lecture", 99)), c.get("order", 99)))
    return chunks


def load_discussions():
    if DISCUSSIONS_FILE.exists():
        try:
            return json.loads(DISCUSSIONS_FILE.read_text())
        except Exception:
            return []
    return []


def save_discussions(d):
    DISCUSSIONS_FILE.write_text(json.dumps(d, indent=2))


def load_questions():
    return json.loads(QUESTIONS_FILE.read_text())


def load_results():
    if RESULTS_FILE.exists():
        try:
            return json.loads(RESULTS_FILE.read_text())
        except Exception:
            return []
    return []


def save_results(r):
    RESULTS_FILE.write_text(json.dumps(r, indent=2))
    refresh_weak_areas_md()


def refresh_weak_areas_md():
    """Generate a markdown summary of wrong answers for me to read between sessions."""
    qs = load_questions()
    qs_by_id = {q["id"]: q for q in qs}
    results = load_results()

    # Aggregate per question: latest verdict + count attempts
    latest = {}
    for r in results:
        qid = r["qid"]
        if qid not in latest:
            latest[qid] = {"attempts": 0, "wrong": 0, "right": 0, "user_answers": []}
        latest[qid]["attempts"] += 1
        if r["correct"]:
            latest[qid]["right"] += 1
        else:
            latest[qid]["wrong"] += 1
        latest[qid]["user_answers"].append(r.get("user_answer", ""))

    wrong_qs = [(qid, info) for qid, info in latest.items() if info["wrong"] > 0]
    wrong_qs.sort(key=lambda x: -x[1]["wrong"])  # most-wrong first

    # Aggregate by lecture and topic
    lec_stats = {}
    topic_stats = {}
    for qid, info in latest.items():
        q = qs_by_id.get(qid)
        if not q:
            continue
        lec = q.get("lecture", "?")
        top = q.get("topic", "?")
        for key, stats in [(lec, lec_stats), (top, topic_stats)]:
            if key not in stats:
                stats[key] = {"right": 0, "wrong": 0, "attempts": 0}
            stats[key]["attempts"] += info["attempts"]
            stats[key]["right"] += info["right"]
            stats[key]["wrong"] += info["wrong"]

    lines = [
        "# CS4248 Quizlet — weak areas snapshot",
        "",
        f"_Generated {datetime.now().isoformat(timespec='seconds')} from {RESULTS_FILE.name}._",
        "",
        f"**Total attempts:** {sum(i['attempts'] for i in latest.values())} across {len(latest)} unique questions.",
        f"**Wrong-at-least-once:** {len(wrong_qs)} questions.",
        "",
        "---",
        "",
        "## Per-lecture breakdown",
        "",
        "| Lecture | Right | Wrong | Total | Accuracy |",
        "|---|---|---|---|---|",
    ]
    for lec in sorted(lec_stats.keys(), key=lambda x: int(x) if str(x).isdigit() else 99):
        s = lec_stats[lec]
        acc = f"{100*s['right']/s['attempts']:.0f}%" if s['attempts'] else "—"
        lines.append(f"| L{lec} | {s['right']} | {s['wrong']} | {s['attempts']} | {acc} |")

    lines += [
        "",
        "## Per-topic breakdown",
        "",
        "| Topic | Right | Wrong | Total |",
        "|---|---|---|---|",
    ]
    for top in sorted(topic_stats.keys()):
        s = topic_stats[top]
        lines.append(f"| {top} | {s['right']} | {s['wrong']} | {s['attempts']} |")

    lines += ["", "## Wrong questions (most-missed first)", ""]
    if not wrong_qs:
        lines.append("_None yet._")
    else:
        for qid, info in wrong_qs[:30]:
            q = qs_by_id.get(qid)
            if not q:
                continue
            lines += [
                f"### `{qid}` — L{q.get('lecture','?')} / {q.get('topic','?')} ({q.get('type','?')}, {q.get('difficulty','?')})",
                f"- Wrong **{info['wrong']}** time(s), right {info['right']} time(s).",
                f"- Question: {strip_html(q['q'])[:300]}",
                f"- Correct answer: `{q.get('correct')}`",
                f"- User's most-recent attempts: {info['user_answers'][-3:]}",
                "",
            ]

    WEAK_AREAS_MD.write_text("\n".join(lines))


def strip_html(s):
    import re
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&nbsp;", " ")
    return s.strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{datetime.now().strftime('%H:%M:%S')}] {fmt % args}\n")

    def _send(self, status, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self):
        n = int(self.headers.get("content-length", "0"))
        if n == 0:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except Exception:
            return {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            return self._send(200, INDEX_FILE.read_text(), "text/html; charset=utf-8")
        if path == "/healthz":
            return self._send(200, {"ok": True})
        if path == "/questions":
            return self._send(200, load_questions())
        if path == "/results":
            return self._send(200, load_results())
        if path == "/weak-areas":
            results = load_results()
            wrong_ids = {r["qid"] for r in results if not r["correct"]}
            return self._send(200, list(wrong_ids))
        if path == "/stats":
            results = load_results()
            n = len(results)
            r = sum(1 for x in results if x["correct"])
            return self._send(200, {"total": n, "right": r, "wrong": n - r})
        if path == "/chunks":
            return self._send(200, load_chunks())
        if path == "/discussions":
            return self._send(200, load_discussions())
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        body = self._json_body()
        if self.path == "/answer":
            entry = {
                "qid": body.get("qid"),
                "user_answer": body.get("user_answer"),
                "correct": bool(body.get("correct")),
                "lecture": body.get("lecture"),
                "topic": body.get("topic"),
                "type": body.get("type"),
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            with LOCK:
                results = load_results()
                results.append(entry)
                save_results(results)
            return self._send(200, {"ok": True})
        if self.path == "/discussions":
            question = (body.get("question") or "").strip()
            if not question:
                return self._send(400, {"error": "empty question"})
            entry = {
                "id": uuid.uuid4().hex[:12],
                "lecture": body.get("lecture"),
                "chunk_id": body.get("chunk_id"),
                "chunk_title": body.get("chunk_title"),
                "question": question,
                "answer": None,
                "status": "pending",
                "asked_at": datetime.now().isoformat(timespec="seconds"),
                "answered_at": None,
            }
            with LOCK:
                discussions = load_discussions()
                discussions.append(entry)
                save_discussions(discussions)
            return self._send(200, {"ok": True, "id": entry["id"]})
        return self._send(404, {"error": "not found"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    if not RESULTS_FILE.exists():
        save_results([])
    else:
        refresh_weak_areas_md()
    print(f"CS4248 Quizlet on http://{args.host}:{args.port}")
    print(f"  Keyboard-only quiz. State in {RESULTS_FILE}.")
    print(f"  Weak-area snapshot at {WEAK_AREAS_MD}.")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
