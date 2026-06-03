"""CBSE curriculum RAG for ClassroomAI — loaded from cbse_toc.json
(parsed from GRADE Wise TOC.xlsx: 12 grades, all subjects, ~769 chapters).

Used to ground every generator's output in the official CBSE curriculum.
"""
import json
import os
import re

_HERE = os.path.dirname(__file__)
_TOC = os.path.join(_HERE, "cbse_toc.json")

try:
    with open(_TOC, encoding="utf-8") as _f:
        CBSE_KB = json.load(_f)
except Exception:
    CBSE_KB = {}

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on",
    "at", "by", "for", "with", "about", "as", "and", "or", "but", "if", "this",
    "that", "what", "how", "explain", "create", "make", "worksheet", "lesson",
    "questions", "quiz", "test", "grade", "students", "chapter", "topic",
}


def _grade_key(grade_level) -> str:
    """Normalize various grade inputs to 'Grade N'."""
    s = str(grade_level or "")
    m = re.search(r"\d+", s)
    return f"Grade {m.group()}" if m else ""


def _tokenize(text):
    text = (text or "").lower()
    words = "".join(c if c.isalnum() or c.isspace() else " " for c in text).split()
    return {w for w in words if len(w) > 2 and w not in STOP_WORDS}


def get_subjects(grade_level) -> list:
    return list(CBSE_KB.get(_grade_key(grade_level), {}).keys())


def get_chapters(grade_level, subject) -> list:
    return CBSE_KB.get(_grade_key(grade_level), {}).get(subject, [])


def retrieve_context(query, grade_level, subject="", top_k=3) -> str:
    """Find the most relevant CBSE chapters for a topic/query and return a
    formatted context block to ground the generator prompt."""
    gk = _grade_key(grade_level)
    grade_data = CBSE_KB.get(gk, {})
    if not grade_data:
        return ""

    # Search within the chosen subject if given, else across all subjects of the grade
    candidates = []
    subjects = [subject] if subject and subject in grade_data else list(grade_data.keys())
    for subj in subjects:
        for ch in grade_data.get(subj, []):
            candidates.append((subj, ch))

    if not candidates:
        return ""

    q_tokens = _tokenize(query)
    if not q_tokens:
        return ""

    scored = []
    for subj, ch in candidates:
        text = (ch.get("title", "") + " " + ch.get("concepts", "")
                + " " + ch.get("unit", "") + " " + ch.get("stream", ""))
        overlap = len(q_tokens & _tokenize(text))
        if overlap > 0:
            scored.append((overlap, subj, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    if not top:
        return ""

    lines = [f"OFFICIAL CBSE {gk} CURRICULUM CONTEXT (align content to this):"]
    for _, subj, ch in top:
        unit = ch.get("unit") or ch.get("stream")
        unit_str = f" [{unit}]" if unit else ""
        concepts = ch.get("concepts", "")
        lines.append(f"- {subj} · {ch.get('ch','')}: {ch.get('title','')}{unit_str}"
                     + (f" — {concepts}" if concepts else ""))
    return "\n".join(lines)
