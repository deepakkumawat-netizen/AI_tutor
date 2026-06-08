#!/usr/bin/env python3
"""
AI Tutor Backend + MCP Server
- REST API for Frontend
- MCP Tools for Claude
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables FIRST before importing NLP engine
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from telegram import Update
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from mcp_server import get_topics, explain_topic, explain_topic_stream, practice_question, get_educational_videos, quick_answer, TOOLS
from nlp_engine import nlp_engine
import voyage_service

# Track Voyage soft-fail state so we only log a noisy "no payment method /
# rate limited" error once per process instead of on every embed call.
_voyage_warned = False

def _log_voyage_error(where: str, err: Exception) -> None:
    """Log a Voyage error — full message the first time, then a one-liner
    on subsequent calls so the deploy logs aren't flooded by per-request
    rate-limit/no-payment errors that all degrade gracefully to empty
    results anyway."""
    global _voyage_warned
    msg = str(err)
    is_soft = (
        "payment method" in msg.lower()
        or "rate limit" in msg.lower()
        or "rate_limit" in msg.lower()
        or "reduced rate" in msg.lower()
    )
    if is_soft:
        if not _voyage_warned:
            print(f"[WARN] Voyage degraded ({where}): {msg[:200]} — silencing further messages this process.")
            _voyage_warned = True
        return
    # Real errors still surface every time
    print(f"[ERROR] {where}: {err}")

# Import database with explicit error handling
try:
    from database import db
    DB_IMPORT_SUCCESS = True
    DB_IMPORT_ERROR = None
except Exception as e:
    DB_IMPORT_SUCCESS = False
    DB_IMPORT_ERROR = str(e)
    print(f"[✗] CRITICAL: Failed to import database module: {e}")
    # Create a dummy db object to prevent crashes
    class DummyDB:
        db_path = "ERROR: DATABASE IMPORT FAILED"
    db = DummyDB()

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI()

# Global PTB Application instance (set during startup)
_telegram_app = None

# Startup event - verify database is initialized
@app.on_event("startup")
async def startup_event():
    global _telegram_app
    import time
    print("\n" + "="*60)
    print(f"[✓ STARTUP] AI Tutor Backend v2.0 - {time.time()}")
    print("="*60)

    # Start Telegram bot via webhook
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    print(f"[i] TELEGRAM_BOT_TOKEN present: {bool(token)} (len={len(token)})")
    base_url = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
    print(f"[i] RENDER_EXTERNAL_URL: {base_url or '(not set)'}")

    try:
        from telegram_bot import create_application
        ptb = create_application()
        if ptb is None:
            print("[!] create_application returned None — token missing or empty")
        else:
            await ptb.initialize()
            await ptb.start()
            _telegram_app = ptb  # set BEFORE webhook so status is always correct
            print("[✓] Telegram bot initialized and started")

            if base_url:
                webhook_url = f"{base_url}/telegram/webhook"
                try:
                    await ptb.bot.set_webhook(
                        url=webhook_url,
                        allowed_updates=["message", "callback_query"],
                        drop_pending_updates=True,
                    )
                    print(f"[✓] Webhook registered: {webhook_url}")
                except Exception as wh_err:
                    print(f"[!] Webhook registration failed: {wh_err}")
            else:
                # Local mode: no public URL → fall back to polling
                print("[i] RENDER_EXTERNAL_URL not set — starting polling mode for local testing")
                try:
                    await ptb.bot.delete_webhook(drop_pending_updates=True)
                    await ptb.updater.start_polling(
                        allowed_updates=["message", "callback_query"],
                        drop_pending_updates=True,
                    )
                    print("[✓] Telegram bot polling started (local mode)")
                except Exception as poll_err:
                    print(f"[!] Polling start failed: {poll_err}")
    except Exception as e:
        import traceback
        print(f"[!] Telegram bot startup error: {e}")
        print(traceback.format_exc())

@app.on_event("shutdown")
async def shutdown_event():
    global _telegram_app
    if _telegram_app:
        if _telegram_app.updater and _telegram_app.updater.running:
            await _telegram_app.updater.stop()
        await _telegram_app.stop()
        await _telegram_app.shutdown()

# ── Telegram webhook endpoint ──────────────────────────────────────────────────
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    if not _telegram_app:
        return JSONResponse({"ok": True})  # return 200 so Telegram doesn't retry endlessly
    try:
        data = await request.json()
        update = Update.de_json(data, _telegram_app.bot)
        await _telegram_app.process_update(update)
    except Exception as e:
        print(f"[!] Webhook process error: {e}")
    return JSONResponse({"ok": True})

@app.get("/telegram/status")
async def telegram_status():
    if not _telegram_app:
        return JSONResponse({"bot": "not initialized"})
    try:
        info = await _telegram_app.bot.get_webhook_info()
        return JSONResponse({
            "bot": "running",
            "webhook_url": info.url,
            "pending_updates": info.pending_update_count,
            "last_error": info.last_error_message,
        })
    except Exception as e:
        return JSONResponse({"bot": "error", "detail": str(e)})

    if not DB_IMPORT_SUCCESS:
        print(f"[✗] DATABASE IMPORT FAILED: {DB_IMPORT_ERROR}")
        print("[✗] The following endpoints may not work:")
        print("[✗]   - /api/chat-history")
        print("[✗]   - /api/check-usage")
        print("[✗]   - /api/increment-usage")
        print("[✗]   - /api/save-chat")
    else:
        print(f"[✓] Database Path: {db.db_path}")
        print("[✓] Chat History: /api/chat-history (POST)")
        print("[✓] Check Usage: /api/check-usage (POST)")
        print("[✓] Increment Usage: /api/increment-usage (POST)")
        print("[✓] Save Chat: /api/save-chat (POST)")
        print("[✓] Features: Chat History, Usage Counter, Auto-Cleanup")

    print("="*60 + "\n")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PORT = int(os.getenv("PORT", 5000))

# ═══════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ═══════════════════════════════════════════════════════════════════════════

class GetTopicsRequest(BaseModel):
    subject: str
    grade: str
    chapter: Optional[str] = None  # optional CBSE chapter — when set, returns subtopics from that chapter's NCERT concepts, not the chapter list

class FlashcardRequest(BaseModel):
    topic: str
    grade: str
    subject: str
    num_cards: int = 8
    lesson_context: Optional[str] = None  # actual lesson text the student is reading

class PracticeTestRequest(BaseModel):
    topic: str
    grade: str
    subject: str
    num_questions: int = 5
    lesson_context: Optional[str] = None  # actual lesson text the student just read

class ExplainTopicRequest(BaseModel):
    topic: str
    grade: str
    subject: str
    history: list = []
    chapter: Optional[str] = None  # parent CBSE chapter (e.g. 'Chapter 2. Human Reproduction') so the lesson stays scoped to that chapter even when topic is a subtopic

class ExplainPhotoRequest(BaseModel):
    """Student uploads a photo of a problem (math, science, worksheet, etc.)
    and asks a question about it. Image arrives as a base64-encoded string
    to keep the JSON shape consistent with other endpoints (no multipart
    branch in the frontend code)."""
    image_base64: str               # raw base64 string OR data URL ('data:image/jpeg;base64,...')
    image_mime: str = "image/jpeg"  # 'image/jpeg' | 'image/png' | 'image/webp'
    question: str = "Solve this for me step-by-step."
    grade: str = "Grade 6"
    subject: Optional[str] = None
    chapter: Optional[str] = None
    history: list = []

class PracticeQuestionRequest(BaseModel):
    subject: str
    grade: str

class MCPCallRequest(BaseModel):
    tool_name: str
    params: dict

# ═══════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check endpoint that lists all registered routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if route.methods else ["GET"]
            })

    return {
        "status": "healthy" if DB_IMPORT_SUCCESS else "degraded",
        "database_import": "success" if DB_IMPORT_SUCCESS else "failed",
        "database_error": DB_IMPORT_ERROR,
        "database": db is not None,
        "db_path": db.db_path if db else "not initialized",
        "total_routes": len(routes),
        "routes": sorted(routes, key=lambda x: x['path'])
    }

# ═══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS - Using MCP Server Functions
# ═══════════════════════════════════════════════════════════════════════════

# CBSE TOC — drives the Grade → Subject → Chapter dropdowns and the
# curriculum-grounding block in get_topics / explain_topic prompts.
try:
    import cbse_kb
    CBSE_AVAILABLE = bool(cbse_kb.CBSE_KB)
except Exception as _e:
    print(f"[CBSE] knowledge base unavailable: {_e}")
    cbse_kb = None
    CBSE_AVAILABLE = False


@app.get("/api/curriculum")
async def api_curriculum(grade: str = "", subject: str = ""):
    """Return the CBSE TOC for the Grade → Subject → Chapter pickers.

    No params: full {grade: {subject: [chapters]}} tree, plus a sorted
    list of grade keys so the frontend can populate dropdowns in order.
    grade only: subjects available for that grade + chapters per subject.
    grade + subject: chapters for that exact (grade, subject) pair.
    """
    if not CBSE_AVAILABLE:
        return {"available": False, "grades": []}

    grades_sorted = sorted(
        cbse_kb.CBSE_KB.keys(),
        key=lambda g: int("".join(c for c in g if c.isdigit()) or "0"),
    )

    if not grade:
        return {"available": True, "grades": grades_sorted, "tree": cbse_kb.CBSE_KB}

    subjects = cbse_kb.get_subjects(grade)
    if not subject:
        return {
            "available": True,
            "grade": grade,
            "subjects": subjects,
            "chapters": {s: cbse_kb.get_chapters(grade, s) for s in subjects},
        }
    return {
        "available": True,
        "grade": grade,
        "subject": subject,
        "chapters": cbse_kb.get_chapters(grade, subject),
    }


@app.post("/api/mcp/get-topics")
async def api_get_topics(request: GetTopicsRequest):
    """Get topics for a subject (standard or custom).

    Three modes:
      1. (grade, subject, chapter) all set AND chapter found in CBSE TOC:
         return the chapter's NCERT subtopics derived from its `concepts`
         field — these are the chips a student sees AFTER picking a chapter.
      2. (grade, subject) set AND the pair exists in CBSE TOC: return the
         official chapter titles (used before a chapter is picked).
      3. otherwise: fall back to the LLM-generated topic list (custom
         subjects, languages, non-CBSE inputs).
    """
    if CBSE_AVAILABLE:
        # Mode 1 — subtopics within a specific chapter
        if request.chapter:
            ch = cbse_kb.find_chapter(request.grade, request.subject, request.chapter)
            if ch:
                subtopics = cbse_kb.concepts_to_topics(ch.get("concepts", ""))
                chapter_title = ch.get("title", "").strip()
                # Literature chapters (Hindi/English/Sanskrit) often have a
                # single descriptive `concepts` line that doesn't cleanly
                # split — in that case surface the chapter title itself as
                # the one-and-only chip so the student can click it to load
                # the actual NCERT story/poem lesson.
                if not subtopics and chapter_title:
                    subtopics = [chapter_title]
                if subtopics:
                    return {
                        "subject": request.subject,
                        "grade": request.grade,
                        "chapter": ch.get("ch", ""),
                        "chapter_title": chapter_title,
                        "topics": subtopics,
                        "count": len(subtopics),
                        "type": "cbse_subtopics",
                    }
        # Mode 2 — full chapter list for the grade+subject
        chapters = cbse_kb.get_chapters(request.grade, request.subject)
        if chapters:
            topics = [
                ch.get("title", "").strip()
                for ch in chapters
                if ch.get("title")
            ]
            if topics:
                return {
                    "subject": request.subject,
                    "grade": request.grade,
                    "topics": topics,
                    "count": len(topics),
                    "type": "cbse",
                    "chapters": chapters,  # full chapter objects (ch, title, concepts) for richer UI
                }
    return get_topics(request.subject, request.grade)

@app.post("/api/mcp/explain-topic")
async def api_explain_topic(request: ExplainTopicRequest):
    """Explain a topic"""
    result = explain_topic(request.topic, request.grade, request.subject, history=request.history or [], chapter=request.chapter)

    # Log what we're returning
    import json
    return_obj = {
        "topic": result.get("topic"),
        "grade": result.get("grade"),
        "subject": result.get("subject"),
        "explanation": result.get("explanation"),
        "sections": result.get("sections") or {},
        "gradeLevel": result.get("gradeLevel")
    }
    print(f"[RETURN] Keys being returned: {list(return_obj.keys())}")
    print(f"[RETURN] Sections type: {type(return_obj['sections'])}, len: {len(return_obj.get('sections', {}))}")
    return return_obj

@app.post("/api/mcp/explain-photo")
async def api_explain_photo(request: ExplainPhotoRequest):
    """Multimodal vision endpoint: student uploads a photo of a problem
    (math problem, worksheet, science diagram, etc.) and asks a question
    about it. Routed through Gemini Flash (the only provider in our chain
    with native vision). Returns a structured response the chat UI can
    render as a regular bot message.

    Image arrives as base64; we strip any 'data:image/...;base64,' prefix
    the frontend may have included from FileReader.readAsDataURL()."""
    import base64
    import os
    from google import genai
    from google.genai import types as gtypes

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        return {"error": "GEMINI_API_KEY not configured", "answer": "Photo solving is not configured. Please contact support."}

    # Strip data-URL prefix if present (frontend FileReader.readAsDataURL
    # produces 'data:image/jpeg;base64,...').
    b64 = request.image_base64
    if "," in b64 and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(b64, validate=False)
    except Exception as e:
        return {"error": f"Invalid base64 image: {e}", "answer": "Could not decode the photo. Try uploading again."}

    if len(image_bytes) > 5_000_000:  # 5 MB ceiling
        return {"error": "Image too large", "answer": "Photo is too large. Please use a smaller image (under 5 MB)."}

    # Grade-aware system instruction so the explanation matches the student
    grade_num = int("".join(c for c in (request.grade or "") if c.isdigit()) or 6)
    if grade_num <= 3:
        style = "Use VERY simple words, short sentences, and a friendly tone. Maximum 150 words."
    elif grade_num <= 6:
        style = "Use clear age-appropriate language with fun examples. 200-300 words."
    elif grade_num <= 9:
        style = "Use slightly technical language with detailed steps. 300-500 words."
    else:
        style = "Use academic language with rigorous step-by-step working. 500-700 words."

    subject_hint = f" in {request.subject}" if request.subject else ""
    chapter_hint = f" (related to NCERT chapter '{request.chapter}')" if request.chapter else ""

    system_prompt = (
        f"You are a CBSE/NCERT tutor for {request.grade} students{subject_hint}{chapter_hint}. "
        f"The student has uploaded a photo of a problem they are working on. "
        f"{style} "
        "Format your reply EXACTLY as:\n"
        "WHAT I SEE:\n[1-2 lines describing what's in the photo]\n\n"
        "ANSWER:\n[the direct answer or solution]\n\n"
        "STEP-BY-STEP:\n[numbered steps a student can follow]\n\n"
        "WHY THIS WORKS:\n[the underlying concept in 1-2 lines]\n\n"
        "Keep the section LABELS in English even if the body language matches the subject (Hindi/Sanskrit/Urdu/regional)."
    )

    try:
        client = genai.Client(api_key=gemini_key)
        # Build a multimodal message: image part + question part
        image_part = gtypes.Part.from_bytes(data=image_bytes, mime_type=request.image_mime or "image/jpeg")
        text_part = request.question or "Solve this for me step-by-step."

        resp = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
            contents=[image_part, text_part],
            config=gtypes.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4,
                max_output_tokens=1500,
                thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
            ),
        )
        answer = resp.text or ""
        return {
            "answer": answer,
            "grade": request.grade,
            "subject": request.subject,
            "model": "gemini-flash-latest",
        }
    except Exception as e:
        print(f"[ERROR] explain-photo: {e}")
        return {"error": str(e), "answer": "Sorry, I couldn't read the photo just now. Please try again in a moment."}


@app.post("/api/mcp/explain-topic-stream")
async def api_explain_topic_stream(request: ExplainTopicRequest):
    """Stream explanation token by token — text appears as it's generated"""
    import json

    def generate():
        try:
            for token in explain_topic_stream(
                request.topic, request.grade, request.subject,
                history=request.history or [], chapter=request.chapter
            ):
                yield f"data: {json.dumps({'text': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.post("/api/mcp/practice-question")
async def api_practice_question(request: PracticeQuestionRequest):
    """Generate a practice question"""
    return practice_question(request.subject, request.grade)

# ═══════════════════════════════════════════════════════════════════════════
# FLASHCARDS & PRACTICE TEST
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/flashcards")
async def generate_flashcards(request: FlashcardRequest):
    """Generate flashcards for a topic, calibrated to the student's grade via NLP."""
    from mcp_server import chat_with_fallback, get_grade_language
    lang_style = get_grade_language(request.grade)
    lesson_block = ""
    if request.lesson_context and request.lesson_context.strip():
        lesson_block = (
            "\n\nLESSON CONTEXT — the actual lesson text the student is reading. "
            "Derive flashcards from THIS content (key terms, facts, examples, "
            "definitions actually mentioned). Do NOT invent terms outside the lesson:\n"
            f"---\n{request.lesson_context.strip()[:5000]}\n---\n"
        )
    try:
        resp = chat_with_fallback(
            prefer_anthropic=True,
            messages=[{
                "role": "system",
                "content": (
                    f"You are an expert teacher creating flashcards for a {request.grade} student.\n\n"
                    f"{lang_style}\n\n"
                    "Both the FRONT (key term/question) and BACK (definition/answer) must use vocabulary "
                    f"and sentence structure appropriate for {request.grade}. Never use words harder than "
                    "the target term itself on the back side. "
                    "When LESSON CONTEXT is provided, every flashcard must come from that lesson's content. "
                    "Return ONLY valid JSON, no markdown."
                )
            }, {
                "role": "user",
                "content": (
                    f"Create {request.num_cards} flashcards for '{request.topic}' "
                    f"(subject: {request.subject}, reading level: {request.grade}).\n"
                    "Each flashcard: front = key term/question (1-5 words), back = clear definition/answer "
                    "(1-2 short sentences using grade-appropriate words).\n"
                    f"{lesson_block}"
                    'Return JSON: {"flashcards": [{"front": "...", "back": "..."}]}'
                )
            }],
            temperature=0.6,
            max_tokens=1200,
        )
        import json
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return {"success": True, "flashcards": data["flashcards"], "topic": request.topic}
    except Exception as e:
        print(f"[ERROR] flashcards: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/practice-test")
async def generate_practice_test(request: PracticeTestRequest):
    """Generate a grade-calibrated multi-question practice test from lesson content."""
    from mcp_server import chat_with_fallback, get_grade_language
    lang_style = get_grade_language(request.grade)
    lesson_block = ""
    if request.lesson_context and request.lesson_context.strip():
        lesson_block = (
            "\n\nLESSON CONTEXT — the actual lesson text the student just read. "
            "Every question must test something explicitly covered in this lesson. "
            "Never ask about facts that are not in the lesson, never invent absurd "
            "distractors like 'a map drawn by a robot' — wrong options must be "
            "plausible misunderstandings of the lesson content:\n"
            f"---\n{request.lesson_context.strip()[:5000]}\n---\n"
        )
    try:
        resp = chat_with_fallback(
            prefer_anthropic=True,
            messages=[{
                "role": "system",
                "content": (
                    f"You are an expert teacher writing a multiple-choice practice test for a "
                    f"{request.grade} student.\n\n{lang_style}\n\n"
                    f"Every question stem, option, and explanation must use vocabulary and "
                    f"sentence structure appropriate for {request.grade}. Distractors must be "
                    "plausible misconceptions, never silly or off-topic. "
                    "Return ONLY valid JSON, no markdown, no code fences."
                )
            }, {
                "role": "user",
                "content": (
                    f"Create {request.num_questions} multiple-choice questions about '{request.topic}' "
                    f"(subject: {request.subject}, reading level: {request.grade}).\n"
                    "Each question: 4 options labeled A/B/C/D, exactly one correct answer, "
                    "a brief (1-2 sentence) explanation written for the grade level.\n"
                    f"{lesson_block}"
                    'Return JSON: {"questions": [{"question":"...","options":["A) ...","B) ...","C) ...","D) ..."],"correct":"A","explanation":"..."}]}'
                )
            }],
            temperature=0.5,
            max_tokens=1800,
        )
        import json
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return {"success": True, "questions": data["questions"], "topic": request.topic}
    except Exception as e:
        print(f"[ERROR] practice-test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════
# YOUTUBE VIDEO SEARCH
# ═══════════════════════════════════════════════════════════════════════════

class YouTubeRequest(BaseModel):
    subject: Optional[str] = None
    grade: Optional[str] = None
    topic: Optional[str] = None
    query: Optional[str] = None       # explicit search query (CBSE-aware, built on the frontend)
    chapter: Optional[str] = None     # CBSE chapter title — presence triggers ncert_only mode
    ncert_only: Optional[bool] = None  # explicit override; if None, auto-on when chapter is set

class SaveChatRequest(BaseModel):
    student_id: str
    topic: str
    grade_level: str
    subject: str
    request_data: dict
    response_preview: str
    response_content: str = None
    session_id: Optional[str] = None

class ChatHistoryRequest(BaseModel):
    student_id: str
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    session_id: Optional[str] = None
    limit: int = 100

class SessionListRequest(BaseModel):
    student_id: str

class UsageCheckRequest(BaseModel):
    student_id: str
    lesson_type: str

class UsageIncrementRequest(BaseModel):
    student_id: str
    lesson_type: str
    query: Optional[str] = None

class QuickAnswerRequest(BaseModel):
    question: str
    grade: Optional[str] = "Grade 6"

@app.post("/api/youtube")
async def search_youtube(request: YouTubeRequest):
    """Search YouTube for educational videos.

    When a CBSE chapter is present (request.chapter) we automatically turn
    on ncert_only mode so results are restricted to NCERT / Khan Academy /
    DIKSHA and other trusted educational channels — keeps Bollywood and
    random folklore videos out of the lesson view. Caller can also pass
    request.ncert_only explicitly to force the mode."""
    topic = request.topic or request.query or request.subject
    ncert_only = request.ncert_only if request.ncert_only is not None else bool(request.chapter)
    print(f"🎬 Searching videos for {topic} ({request.grade}) ncert_only={ncert_only}...")
    return get_educational_videos(
        subject=request.subject,
        grade=request.grade,
        topic=topic,
        query=request.query,
        ncert_only=ncert_only,
    )

@app.post("/api/quick-answer")
async def get_quick_answer(request: QuickAnswerRequest):
    """Get quick answer to any question (2-4 sentences) with current information"""
    print(f"❓ Answering question: {request.question[:50]}...")
    return quick_answer(request.question, request.grade)

# ═══════════════════════════════════════════════════════════════════════════
# MCP ENDPOINTS - For Claude to call
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/mcp/tools")
async def mcp_tools():
    """Get available MCP tools"""
    return {"tools": TOOLS}

@app.post("/mcp/call")
async def mcp_call(request: MCPCallRequest):
    """Call an MCP tool directly"""
    tool_name = request.tool_name
    params = request.params

    if tool_name == "get-topics":
        return get_topics(params["subject"], params["grade"])
    elif tool_name == "explain-topic":
        return explain_topic(params["topic"], params["grade"], params["subject"])
    elif tool_name == "practice-question":
        return practice_question(params["subject"], params["grade"])
    elif tool_name == "get-videos":
        return get_educational_videos(params["subject"], params["grade"])
    else:
        return {"error": f"Unknown tool: {tool_name}"}

# ═══════════════════════════════════════════════════════════════════════════
# CHAT HISTORY & USAGE TRACKING
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/save-chat")
async def save_chat(request: SaveChatRequest):
    """Save a chat to history (tagged with the active login session_id)"""
    try:
        chat_id = db.save_chat(
            request.student_id,
            request.topic,
            request.grade_level,
            request.subject,
            request.request_data,
            request.response_preview,
            request.response_content or request.response_preview,
            session_id=request.session_id,
        )
        return {"success": True, "chat_id": chat_id, "message": "Chat saved to history"}
    except Exception as e:
        print(f"[ERROR] Failed to save chat: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/chat-history")
async def get_chat_history(request: ChatHistoryRequest):
    """Get chat history with optional date/session filters"""
    try:
        chats = db.get_history(
            request.student_id,
            date_from=request.date_from,
            date_to=request.date_to,
            session_id=request.session_id,
            limit=request.limit,
        )
        return {
            "student_id": request.student_id,
            "chats": chats,
            "count": len(chats),
            "success": True,
        }
    except Exception as e:
        print(f"[ERROR] Failed to get chat history: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/chat-sessions")
async def list_chat_sessions(request: SessionListRequest):
    """List distinct login sessions for the session-filter dropdown."""
    try:
        sessions = db.list_sessions(request.student_id)
        return {"student_id": request.student_id, "sessions": sessions, "success": True}
    except Exception as e:
        print(f"[ERROR] Failed to list chat sessions: {e}")
        return {"success": False, "error": str(e), "sessions": []}

@app.post("/api/check-usage")
async def check_usage(request: UsageCheckRequest):
    """Check daily usage for a lesson type"""
    try:
        result = db.check_usage(request.student_id, request.lesson_type)
        return result
    except Exception as e:
        print(f"[ERROR] Failed to check usage: {e}")
        return {"usage_count": 0, "limit": 50, "remaining": 50, "exceeded": False}

@app.post("/api/increment-usage")
async def increment_usage(request: UsageIncrementRequest):
    """Increment usage count for today"""
    try:
        result = db.increment_usage(request.student_id, request.lesson_type)
        return result
    except Exception as e:
        print(f"[ERROR] Failed to increment usage: {e}")
        return {"usage_count": 0, "limit": 50, "remaining": 50, "exceeded": False}

# ═══════════════════════════════════════════════════════════════════════════
# NLP ANALYSIS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

class NLPAnalysisRequest(BaseModel):
    question: str
    context: str = ""
    grade_level: str = "6"

class NLPIntentRequest(BaseModel):
    text: str

class NLPSentimentRequest(BaseModel):
    text: str

class NLPTopicRequest(BaseModel):
    text: str

@app.post("/api/nlp/analyze")
async def analyze_question(request: NLPAnalysisRequest):
    """Comprehensive NLP analysis of student question"""
    try:
        analysis = nlp_engine.analyze_question(request.question, request.context)
        return {"success": True, "analysis": analysis}
    except Exception as e:
        print(f"[ERROR] NLP Analysis failed: {e}")
        return {"success": False, "error": str(e), "analysis": None}

@app.post("/api/nlp/intent")
async def detect_intent(request: NLPIntentRequest):
    """Detect intent from text (explain, debug, practice, etc)"""
    try:
        intent = nlp_engine.detect_intent(request.text)
        return {"success": True, "intent": intent}
    except Exception as e:
        print(f"[ERROR] Intent detection failed: {e}")
        return {"success": False, "error": str(e), "intent": None}

@app.post("/api/nlp/sentiment")
async def analyze_sentiment(request: NLPSentimentRequest):
    """Analyze emotional state (frustration, confusion, confidence)"""
    try:
        sentiment = nlp_engine.analyze_sentiment(request.text)
        return {"success": True, "sentiment": sentiment}
    except Exception as e:
        print(f"[ERROR] Sentiment analysis failed: {e}")
        return {"success": False, "error": str(e), "sentiment": None}

@app.post("/api/nlp/topics")
async def extract_topics(request: NLPTopicRequest):
    """Extract programming topics from text"""
    try:
        topics = nlp_engine.extract_topics(request.text)
        return {"success": True, "topics": topics}
    except Exception as e:
        print(f"[ERROR] Topic extraction failed: {e}")
        return {"success": False, "error": str(e), "topics": []}

@app.post("/api/nlp/classify")
async def classify_question(request: NLPAnalysisRequest):
    """Classify question and get teaching strategy"""
    try:
        strategy = nlp_engine.classify_question_type(request.question)
        return {"success": True, "strategy": strategy}
    except Exception as e:
        print(f"[ERROR] Question classification failed: {e}")
        return {"success": False, "error": str(e), "strategy": None}

@app.post("/api/nlp/adaptive-response")
async def generate_adaptive_response(request: NLPAnalysisRequest):
    """Generate adaptive response with NLP insights"""
    try:
        # In real usage, base_response would come from the AI
        # For now, we'll just return the analysis
        adaptive = nlp_engine.generate_adaptive_response(
            request.question,
            "Base response from AI model",
            request.grade_level
        )
        return {"success": True, "adaptive": adaptive}
    except Exception as e:
        print(f"[ERROR] Adaptive response generation failed: {e}")
        return {"success": False, "error": str(e), "adaptive": None}

# ═══════════════════════════════════════════════════════════════════════════
# VOYAGE AI SEMANTIC SEARCH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

class SemanticSearchRequest(BaseModel):
    query: str
    subject: str
    grade: str
    top_k: int = 5

class EmbedTopicsRequest(BaseModel):
    subject: str
    grade: str
    topics: list[str]

class RecommendRequest(BaseModel):
    subject: str
    grade: str
    recent_topics: list[str]
    top_k: int = 5

@app.post("/api/semantic/embed-topics")
async def embed_topics(request: EmbedTopicsRequest):
    """Embed a list of topics and store in DB. Call once when topics load."""
    if not voyage_service.is_configured():
        return {"success": False, "error": "Voyage AI not configured"}
    try:
        existing = {row["topic"] for row in db.get_topic_embeddings(request.subject, request.grade)}
        new_topics = [t for t in request.topics if t not in existing]
        if new_topics:
            embeddings = voyage_service.embed_batch(new_topics)
            for topic, emb in zip(new_topics, embeddings):
                db.save_topic_embedding(request.subject, request.grade, topic, emb)
        return {"success": True, "embedded": len(new_topics), "total": len(request.topics)}
    except Exception as e:
        _log_voyage_error("embed_topics", e)
        return {"success": False, "error": str(e)}

@app.post("/api/semantic/search")
async def semantic_search(request: SemanticSearchRequest):
    """Find topics semantically similar to a free-text query."""
    if not voyage_service.is_configured():
        return {"success": False, "results": []}
    try:
        candidates = db.get_topic_embeddings(request.subject, request.grade)
        if not candidates:
            return {"success": True, "results": [], "message": "No embeddings yet"}
        query_emb = voyage_service.embed_text(request.query)
        results = voyage_service.rank_by_similarity(query_emb, candidates, top_k=request.top_k)
        return {"success": True, "results": [{"topic": r["topic"], "score": r["score"]} for r in results]}
    except Exception as e:
        _log_voyage_error("semantic_search", e)
        return {"success": False, "error": str(e), "results": []}

@app.post("/api/semantic/related")
async def related_topics(request: SemanticSearchRequest):
    """Find topics related to the query topic within same subject+grade."""
    if not voyage_service.is_configured():
        # Voyage not configured — fall back to curriculum topics
        try:
            topics_data = get_topics(request.subject, request.grade)
            all_topics = topics_data.get("topics", [])
            filtered = [t for t in all_topics if t.lower() != request.query.lower()]
            query_words = set(request.query.lower().split())
            filtered.sort(key=lambda t: len(query_words & set(t.lower().split())), reverse=True)
            results = [{"topic": t, "score": 0.5} for t in filtered[:request.top_k]]
            return {"success": True, "results": results}
        except:
            return {"success": True, "results": []}
    try:
        candidates = db.get_topic_embeddings(request.subject, request.grade)

        # Auto-embed if this subject+grade has no stored embeddings yet
        if not candidates:
            print(f"[SEMANTIC] No embeddings for {request.subject}/{request.grade} — auto-embedding now")
            try:
                topics_data = get_topics(request.subject, request.grade)
                topics = topics_data.get("topics", [])
                if topics:
                    embeddings = voyage_service.embed_batch(topics)
                    for topic, emb in zip(topics, embeddings):
                        db.save_topic_embedding(request.subject, request.grade, topic, emb)
                    candidates = db.get_topic_embeddings(request.subject, request.grade)
                    print(f"[SEMANTIC] Auto-embedded {len(candidates)} topics for {request.subject}/{request.grade}")
            except Exception as embed_err:
                print(f"[WARN] Auto-embed failed: {embed_err}")

        candidates = [c for c in candidates if c["topic"].lower() != request.query.lower()]
        if not candidates:
            return {"success": True, "results": []}
        query_emb = voyage_service.embed_text(request.query)
        results = voyage_service.rank_diverse(query_emb, candidates, top_k=request.top_k, min_score=0.3)
        return {"success": True, "results": [{"topic": r["topic"], "score": r["score"]} for r in results]}
    except Exception as e:
        _log_voyage_error("related_topics", e)
        return {"success": False, "error": str(e), "results": []}

@app.post("/api/semantic/recommend")
async def recommend_topics(request: RecommendRequest):
    """Recommend next topics based on what the student recently studied."""
    if not voyage_service.is_configured():
        return {"success": False, "results": []}
    try:
        all_candidates = db.get_all_embeddings()
        studied = {t.lower() for t in request.recent_topics}
        candidates = [c for c in all_candidates if c["topic"].lower() not in studied]
        if not candidates or not request.recent_topics:
            return {"success": True, "results": []}
        query_text = " ".join(request.recent_topics[-3:])
        query_emb = voyage_service.embed_text(query_text)
        results = voyage_service.rank_by_similarity(query_emb, candidates, top_k=request.top_k, min_score=0.4)
        return {
            "success": True,
            "results": [{"topic": r["topic"], "subject": r.get("subject"), "score": r["score"]} for r in results]
        }
    except Exception as e:
        _log_voyage_error("recommend_topics", e)
        return {"success": False, "error": str(e), "results": []}


# ═══════════════════════════════════════════════════════════════════════════
# SERVE FRONTEND
# ═══════════════════════════════════════════════════════════════════════════

FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Serve the SPA's index.html on GET. HEAD is accepted (returns the same
    response without a body) so Render's internal health checks stop logging
    'HEAD / 405 Method Not Allowed' on every probe."""
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend not built"}

@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str):
    # Only serve static files and fallback to index.html
    # API and MCP routes are handled by specific endpoints above

    # Don't serve /api/* or /mcp/* paths - let FastAPI handle them
    if full_path.startswith("api/") or full_path.startswith("mcp/"):
        return {"error": "Not found"}

    file_path = FRONTEND_DIST / full_path
    if file_path.exists():
        return FileResponse(file_path)

    # Fallback to index.html for client-side routing
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    return {"error": "Not found"}


# ═══════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print(f"[*] AI Tutor Backend running on http://localhost:{PORT}")
    print("[*] Features: Chat History, Usage Counter, Auto-Cleanup")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
