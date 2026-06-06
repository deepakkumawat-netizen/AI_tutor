#!/usr/bin/env python3
"""
AI Tutor MCP Server
- Handles both standard subjects and custom subjects
- Claude can call these tools directly via MCP
"""

import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from llm_client import chat_with_fallback, GROQ_PRIMARY as OPENAI_MODEL

api_key = os.getenv("GROQ_API_KEY", "").strip() or "missing-set-GROQ_API_KEY-in-env"
# Streaming endpoints still use the OpenAI-shaped Groq client directly
# (mid-stream provider switching isn't safe).
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

# ═══════════════════════════════════════════════════════════════════════════
# HARD-CODED LANGUAGE TOPICS
# ═══════════════════════════════════════════════════════════════════════════

LANGUAGE_TOPICS = {
    "spanish": ["Spanish Alphabet", "Spanish Vocabulary", "Spanish Grammar", "Spanish Verbs", "Spanish Sentences", "Spanish Listening", "Spanish Writing", "Spanish Speaking"],
    "german": ["German Alphabet", "German Vocabulary", "German Grammar", "German Verbs", "German Sentences", "German Listening", "German Writing", "German Speaking"],
    "france": ["French Alphabet", "French Vocabulary", "French Grammar", "French Verbs", "French Sentences", "French Listening", "French Writing", "French Speaking"],
    "french": ["French Alphabet", "French Vocabulary", "French Grammar", "French Verbs", "French Sentences", "French Listening", "French Writing", "French Speaking"],
    "russian": ["Russian Alphabet", "Russian Vocabulary", "Russian Grammar", "Russian Verbs", "Russian Sentences", "Russian Listening", "Russian Writing", "Russian Speaking"],
    "chinese": ["Chinese Characters", "Chinese Vocabulary", "Chinese Grammar", "Chinese Tones", "Chinese Sentences", "Chinese Listening", "Chinese Writing", "Chinese Speaking"],
    "japanese": ["Japanese Hiragana", "Japanese Vocabulary", "Japanese Grammar", "Japanese Kanji", "Japanese Sentences", "Japanese Listening", "Japanese Writing", "Japanese Speaking"],
    "korean": ["Korean Alphabet", "Korean Vocabulary", "Korean Grammar", "Korean Verbs", "Korean Sentences", "Korean Listening", "Korean Writing", "Korean Speaking"],
    "italian": ["Italian Alphabet", "Italian Vocabulary", "Italian Grammar", "Italian Verbs", "Italian Sentences", "Italian Listening", "Italian Writing", "Italian Speaking"],
    "portuguese": ["Portuguese Alphabet", "Portuguese Vocabulary", "Portuguese Grammar", "Portuguese Verbs", "Portuguese Sentences", "Portuguese Listening", "Portuguese Writing", "Portuguese Speaking"],
    "arabic": ["Arabic Alphabet", "Arabic Vocabulary", "Arabic Grammar", "Arabic Verbs", "Arabic Sentences", "Arabic Listening", "Arabic Writing", "Arabic Speaking"],
}

# ═══════════════════════════════════════════════════════════════════════════
# MCP TOOLS
# ═══════════════════════════════════════════════════════════════════════════

def get_topics(subject: str, grade: str) -> dict:
    """Get learning topics for any subject - custom or standard"""
    subject_lower = subject.lower().strip()

    # Check if it's a known language
    if subject_lower in LANGUAGE_TOPICS:
        return {
            "subject": subject,
            "grade": grade,
            "topics": LANGUAGE_TOPICS[subject_lower],
            "count": len(LANGUAGE_TOPICS[subject_lower]),
            "type": "language"
        }

    # For custom subjects, generate topics using AI
    print(f"🔄 Generating topics for custom subject: {subject}")

    try:
        response = chat_with_fallback(
            messages=[
                {"role": "system", "content": f"""You are a curriculum designer creating topics for teaching {subject}.

IMPORTANT: Generate topics ONLY about {subject}. Do not generate topics about related concepts, culture, customs, or traditions unless {subject} is explicitly about those things.

For example:
- If subject is "Cooking": topics should be "Knife Skills", "Baking Basics", "Food Safety", NOT "Culinary Culture"
- If subject is "Guitar": topics should be "Guitar Parts", "Basic Chords", "Finger Technique", NOT "History of Music"
- If subject is "Spanish": topics should be "Spanish Alphabet", "Spanish Grammar", NOT "Spanish Customs"

Generate exactly 8 learning topics for {subject}."""},
                {"role": "user", "content": f"""Generate 8 specific learning topics for teaching "{subject}" at {grade} level.

The topics should be directly about {subject} itself - the skills, techniques, concepts, and knowledge students need to learn about {subject}.

Format EXACTLY as numbered list:
1. Topic Name
2. Topic Name
3. Topic Name
4. Topic Name
5. Topic Name
6. Topic Name
7. Topic Name
8. Topic Name

Include only the numbered list. No explanations or other text."""}
            ],
            temperature=0.7,
            max_tokens=350
        )

        content = response.choices[0].message.content or ""

        # Parse topics
        topics = []
        for line in content.split('\n'):
            line = line.strip()
            if line and any(char.isdigit() for char in line[:2]):
                # Remove numbering
                topic = line.split('.', 1)[-1].strip()
                if topic and len(topic) > 2:
                    topics.append(topic)

        if len(topics) < 8:
            # Fallback if parsing failed
            topics = [
                f"{subject} Basics",
                f"Introduction to {subject}",
                f"Core Skills in {subject}",
                f"{subject} Techniques",
                f"Practical {subject} Applications",
                f"Advanced {subject} Topics",
                f"Common {subject} Challenges",
                f"{subject} Mastery"
            ]

        return {
            "subject": subject,
            "grade": grade,
            "topics": topics[:8],
            "count": len(topics[:8]),
            "type": "custom"
        }

    except Exception as e:
        print(f"❌ Error generating topics: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Fallback topics
        return {
            "subject": subject,
            "grade": grade,
            "topics": [
                f"{subject} Basics",
                f"Introduction to {subject}",
                f"Core Skills",
                f"Techniques & Methods",
                f"Practical Applications",
                f"Advanced Topics",
                f"Practice & Exercises",
                f"Mastery & Excellence"
            ],
            "count": 8,
            "type": "fallback"
        }

def get_grade_language(grade: str) -> str:
    """Return language style based on grade"""
    grade_num = int(''.join(filter(str.isdigit, grade)) or 6)

    if grade_num <= 3:
        return "Use VERY SIMPLE words, SHORT sentences. Explain like talking to a 7-year-old. Use lots of emojis 🎉 and simple examples. Maximum 200 words."
    elif grade_num <= 6:
        return "Use clear, age-appropriate language with fun examples. Grades 4-6 level. Include some interesting facts. 300-400 words."
    elif grade_num <= 9:
        return "Use slightly technical language. Include more detailed examples and explanations. Grades 7-9 level. 400-500 words."
    else:
        return "Use academic and technical language. Include advanced concepts and detailed analysis. High school level (10-12). 500-700 words."

# ─── CBSE grounding ──────────────────────────────────────────────────────────
# When a (grade, subject) pair has CBSE chapters in cbse_toc.json, every
# explain/practice prompt below is anchored to the actual CBSE NCERT chapter
# so the tutor stays inside the official syllabus instead of drifting into
# off-curriculum material.
try:
    import cbse_kb as _cbse_kb
    _CBSE_AVAILABLE = bool(_cbse_kb.CBSE_KB)
except Exception:
    _cbse_kb = None
    _CBSE_AVAILABLE = False


def _cbse_block(topic: str, grade: str, subject: str, chapter: str = None) -> str:
    """Return a CBSE-grounding prompt fragment for a topic/grade/subject,
    or '' if the topic isn't in the CBSE TOC (custom subject, language, etc).

    Resolution order for the parent chapter:
      1. explicit `chapter` arg (passed from frontend when the student picked
         it in the sidebar — most reliable)
      2. exact match of `topic` against a chapter title (lesson loaded by
         clicking a chapter chip directly)
      3. token-overlap fallback — find the chapter whose `concepts` field
         best mentions the topic (lets a subtopic chip like 'Spermatogenesis'
         still resolve to Chapter 2 Human Reproduction)
    """
    if not _CBSE_AVAILABLE:
        return ""
    chapters = _cbse_kb.get_chapters(grade, subject)
    if not chapters:
        return ""

    match = None

    # 1. explicit chapter from the frontend
    if chapter:
        match = _cbse_kb.find_chapter(grade, subject, chapter) or None

    topic_low = (topic or "").lower().strip()

    # 2. topic == a chapter title
    if not match and topic_low:
        for ch in chapters:
            if (ch.get("title") or "").lower().strip() == topic_low:
                match = ch
                break

    # 3. token-overlap fallback — find the chapter whose concepts cover this subtopic
    if not match and topic_low:
        best_score = 0
        topic_tokens = _cbse_kb._tokenize(topic_low)
        if topic_tokens:
            for ch in chapters:
                blob = (ch.get("title", "") + " " + ch.get("concepts", ""))
                score = len(topic_tokens & _cbse_kb._tokenize(blob))
                if score > best_score:
                    best_score = score
                    match = ch

    if not match:
        return ""

    subtopic_line = ""
    topic_clean = (topic or "").strip()
    chapter_title_clean = (match.get("title") or "").strip()
    if topic_clean and topic_clean.lower() != chapter_title_clean.lower():
        subtopic_line = (
            f"The student is studying the subtopic '{topic_clean}' WITHIN this chapter — "
            "stay tight on this subtopic and use the chapter context only to anchor it.\n"
        )

    # If the chapter title is in a non-English script (Devanagari etc.) it
    # IS the canonical NCERT name — keep it verbatim and tell the model not
    # to translate it.
    title_has_native_script = any(ord(c) > 127 for c in chapter_title_clean)
    native_line = ""
    if title_has_native_script:
        native_line = (
            f"The chapter is called '{chapter_title_clean}' in NCERT — this is the "
            "actual name in its original script. Keep this title verbatim in the lesson; "
            "do not translate it to English and do not invent an English title.\n"
        )

    return (
        "\n=== CBSE / NCERT CURRICULUM GROUNDING ===\n"
        f"This is {match.get('ch', 'a chapter')} of the {subject} NCERT textbook for {grade}.\n"
        f"Official chapter title: {match.get('title', '')}\n"
        f"Official chapter concepts: {match.get('concepts', '')}\n"
        f"{native_line}"
        f"{subtopic_line}"
        "You MUST teach the actual content of this NCERT chapter — the same story, "
        "poem, characters, definitions, terminology, examples and sequence that appear "
        "in the official NCERT textbook for this grade. For literature chapters (Hindi, "
        "English, Sanskrit, regional languages) the chapter is a SPECIFIC story or poem "
        "— teach THAT story/poem (its plot, characters, moral, key lines) instead of "
        "inventing an abstract topic. Do NOT introduce material from other chapters or "
        "other grades. Do NOT replace the NCERT chapter name with a generic descriptive "
        "title.\n"
        "=== END CBSE GROUNDING ===\n"
    )


def _lang_block(subject: str) -> str:
    """Return a strong language directive when the subject is taught IN a
    specific language (Hindi, Sanskrit, Urdu, regional). Empty otherwise.
    This runs independently of CBSE matching so the directive applies even
    for custom subjects or chapters missing from the TOC."""
    if not _CBSE_AVAILABLE:
        return ""
    lang = _cbse_kb.language_for_subject(subject)
    if not lang:
        return ""
    return (
        f"\n=== LANGUAGE DIRECTIVE ===\n"
        f"This subject is {subject}. Write the ENTIRE lesson — every section, every "
        f"heading, every bullet, every example — in {lang}. Do not write any part of "
        f"the lesson in English (English is acceptable only inside parentheses to gloss "
        f"a difficult word for the student). Use vocabulary appropriate for the grade "
        f"level. The student is studying {subject}, so the response must be in {lang}.\n"
        f"=== END LANGUAGE DIRECTIVE ===\n"
    )


def explain_topic(topic: str, grade: str, subject: str, history: list = None, chapter: str = None) -> dict:
    """Explain a topic in detail with grade-appropriate formatting"""
    try:
        grade_num = int(''.join(filter(str.isdigit, grade)) or 6)
        lang_style = get_grade_language(grade)
        cbse_block = _cbse_block(topic, grade, subject, chapter=chapter)
        lang_block = _lang_block(subject)

        messages = [
            {"role": "system", "content": f"""You are an expert tutor explaining '{topic}' from {subject} to {grade} students.

{lang_style}
{cbse_block}{lang_block}
Format EXACTLY as:
DEFINITION:
[Clear definition]

KEY CONCEPTS:
• Concept 1
• Concept 2
• Concept 3

REAL-WORLD EXAMPLE:
[Practical example students can relate to]

SUMMARY:
[Brief recap in 1-2 sentences]

CRITICAL FORMATTING RULE: The section LABELS — DEFINITION:, KEY CONCEPTS:, REAL-WORLD EXAMPLE:, SUMMARY: — MUST appear in ENGLISH exactly as shown, even when the lesson body is written in Hindi / Sanskrit / Urdu / Punjabi / Tamil / regional languages. Only the CONTENT under each label is translated. The frontend parses these English labels to render the lesson in sections; if you translate the labels (e.g. write 'परिभाषा:' instead of 'DEFINITION:'), the whole lesson renders as one unstructured paragraph."""}
        ]
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": f"Explain '{topic}' for {grade} students learning {subject}."})

        # When the topic is a known CBSE chapter (cbse_block populated above
        # OR the caller passed a chapter explicitly), route to Gemini first.
        # Empirically Gemini Flash has substantial NCERT content memorized —
        # specifically the chapter titles, authors, and plot details of
        # popular CBSE chapters — so it produces more accurate lessons than
        # Groq's Llama for school content. Falls back to Groq → Claude on
        # error.
        prefer_gemini = bool(cbse_block) or bool(chapter)
        response = chat_with_fallback(
            messages=messages,
            max_tokens=600,
            temperature=0.7,
            prefer_gemini=prefer_gemini,
        )

        content = response.choices[0].message.content

        # Parse into sections (handle both markdown and plain text headers).
        # Also accept native-script equivalents in case the LLM translates the
        # labels despite the system prompt telling it not to (Hindi lessons
        # frequently emitted 'परिभाषा:' instead of 'DEFINITION:', collapsing
        # the whole lesson into one unstructured paragraph).
        sections = {}
        current_section = None
        current_content = []

        # Aliases for each section header. Lowercased lookup; the original
        # case-mapping for ENGLISH headers stays via line_upper.
        _section_aliases = {
            'definition': [
                'definition', 'परिभाषा', 'परिभासा',  # Hindi
                'परिभाषा', 'व्याख्या',                   # Sanskrit / alt
                'تعریف', 'تعارف',                       # Urdu
                'ਪਰਿਭਾਸ਼ਾ',                            # Punjabi
                'வரையறை',                              # Tamil
                'వ్యాఖ్యానం',                          # Telugu
                'ವ್ಯಾಖ್ಯಾನ',                            # Kannada
                'നിർവചനം',                             # Malayalam
                'व्याख्या',                              # Marathi
                'সংজ্ঞা',                              # Bengali
                'વ્યાખ્યા',                             # Gujarati
            ],
            'keyPoints': [
                'key concepts', 'key points', 'main ideas',
                'मुख्य अवधारणाएं', 'मुख्य अवधारणाएँ', 'मुख्य बिंदु', 'मुख्य विचार',
                'मुख्य संकल्पना',
                'اہم تصورات', 'کلیدی نکات',
                'ਮੁੱਖ ਧਾਰਨਾਵਾਂ',
                'முக்கிய கருத்துக்கள்',
                'ముఖ్య భావనలు',
                'ಪ್ರಮುಖ ಪರಿಕಲ್ಪನೆಗಳು',
                'പ്രധാന ആശയങ്ങൾ',
                'मुख्य संकल्पना',
                'মূল ধারণা',
                'મુખ્ય ખ્યાલો',
            ],
            'example': [
                'real-world example', 'real world example', 'example',
                'वास्तविक दुनिया का उदाहरण', 'वास्तविक उदाहरण', 'उदाहरण',
                'मिसाल', 'مثال', 'حقیقی مثال',
                'ਅਸਲ-ਸੰਸਾਰ ਉਦਾਹਰਨ', 'ਉਦਾਹਰਨ',
                'நிஜ உலக உதாரணம்', 'உதாரணம்',
                'వాస్తవ ప్రపంచ ఉదాహరణ', 'ఉదాహరణ',
                'ನೈಜ ಪ್ರಪಂಚದ ಉದಾಹರಣೆ', 'ಉದಾಹರಣೆ',
                'യഥാർത്ഥ ലോക ഉദാഹരണം', 'ഉദാഹരണം',
                'वास्तविक जगाचे उदाहरण', 'उदाहरण',
                'বাস্তব উদাহরণ', 'উদাহরণ',
                'વાસ્તવિક દુનિયાનું ઉદાહરણ', 'ઉદાહરણ',
            ],
            'summary': [
                'summary', 'recap',
                'सारांश', 'निष्कर्ष', 'सार',
                'خلاصہ', 'اختصار',
                'ਸਾਰ', 'ਸੰਖੇਪ',
                'சுருக்கம்',
                'సారాంశం',
                'ಸಾರಾಂಶ',
                'സംഗ്രഹം',
                'सारांश',
                'সারাংশ',
                'સારાંશ',
            ],
        }

        def _match_section(line_stripped: str, line_upper: str):
            """Return the section key ('definition'/'keyPoints'/'example'/'summary')
            if this line is a section header, else None. Strips markdown #
            and trailing colon for matching."""
            if not (line_stripped.startswith('#') or line_stripped.endswith(':')):
                return None
            # Strip markdown hashes, trailing colon, and bold markers
            label = line_stripped.lstrip('#').strip().rstrip(':').strip()
            label = label.lstrip('*').rstrip('*').strip()
            label_lower = label.lower()
            for section_key, aliases in _section_aliases.items():
                for alias in aliases:
                    if alias.lower() == label_lower:
                        return section_key
                    # Tolerate the label CONTAINING the alias for English
                    # (handles e.g. 'KEY CONCEPTS OF X:' or '### Summary')
                    if section_key in ('keyPoints', 'definition', 'example', 'summary') and alias.isascii() and alias.lower() in label_lower:
                        return section_key
            return None

        lines = content.split('\n')
        for line in lines:
            line_stripped = line.strip()
            line_upper = line_stripped.upper()

            matched_section = _match_section(line_stripped, line_upper)

            if matched_section:
                # Save previous section
                if current_section and current_content:
                    # Remove empty lines from end
                    while current_content and not current_content[-1].strip():
                        current_content.pop()
                    sections[current_section] = '\n'.join(current_content).strip()
                current_content = []
                current_section = matched_section
            elif current_section is not None:
                # Add line to current section (including blank lines)
                current_content.append(line)

        # Save last section
        if current_section and current_content:
            while current_content and not current_content[-1].strip():
                current_content.pop()
            sections[current_section] = '\n'.join(current_content).strip()

        return {
            "topic": topic,
            "grade": grade,
            "subject": subject,
            "explanation": content,
            "sections": sections,
            "gradeLevel": grade_num
        }

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error explaining topic '{topic}': {type(e).__name__}: {error_msg}")
        import traceback
        traceback.print_exc()

        if "insufficient_quota" in error_msg or "429" in error_msg:
            user_msg = "⚠️ AI service is temporarily unavailable (quota exceeded). Please try again later."
        elif "invalid_api_key" in error_msg or "401" in error_msg:
            user_msg = "⚠️ AI service configuration error. Please contact support."
        else:
            user_msg = "Sorry, I couldn't explain this topic. Please try again."

        return {
            "topic": topic,
            "grade": grade,
            "subject": subject,
            "explanation": user_msg,
            "sections": {
                "definition": user_msg,
                "keyPoints": "",
                "example": "",
                "summary": ""
            },
            "gradeLevel": grade_num,
            "error": error_msg
        }

def explain_topic_stream(topic: str, grade: str, subject: str, history: list = None, chapter: str = None):
    """Stream explanation token by token using OpenAI streaming"""
    grade_num = int(''.join(filter(str.isdigit, grade)) or 6)
    lang_style = get_grade_language(grade)
    cbse_block = _cbse_block(topic, grade, subject, chapter=chapter)
    lang_block = _lang_block(subject)

    messages = [
        {"role": "system", "content": f"""You are an expert tutor explaining '{topic}' from {subject} to {grade} students.

{lang_style}
{cbse_block}{lang_block}
Format EXACTLY as:
DEFINITION:
[Clear definition]

KEY CONCEPTS:
• Concept 1
• Concept 2
• Concept 3

REAL-WORLD EXAMPLE:
[Practical example students can relate to]

SUMMARY:
[Brief recap in 1-2 sentences]

CRITICAL FORMATTING RULE: The section LABELS — DEFINITION:, KEY CONCEPTS:, REAL-WORLD EXAMPLE:, SUMMARY: — MUST appear in ENGLISH exactly as shown, even when the lesson body is written in Hindi / Sanskrit / Urdu / Punjabi / Tamil / regional languages. Only the CONTENT under each label is translated. The frontend parses these English labels to render the lesson in sections; if you translate the labels (e.g. write 'परिभाषा:' instead of 'DEFINITION:'), the whole lesson renders as one unstructured paragraph."""}
    ]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": f"Explain '{topic}' for {grade} students learning {subject}."})

    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=600,
        temperature=0.7,
        stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def practice_question(subject: str, grade: str) -> dict:
    """Generate a practice question — CBSE-grounded when available."""
    # Practice questions don't get a single `topic`; ground on the broader
    # subject syllabus by listing all chapter titles for this grade+subject.
    cbse_ctx = ""
    cbse_grounded = False
    if _CBSE_AVAILABLE:
        chapters = _cbse_kb.get_chapters(grade, subject)
        if chapters:
            cbse_grounded = True
            titles = [ch.get("title", "") for ch in chapters if ch.get("title")]
            cbse_ctx = (
                f"\n\nCBSE NCERT {grade} {subject} chapters: {', '.join(titles)}.\n"
                f"The question MUST be from one of these chapters, in CBSE NCERT exam style.\n"
            )
    try:
        response = chat_with_fallback(
            prefer_gemini=cbse_grounded,  # Gemini has better NCERT memorization → use for CBSE practice questions
            messages=[
                {"role": "system", "content": f"Generate a practice question for {grade} students learning {subject}.{cbse_ctx}"},
                {"role": "user", "content": f"Create a practice question for {subject} at {grade} level with answer."}
            ],
            max_tokens=500
        )

        return {
            "subject": subject,
            "grade": grade,
            "question": response.choices[0].message.content
        }

    except Exception as e:
        err = str(e)
        if "insufficient_quota" in err or "429" in err:
            msg = "⚠️ AI service quota exceeded. Please try again later."
        else:
            msg = f"Could not generate a practice question. Please try again."
        return {"error": err, "question": msg, "subject": subject, "grade": grade}

# Whitelist of trusted educational YouTube channels. When the student is on a
# CBSE chapter, video results are restricted to these so they only see
# NCERT-aligned / official educational content — not random folk-tale,
# Bollywood, or mythology channels that the generic search was returning.
# Matched case-insensitively as a substring of the YouTube channelTitle.
TRUSTED_EDU_CHANNELS = [
    # NCERT + Government of India
    "ncert", "ncert official",
    "diksha", "pm evidya", "pmevidya",
    "swayamprabha", "swayam", "nios",
    "cbse", "kvs", "kendriya vidyalaya",
    # Trusted Indian K-12 platforms
    "khan academy",                # global + India variants
    "iken edu", "ikenedu",
    "magnet brains",
    "byju", "byju's",
    "vedantu", "unacademy",
    "lido learning", "extramarks",
    "doubtnut",
    # Trusted explainer channels often used for school content
    "ted-ed", "ted ed",
    "crashcourse", "crash course",
]


def _is_trusted_edu_channel(channel_name: str) -> bool:
    """Case-insensitive substring match against TRUSTED_EDU_CHANNELS."""
    if not channel_name:
        return False
    cn = channel_name.lower()
    return any(t in cn for t in TRUSTED_EDU_CHANNELS)


def get_educational_videos(subject: str, grade: str, topic: str = None, query: str = None, ncert_only: bool = False) -> dict:
    """Get educational YouTube videos for a subject, topic, and grade level.

    Args:
        subject: subject name (e.g. 'Hindi', 'Maths')
        grade: grade string (e.g. 'Grade 5')
        topic: specific topic / chapter name to search (preferred over subject)
        query: explicit search query — overrides the auto-built one when provided
            (frontend builds CBSE-aware queries like
            '\"चतुर चित्रकार\" NCERT Class 5 Hindi chapter')
        ncert_only: if True, hard-filter results to TRUSTED_EDU_CHANNELS only.
            If fewer than 2 trusted videos found, falls back to including
            the top untrusted results so the page isn't empty.
    """
    try:
        youtube_api_key = os.getenv("YOUTUBE_API_KEY")

        if not youtube_api_key:
            return {
                "subject": subject,
                "grade": grade,
                "videos": [],
                "message": "YouTube API key not configured"
            }

        # Extract grade number
        grade_num = int(''.join(filter(str.isdigit, grade)) or 6)

        # Use explicit query if supplied (frontend usually builds a better
        # CBSE-aware one with the NCERT chapter title in native script).
        if query and query.strip():
            search_query = query.strip()
        else:
            search_term = topic if topic else subject
            if grade_num <= 3:
                search_query = f"{search_term} lesson for kids"
            elif grade_num <= 6:
                search_query = f"{search_term} tutorial for {grade}"
            else:
                search_query = f"{search_term} lesson {grade}"

        # Fetch more results than we need so we can filter to trusted
        # educational channels and still return a full set of 6.
        youtube_url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": search_query,
            "type": "video",
            "maxResults": 25 if ncert_only else 8,
            "key": youtube_api_key,
            "order": "relevance",
            "safeSearch": "strict",
        }

        response = requests.get(youtube_url, params=params, timeout=10)

        if response.status_code != 200:
            return {
                "subject": subject,
                "grade": grade,
                "videos": [],
                "error": f"YouTube API error: {response.status_code}"
            }

        data = response.json()

        all_videos = []
        for item in data.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})
            if not video_id:
                continue
            channel = snippet.get("channelTitle", "Unknown")
            all_videos.append({
                "id": video_id,
                "title": snippet.get("title", "Untitled"),
                "description": snippet.get("description", ""),
                "thumbnail": snippet.get("thumbnails", {}).get("default", {}).get("url", ""),
                "channel": channel,
                "trusted": _is_trusted_edu_channel(channel),
                "url": f"https://www.youtube.com/watch?v={video_id}",
            })

        if ncert_only:
            trusted = [v for v in all_videos if v["trusted"]][:6]
            # If we couldn't find enough trusted videos, top up with the
            # most-relevant untrusted ones so the page isn't empty — those
            # come AFTER the trusted ones in display order.
            if len(trusted) < 2:
                extras = [v for v in all_videos if not v["trusted"]][: max(0, 6 - len(trusted))]
                videos = trusted + extras
            else:
                videos = trusted
        else:
            # In default mode, still surface trusted channels first.
            videos = sorted(all_videos[:6], key=lambda v: 0 if v["trusted"] else 1)

        return {
            "subject": subject,
            "grade": grade,
            "videos": videos,
            "count": len(videos),
            "query": search_query,
            "ncert_only": ncert_only,
        }

    except requests.exceptions.Timeout:
        return {
            "subject": subject,
            "grade": grade,
            "videos": [],
            "error": "Request timeout"
        }
    except Exception as e:
        return {
            "subject": subject,
            "grade": grade,
            "videos": [],
            "error": str(e)
        }

def quick_answer(question: str, grade: str = "Grade 6") -> dict:
    """Answer a question with structured, grade-appropriate content."""
    try:
        response = chat_with_fallback(
            messages=[
                {"role": "system", "content": f"""You are a helpful tutor answering student questions clearly and in a structured way.

Grade level: {grade}
For younger grades (K-3): Use very simple words and short sentences.
For older grades (7-12): Use more technical, academic language.

Always structure your response with these plain-text headers (no asterisks, no markdown):
ANSWER:
[your direct answer]

EXAMPLE:
[a clear, relatable example]

REMEMBER:
[one key takeaway sentence]

Today's date: {__import__('datetime').date.today().strftime('%B %d, %Y')}."""},
                {"role": "user", "content": question}
            ],
            max_tokens=600,
            temperature=0.7
        )

        answer = response.choices[0].message.content

        return {
            "question": question,
            "answer": answer,
            "grade": grade,
            "source": "real-time knowledge"
        }
    except Exception as e:
        return {
            "question": question,
            "answer": f"Sorry, I couldn't find an answer to that question. Error: {str(e)}",
            "error": str(e)
        }

# ═══════════════════════════════════════════════════════════════════════════
# MCP TOOL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

TOOLS = {
    "get-topics": {
        "description": "Get learning topics for any subject (standard languages or custom subjects)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Subject name (Spanish, Cooking, Economics, etc.)"},
                "grade": {"type": "string", "description": "Grade level (Grade 6, Grade 12, etc.)"}
            },
            "required": ["subject", "grade"]
        }
    },
    "explain-topic": {
        "description": "Explain a specific topic in detail",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic to explain"},
                "grade": {"type": "string", "description": "Grade level"},
                "subject": {"type": "string", "description": "Subject"}
            },
            "required": ["topic", "grade", "subject"]
        }
    },
    "practice-question": {
        "description": "Generate a practice question",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Subject"},
                "grade": {"type": "string", "description": "Grade level"}
            },
            "required": ["subject", "grade"]
        }
    },
    "get-videos": {
        "description": "Get educational YouTube videos for a subject and grade level",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Subject name (Spanish, Math, Science, etc.)"},
                "grade": {"type": "string", "description": "Grade level (Grade 6, Grade 12, etc.)"}
            },
            "required": ["subject", "grade"]
        }
    },
    "quick-answer": {
        "description": "Answer any question in a few lines (2-4 sentences) with current/recent information",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Any question to answer (e.g., 'Who is the recent PM of Pakistan?')"},
                "grade": {"type": "string", "description": "Grade level for language adjustment (default: Grade 6)"}
            },
            "required": ["question"]
        }
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# TOOL EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════

def execute_tool(tool_name: str, params: dict) -> dict:
    """Execute an MCP tool"""
    if tool_name == "get-topics":
        return get_topics(params["subject"], params["grade"])
    elif tool_name == "explain-topic":
        return explain_topic(params["topic"], params["grade"], params["subject"])
    elif tool_name == "practice-question":
        return practice_question(params["subject"], params["grade"])
    elif tool_name == "get-videos":
        return get_educational_videos(params["subject"], params["grade"])
    elif tool_name == "quick-answer":
        grade = params.get("grade", "Grade 6")
        return quick_answer(params["question"], grade)
    else:
        return {"error": f"Unknown tool: {tool_name}"}

# ═══════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing MCP Tools...\n")

    # Test standard language
    print("1️⃣ Spanish (standard language):")
    result = get_topics("spanish", "Grade 6")
    print(f"   Topics: {result['topics']}\n")

    # Test custom subject
    print("2️⃣ Cooking (custom subject):")
    result = get_topics("cooking", "Grade 6")
    print(f"   Topics: {result['topics']}\n")

    print("✅ MCP Server ready!")
    print("\nTools available:")
    for tool_name in TOOLS:
        print(f"  - {tool_name}")
