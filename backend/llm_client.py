"""Multi-provider LLM client with auto-fallback.

Single entry point: `chat_with_fallback(messages, **kwargs)`. Walks a chain
of providers and returns the first successful response in an OpenAI-shaped
object (`resp.choices[0].message.content`). The chain is:

  1. Claude Haiku 4.5              (preferred — paid, prompt-cached, fastest)
  2. Groq llama-3.3-70b-versatile  (free tier — best quality among the 3)
  3. Groq llama-3.1-8b-instant     (smaller Groq model — separate daily quota)
  4. Groq gemma2-9b-it             (separate daily quota again)
  5. Gemini Flash (latest)         (paid, cheapest — final safety net)

Why this order: Claude first when credits are available (prompt caching
makes repeat lessons very cheap), Groq's free-tier per-model quotas next
(each model has its own 100K-tokens/day budget), then Gemini as the final
paid safety net so the tool never goes completely dark.

Callers do not need to know which provider answered — the response shape
is identical (`resp.choices[0].message.content`).
"""

import os
import requests
from openai import OpenAI

_GROQ_KEY   = (os.getenv("GROQ_API_KEY")      or "").strip()
_ANTHRO_KEY = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
_GEMINI_KEY = (os.getenv("GEMINI_API_KEY")    or "").strip()

GROQ_PRIMARY = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODELS = [GROQ_PRIMARY, "llama-3.1-8b-instant", "gemma2-9b-it"]

CLAUDE_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

_groq = OpenAI(api_key=_GROQ_KEY or "missing", base_url="https://api.groq.com/openai/v1") if _GROQ_KEY else None

_anthropic = None
if _ANTHRO_KEY:
    try:
        import anthropic
        _anthropic = anthropic.Anthropic(api_key=_ANTHRO_KEY)
    except ImportError:
        _anthropic = None

_gemini_ready = bool(_GEMINI_KEY)


def _is_rate_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in (
        "rate_limit", "429", "tokens per day", "tpd", "quota", "rate limit",
    ))


# Track Claude state per-process so a permanent error (credit exhausted,
# invalid key, etc.) doesn't cause every single request to retry it and
# pay ~500ms of round-trip latency before falling through to Groq.
_claude_disabled_reason = None
_gemini_disabled_reason = None


def _is_claude_permanent_failure(err: Exception) -> bool:
    """Errors that won't resolve by retrying — disable Claude until the
    process restarts (next deploy)."""
    msg = str(err).lower()
    return any(s in msg for s in (
        "credit balance is too low",
        "invalid x-api-key",
        "authentication_error",
        "permission_error",
    ))


def _is_gemini_permanent_failure(err: Exception) -> bool:
    """Errors that won't resolve by retrying for Gemini — bad key, no
    billing, perm denied. Don't waste latency retrying these."""
    msg = str(err).lower()
    return any(s in msg for s in (
        "api key not valid",
        "permission_denied",
        "invalid_argument",
        "api_key_invalid",
        "billing",
        "consumer_invalid",
    ))


class _Msg:
    __slots__ = ("content", "role")
    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role


class _Choice:
    __slots__ = ("message", "index", "finish_reason")
    def __init__(self, content: str):
        self.message = _Msg(content)
        self.index = 0
        self.finish_reason = "stop"


class _ClaudeResponse:
    """OpenAI-shaped wrapper around an Anthropic response."""
    def __init__(self, content: str, model: str):
        self.choices = [_Choice(content)]
        self.model = model


class _GeminiResponse:
    """OpenAI-shaped wrapper around a Gemini response."""
    def __init__(self, content: str, model: str):
        self.choices = [_Choice(content)]
        self.model = model


def _call_gemini(messages, **kwargs):
    """Call Gemini via raw HTTP (no SDK dependency — uses the same
    endpoint shape the user verified with curl):
        POST /v1beta/models/{model}:generateContent
        X-goog-api-key: <key>
        body: { contents: [{parts: [{text: ...}], role}], systemInstruction, generationConfig }

    Converts OpenAI-style messages (system/user/assistant) to Gemini's
    contents+systemInstruction format. Returns an OpenAI-shaped object so
    callers don't need to know Gemini answered."""
    if not _gemini_ready:
        raise RuntimeError("Gemini not configured — set GEMINI_API_KEY in env")

    # Gemini uses a separate `systemInstruction` field; conversational
    # turns go in `contents`. Roles in Gemini: 'user' and 'model'.
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    contents = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            continue
        if role == "assistant":
            role = "model"
        if role not in ("user", "model"):
            role = "user"
        contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

    if not contents:
        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

    gen_cfg = {}
    if "temperature" in kwargs:
        gen_cfg["temperature"] = kwargs["temperature"]
    if "max_tokens" in kwargs or "max_completion_tokens" in kwargs:
        gen_cfg["maxOutputTokens"] = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 2048

    body = {"contents": contents}
    if system_parts:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    if gen_cfg:
        body["generationConfig"] = gen_cfg

    resp = requests.post(
        GEMINI_ENDPOINT,
        headers={"Content-Type": "application/json", "X-goog-api-key": _GEMINI_KEY},
        json=body,
        timeout=60,
    )
    if resp.status_code != 200:
        # Surface the API's actual message so the permanent-failure detector
        # can recognize 'API key not valid' / 'billing' / etc.
        raise RuntimeError(f"Gemini API {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {data}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    return _GeminiResponse(text, GEMINI_MODEL)


def _call_claude(messages, **kwargs):
    if _anthropic is None:
        raise RuntimeError("Anthropic SDK not configured — set ANTHROPIC_API_KEY and install `anthropic`")

    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    chat_msgs = []
    for m in messages:
        if m.get("role") == "system":
            continue
        role = m.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        chat_msgs.append({"role": role, "content": m.get("content", "")})

    if not chat_msgs:
        chat_msgs = [{"role": "user", "content": "Hello"}]

    params = {
        "model": CLAUDE_MODEL,
        "max_tokens": kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 2048,
        "messages": chat_msgs,
    }
    if system_parts:
        # Anthropic prompt caching: when the system prompt is large and
        # repeats (CBSE chapter grounding + language directive + grade-style
        # block + format spec on every explain_topic call), wrap it in the
        # structured array form with cache_control. Reads from cache cost
        # ~10% of input tokens and are ~30-50% faster. The minimum cached
        # block size is 1024 tokens for sonnet/opus and 2048 for haiku; our
        # combined system prompt for a CBSE lesson easily clears that.
        # Cache survives 5 minutes between requests (ephemeral TTL) — long
        # enough that consecutive student requests in a session hit it.
        params["system"] = [
            {
                "type": "text",
                "text": "\n\n".join(system_parts),
                "cache_control": {"type": "ephemeral"},
            }
        ]
    if "temperature" in kwargs:
        params["temperature"] = kwargs["temperature"]

    resp = _anthropic.messages.create(**params)
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return _ClaudeResponse(text, CLAUDE_MODEL)


def chat_with_fallback(messages, prefer_anthropic: bool = True, **kwargs):
    """Run a chat completion, falling back across providers on rate-limit errors.

    Strips any model kwarg the caller passed — this helper chooses the model.
    Non-rate-limit errors propagate immediately so real bugs surface.

    Default is prefer_anthropic=True: Claude Haiku 4.5 is tried FIRST since
    the Groq free tier's daily quotas (100K tokens/model) are tight enough
    that students see 429s in normal use. Claude is paid and uncapped at our
    usage level, so it's both faster (no quota stalls) and steadier. If
    Claude is unreachable or no key is set, the call drops through to the
    Groq chain so the tool keeps working as a free-tier fallback.

    Pass prefer_anthropic=False explicitly for any low-stakes background
    call where the cost difference matters more than latency.
    """
    kwargs.pop("model", None)
    global _claude_disabled_reason

    last_err = None
    claude_ready = _anthropic is not None and _claude_disabled_reason is None

    if prefer_anthropic and claude_ready:
        try:
            return _call_claude(messages, **kwargs)
        except Exception as e:
            last_err = e
            if _is_claude_permanent_failure(e):
                _claude_disabled_reason = str(e).splitlines()[0][:200]
                print(f"[llm] Claude disabled for this process — {_claude_disabled_reason}. Future requests skip Claude.")
            else:
                print(f"[llm] Claude (preferred) failed: {e} — falling through to Groq")

    if _groq is not None:
        for model in GROQ_FALLBACK_MODELS:
            try:
                return _groq.chat.completions.create(model=model, messages=messages, **kwargs)
            except Exception as e:
                if not _is_rate_limit(e):
                    raise
                last_err = e
                print(f"[llm] Groq {model} rate-limited, trying next…")

    if not prefer_anthropic and claude_ready:
        try:
            print(f"[llm] All Groq models exhausted — falling back to Claude {CLAUDE_MODEL}")
            return _call_claude(messages, **kwargs)
        except Exception as e:
            last_err = e
            if _is_claude_permanent_failure(e):
                _claude_disabled_reason = str(e).splitlines()[0][:200]
                print(f"[llm] Claude disabled for this process — {_claude_disabled_reason}")
            else:
                print(f"[llm] Claude fallback failed: {e}")

    # Final safety net: Gemini. Only reached when both Claude and every
    # Groq model failed (rate-limited or unreachable). Cheapest of the
    # paid options so it's the last resort, not the first.
    global _gemini_disabled_reason
    gemini_ready = _gemini_ready and _gemini_disabled_reason is None
    if gemini_ready:
        try:
            print(f"[llm] All other providers exhausted — falling back to Gemini {GEMINI_MODEL}")
            return _call_gemini(messages, **kwargs)
        except Exception as e:
            last_err = e
            if _is_gemini_permanent_failure(e):
                _gemini_disabled_reason = str(e).splitlines()[0][:200]
                print(f"[llm] Gemini disabled for this process — {_gemini_disabled_reason}. Future requests skip Gemini.")
            else:
                print(f"[llm] Gemini fallback failed: {e}")

    if last_err is not None:
        raise last_err
    raise RuntimeError("No LLM provider configured — set GROQ_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY")
