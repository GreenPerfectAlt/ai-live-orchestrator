"""
pipeline.py — streaming turn pipeline for ai-live-orchestrator.

Ports the core architecture from parlor v2 (fikrikarim/parlor):
- StreamParser: transcript-first protocol (###TRANSCRIPT: line leads the reply)
- pad_tail_silence: 300ms silence appended to WAV to stop Gemma 4 audio
  encoder from hallucinating completions of abruptly-cut final words
- prime_cache: fire-and-discard request to warm llama-server's KV-cache
  while the user is still talking
- estimate_tokens: rough per-part token cost for context rotation
- extract_speak_chunks: sentence-aware TTS chunking for streaming audio
- valid_audio: rejects <100ms WAV files that would 400 the server and
  poison every subsequent turn in history

Adapted for this project's stack (Silero + Supertonic, faster-whisper,
the .bat-launched llama-server).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
import wave
from typing import Awaitable, Callable

import numpy as np

import config

# ── Streaming turn parser ──────────────────────────────────────────────
# The transcript line LEADS the reply: the model commits to what it heard
# BEFORE answering. Transcribing after the response instead makes it a
# paraphrase from memory (WER 0.39 vs 0.00 on clean 33-word utterances,
# measured in parlor v2), and the leading line reaches the client while
# the response is still decoding.

TRANSCRIPT_TAG_RE = re.compile(
    r"#{2,}[ \t]*TRANSCRIPT[ \t]*:[ \t]*",
    re.IGNORECASE,
)
# Parsed tolerantly ("### TRANSCRIPT : ..." happens), but the colon is
# REQUIRED and only [ \t] may follow: this regex runs against a partially
# streamed buffer, so an optional colon matches before the ':' token
# arrives (leaking it into the transcript) and \s* would let a newline
# delta terminate an empty transcript line.

SENTENCE_END_RE = re.compile(r"[.!?…]+\s|[\n]+")

# Appended inside the WAV before the LLM sees the utterance: audio that
# stops abruptly at the VAD cutoff makes Gemma 4's audio encoder
# hallucinate a confident completion of the last word. A beat of silence
# fixes it (measured in parlor v2, tailprobe.py).
TAIL_SILENCE_S = 0.3

# Rough per-part token costs for the context-rotation estimate.
AUDIO_TOKENS_PER_SEC = 32   # Gemma 4's audio encoder at 16kHz s16
IMAGE_TOKENS = 300          # typical vision-encoder cost per frame


class StreamParser:
    """Incrementally parses '### TRANSCRIPT: <words>\\n<response>'.

    feed(delta) returns complete response sentences as they become
    available (transcript-line deltas return none); finalize() returns
    the trailing partial sentence and the transcript.

    With expect_transcript=False (text/image turns without audio) the
    reply streams directly and any imitated trailing tag is cut, never
    spoken.
    """

    def __init__(self, expect_transcript: bool = True):
        self.response = ""
        self.transcript: str | None = None
        self._awaiting = expect_transcript
        self._got_tag = False
        self._buf = ""
        self._before_tag = ""  # stray text before the tag -> response prefix
        self._emitted = 0

    def feed(self, delta: str) -> list[str]:
        if self._awaiting:
            self._buf += delta
            if not self._got_tag:
                m = TRANSCRIPT_TAG_RE.search(self._buf)
                if not m:
                    return []
                self._got_tag = True
                self._before_tag = self._buf[: m.start()]
                self._buf = self._buf[m.end() :]
            # A leading "\n" delta must not terminate an empty transcript.
            self._buf = self._buf.lstrip()
            newline = self._buf.find("\n")
            if newline == -1:
                if len(self._buf) < 600:
                    return []
                # Runaway transcript line: take the first sentence and
                # stream the rest, rather than holding TTS hostage.
                m = SENTENCE_END_RE.search(self._buf)
                newline = m.end() - 1 if m else len(self._buf) - 1
            self.transcript = self._buf[:newline].strip() or None
            self.response = (self._before_tag + self._buf[newline + 1 :]).lstrip()
            self._awaiting = False
            self._buf = ""
        else:
            self.response += delta
        return self._complete_sentences()

    def _complete_sentences(self) -> list[str]:
        # The model occasionally imitates tag-like "##..." markup — never
        # speak anything from one onwards.
        end = len(self.response)
        hash_pos = self.response.find("##", self._emitted)
        if hash_pos != -1:
            end = min(end, hash_pos)
        sentences: list[str] = []
        while True:
            m = SENTENCE_END_RE.search(self.response, self._emitted, end)
            if not m:
                break
            sentence = self.response[self._emitted : m.end()].strip()
            self._emitted = m.end()
            if sentence:
                sentences.append(sentence)
        return sentences

    def finalize(self) -> tuple[list[str], str | None]:
        if self._awaiting:
            if self._got_tag:
                # No newline ever arrived (truncated stream / model ran
                # the reply onto the tag line): first sentence is the
                # transcript, the rest is the reply — never swallow it
                # all silently.
                m = SENTENCE_END_RE.search(self._buf)
                cut = m.end() if m else len(self._buf)
                self.transcript = self._buf[:cut].strip() or None
                self.response = self._before_tag + self._buf[cut:]
            else:
                self.response = self._buf
            self._awaiting = False
        sentences = self._complete_sentences()
        # Cut any imitated tag markup — never speak it.
        tail = re.split(r"#{2,}", self.response[self._emitted :])[0].strip()
        return sentences + ([tail] if tail else []), self.transcript


# ── Message content builders ───────────────────────────────────────────

def image_part(b64: str) -> dict:
    """Vision content part for llama-server's chat completions API."""
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64," + b64},
    }


def audio_part(b64: str) -> dict:
    """Audio content part for llama-server's chat completions API.
    Gemma 4 expects WAV at 16kHz s16 mono."""
    return {
        "type": "input_audio",
        "input_audio": {"data": b64, "format": "wav"},
    }


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def valid_audio(b64: str | None) -> bool:
    """At least ~100ms of 16kHz s16 WAV. llama-server 400s on empty audio,
    and one bad message in history would poison every later request."""
    if not b64:
        return False
    # Base64 length * 3/4 = byte length; header is 44 bytes.
    byte_len = len(b64) * 3 // 4
    return byte_len > 44 + 3200   # ~100ms at 16kHz s16 = 3200 bytes


def user_content(
    image_b64: str | None,
    audio_b64s: list[str],
) -> list[dict]:
    """Media parts of the current user turn, in canonical (cache-stable)
    order: image first, then audio segments oldest-to-newest. Stable
    ordering is required for KV-cache reuse across turns."""
    parts: list[dict] = []
    if image_b64:
        parts.append(image_part(image_b64))
    for b in audio_b64s:
        if valid_audio(b):
            parts.append(audio_part(b))
    return parts


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate for context-rotation decisions. The REAL
    count from llama-server's usage.prompt_tokens wins over this when
    available; this is the safety net for the next turn that hasn't
    been sent yet."""
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content) // 4 + 8
            continue
        if isinstance(content, list):
            for p in content:
                t = p.get("type") if isinstance(p, dict) else None
                if t == "text":
                    total += len(p.get("text", "")) // 4
                elif t == "input_audio":
                    wav_bytes = len(p.get("input_audio", {}).get("data", "")) * 3 // 4
                    total += (wav_bytes // 32000) * AUDIO_TOKENS_PER_SEC
                else:
                    total += IMAGE_TOKENS
                total += 8
    return total


# ── WAV utilities ──────────────────────────────────────────────────────

def wav_to_float32(b64: str) -> np.ndarray:
    """Decode a base64 WAV (16kHz s16) to float32 mono."""
    with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def float32_to_wav_base64(samples: np.ndarray, sample_rate: int = 16000) -> str:
    """Encode float32 mono samples to a base64 WAV (s16 little-endian)."""
    buf = io.BytesIO()
    pcm_int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_int16.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def pad_tail_silence(b64: str, seconds: float = TAIL_SILENCE_S) -> str:
    """Append silence inside the WAV (same file, not a separate part).
    A separate silence part doesn't stop the encoder hallucinating a
    completion of an abruptly-cut last word — the silence must be part
    of the SAME audio the model is asked to transcribe."""
    with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    silence = b"\x00" * (params.sampwidth * params.nchannels
                         * int(seconds * params.framerate))
    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as out:
        out.setparams(params)
        out.writeframes(frames + silence)
    return base64.b64encode(out_buf.getvalue()).decode()


# ── Text cleaning ──────────────────────────────────────────────────────

# Safety-net regexes: excise thought/channel/control tags if the model
# leaks them despite reasoning being disabled. This is NOT a "thinking"
# feature — it's garbage cleanup.
CONTROL_TOKEN_RE = re.compile(r"<\|/?[^>\n]{0,80}?\|>", re.IGNORECASE)
XML_CONTROL_RE = re.compile(
    r"</?(?:tool|tool_call|tool_response|turn|channel|assistant|model|user|system)[^>]*>",
    re.IGNORECASE,
)
THINK_PAIR_RE = re.compile(
    r"<(think|thought|analysis|reasoning)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
THINK_OPEN_RE = re.compile(
    r"<(think|thought|analysis|reasoning)\b[^>]*>.*$",
    re.IGNORECASE | re.DOTALL,
)
CHANNEL_PAIR_RE = re.compile(
    r"<\|channel>\s*(?:thought|analysis|reasoning)\b.*?<channel\|>",
    re.IGNORECASE | re.DOTALL,
)
CHANNEL_OPEN_RE = re.compile(
    r"<\|channel>\s*(?:thought|analysis|reasoning)\b.*$",
    re.IGNORECASE | re.DOTALL,
)
LABEL_RE = re.compile(
    r"\b(?:Транскрипция|Ответ|Assistant|Model)\s*:\s*",
    re.IGNORECASE,
)
SPACE_RE = re.compile(r"[ \t]{2,}")


def strip_thinking_and_controls(text: str, *, final: bool = False) -> str:
    """Remove thought/channel/control markup from streamed or final text.
    When final=True, also strips open-ended tags that were waiting for a
    close that never came (truncated stream)."""
    if not text:
        return ""
    text = CHANNEL_PAIR_RE.sub("", text)
    text = THINK_PAIR_RE.sub("", text)
    if not final:
        text = CHANNEL_OPEN_RE.sub("", text)
        text = THINK_OPEN_RE.sub("", text)
    text = CONTROL_TOKEN_RE.sub("", text)
    text = XML_CONTROL_RE.sub("", text)
    text = LABEL_RE.sub("", text)
    text = text.replace("<channel|>", "").replace("<tool|>", "").replace("<turn|>", "")
    text = SPACE_RE.sub(" ", text)
    return text.strip() if final else text


def sanitize_tts_text(text: str) -> str:
    """Clean text for TTS: strip markup, emojis, markdown, collapse spaces."""
    text = strip_thinking_and_controls(text or "", final=True)
    text = re.sub(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]+", " ", text)
    text = text.replace("*", "").replace("_", "").replace("`", "")
    text = re.sub(r"\s+([.!?…])", r"\1", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def collapse_generated_repeats(text: str) -> str:
    """Dedup stuttered generations: 'the the the' -> 'the'."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"([.!?…])\s*\1+", r"\1", text)
    text = re.sub(r"\b([A-Za-zА-Яа-яЁё]{3,})(?:\1\b)+", r"\1", text)
    text = re.sub(r"\b([\wА-Яа-яЁё-]{2,})(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def clean_generated_response(text: str) -> str:
    """Final cleanup for the stored history entry."""
    return collapse_generated_repeats(strip_thinking_and_controls(text, final=True))


def normalize_stream_delta(chunk_text: str, emitted_text: str) -> tuple[str, str]:
    """Handle the case where llama-server re-emits a prefix of the stream
    (rare, but happens on some backends). Returns (delta_to_emit, new_emitted)."""
    text = chunk_text or ""
    if not text:
        return "", emitted_text
    if text.startswith(emitted_text):
        return text[len(emitted_text) :], text
    if emitted_text.endswith(text):
        return "", emitted_text
    max_overlap = min(len(emitted_text), len(text), 512)
    for n in range(max_overlap, 0, -1):
        if emitted_text.endswith(text[:n]):
            return text[n:], emitted_text + text[n:]
    return text, emitted_text + text


# ── TTS chunking for streaming audio ──────────────────────────────────

def extract_sentences(buffer: str) -> tuple[list[str], str]:
    """Split buffer on sentence boundaries; return (complete, tail)."""
    complete: list[str] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(buffer):
        end = match.end()
        sentence = buffer[start:end].strip()
        if sentence:
            complete.append(sentence)
        start = end
    return complete, buffer[start:]


def extract_speak_chunks(
    buffer: str,
    *,
    force: bool = False,
    first: bool = False,
) -> tuple[list[str], str]:
    """Break buffer into TTS-sized chunks, preferring sentence and comma
    boundaries. Clamps TTS_EARLY_CHARS / TTS_LONG_CHARS to live-mode
    norms regardless of what the .bat file hardcodes (the bat sets
    TTS_EARLY_CHARS=30 and TTS_SPLIT_ON_COMMA=0, which cause the
    noticeable pause after the first words)."""
    buf = SPACE_RE.sub(" ", (buffer or "").strip())
    if not buf:
        return [], ""

    # Sentence-first mode: prefer natural sentence breaks.
    if config.TTS_SENTENCE_STREAMING:
        complete, tail = extract_sentences(buf)
        if complete:
            min_sentence = 6 if first else 14
            out: list[str] = []
            carry = ""
            for sentence in complete:
                sentence = SPACE_RE.sub(" ", sentence.strip())
                if not sentence:
                    continue
                if carry:
                    sentence = (carry + " " + sentence).strip()
                    carry = ""
                if len(sentence) < min_sentence and not force:
                    carry = sentence
                    continue
                clean = sanitize_tts_text(sentence)
                if clean:
                    out.append(clean)
            if out:
                rest_parts = []
                if carry:
                    rest_parts.append(carry)
                if tail.strip():
                    rest_parts.append(tail.strip())
                return out, " ".join(rest_parts).strip()

    # Fallback: character-window chunking with comma/space split points.
    first_chars = max(6, int(config.TTS_EARLY_CHARS))
    target_chars = max(first_chars + 20, int(config.TTS_LONG_CHARS))
    max_chars = max(target_chars + 40, int(config.TTS_MAX_CHARS))
    out: list[str] = []
    threshold = first_chars if first else target_chars
    min_sentence = 8 if first else 20
    while len(buf) >= threshold:
        window_len = min(len(buf), max_chars)
        window = buf[:window_len]
        split_at = -1
        # Prefer sentence ends within the window.
        sentence_ends = [
            m.end() for m in SENTENCE_END_RE.finditer(window) if m.end() >= min_sentence
        ]
        if sentence_ends:
            split_at = sentence_ends[0]
        # Then comma/semicolon/colon/em-dash splits (LIVE-FIX: always on).
        if split_at < 0 and config.TTS_SPLIT_ON_COMMA:
            for sep in [", ", "; ", ": ", " — ", " - "]:
                idx = window.rfind(sep, threshold, window_len)
                if idx >= threshold:
                    split_at = idx + len(sep)
                    break
        # Last resort: any space.
        if split_at < 0:
            idx = window.rfind(" ", threshold, window_len)
            if idx >= threshold:
                split_at = idx + 1
        if split_at < 0:
            break
        chunk = sanitize_tts_text(buf[:split_at])
        if chunk:
            out.append(chunk)
        buf = buf[split_at:].lstrip()
        threshold = target_chars
        min_sentence = 20
        first = False
    if force and buf:
        chunk = sanitize_tts_text(buf)
        if chunk:
            out.append(chunk)
        buf = ""
    return out, buf


# ── Cache priming (from parlor v2) ─────────────────────────────────────

async def prime_cache_async(
    chat_blocking: Callable[[list, int], str],
    messages: list[dict],
) -> None:
    """Fire-and-discard request that pushes a prompt prefix through
    llama-server's cache while the user is still talking. Run in an
    executor so it never blocks the turn loop. Failure is silently
    swallowed — the turn still works, it just pays full prefill."""
    t0 = time.time()
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: chat_blocking(messages, max_tokens=1)
        )
        print(f"primed cache ({time.time() - t0:.2f}s)")
    except Exception as e:
        # Swallowed on purpose — priming is best-effort.
        print(f"cache priming failed (non-fatal): {e}")