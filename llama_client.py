"""
llama_client.py — llama.cpp server lifecycle + chat API client.

Ports the architecture from parlor v2 (fikrikarim/parlor):
- spawns/manages llama-server (or reuses one started by the .bat launcher)
- blocking + streaming chat completions against /v1/chat/completions
- ChatStream with REAL cancel() via socket shutdown (not just a flag)
- fire-and-discard prime_cache() for speculative KV-cache warmup

Key difference from parlor: the .bat launcher already starts llama-server,
so auto_start defaults to False. Set LLAMA_AUTO_START=1 to let this module
manage the process instead.
"""
from __future__ import annotations

import http.client
import json
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from config import (
    LLAMA_AUTO_START,
    LLAMA_BASE_URL,
    LLAMA_BATCH_SIZE,
    LLAMA_CTX_SIZE,
    LLAMA_EXTRA_ARGS,
    LLAMA_HOST,
    LLAMA_MODEL,
    LLAMA_N_GPU_LAYERS,
    LLAMA_PORT,
    LLAMA_SERVER_EXE,
    LLAMA_THREADS,
    MODELS_DIR,
    MODEL_PATH,
)
import os

# Gemma 4 audio support requires llama.cpp b9503+ (upstream #24084); the
# 12B mmproj additionally needs b9512. An old build silently fails to load
# the mmproj or aborts on audio-capable requests, so we probe the binary
# upfront and refuse to start on anything older.
MIN_BUILD = 9503

_proc: subprocess.Popen | None = None
_active_model: str = ""
_active_mmproj: str = ""


def _is_managed() -> bool:
    """Whether this module (not the .bat launcher) owns the llama-server."""
    return LLAMA_AUTO_START and _proc is not None


def host_port() -> tuple[str, int]:
    """(host, port) parsed from LLAMA_BASE_URL — works for external servers too."""
    try:
        no_scheme = LLAMA_BASE_URL.split("//")[-1].split("/")[0]
        host, _, port = no_scheme.partition(":")
        return host, int(port or 80)
    except (ValueError, IndexError):
        return LLAMA_HOST, LLAMA_PORT


def _connect(timeout: float = 300.0) -> http.client.HTTPConnection:
    """Raw http.client connection — urllib.request has no way to abort a
    streaming read, so we use http.client directly to support cancel()."""
    return http.client.HTTPConnection(*host_port(), timeout=timeout)


def model_label() -> str:
    """Human-readable model name for the UI (matches .bat's MODEL_LABEL
    when set, falls back to the GGUF stem)."""
    path = MODEL_PATH
    if path:
        return Path(path).stem
    return LLAMA_MODEL


def server_command() -> list[str]:
    """The llama.cpp server invocation. Detects both the standalone
    `llama-server` binary and the unified `llama` binary whose `serve`
    subcommand is the same server."""
    # Try the .bat-provided path first (LLAMA_SERVER_EXE from env)
    exe = LLAMA_SERVER_EXE.strip().strip('"') or "llama-server.exe"
    exe_path = Path(exe).expanduser()
    if exe_path.exists():
        return [str(exe_path)]
    # Fall back to PATH lookup (brew on macOS, installer on Linux)
    standalone = shutil.which("llama-server")
    if standalone:
        return [standalone]
    unified = shutil.which("llama")
    if unified:
        return [unified, "serve"]
    raise RuntimeError(
        f"llama-server not found at {exe!r} or on PATH. "
        "Set LLAMA_SERVER_EXE or install llama.cpp."
    )


def check_build(cmd: list[str], floor: int = MIN_BUILD) -> None:
    """Refuse to start on a llama.cpp build below the floor. The version
    line looks like 'version: 10150 (dee2a846b)'; self-built trees that
    print 'version: 0 (unknown)' are let through — the guard is for
    stale installs, not custom builds."""
    try:
        out = subprocess.run(
            [cmd[0], "--version"],
            capture_output=True, text=True, timeout=5,
        )
        m = re.search(r"version:\s*(\d+)", out.stderr + out.stdout)
    except (OSError, subprocess.SubprocessError):
        return
    if m and 0 < int(m.group(1)) < floor:
        raise RuntimeError(
            f"llama.cpp build {m.group(1)} is too old for Gemma 4 audio "
            f"(needs {floor}+). Upgrade: see "
            "https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md"
        )


def start() -> None:
    """Spawn llama-server with the configured model + flags, and wait for
    /health to return 200. Called by load_models() when LLAMA_AUTO_START=1;
    otherwise the .bat launcher already started the process and this is a no-op."""
    global _proc, _active_model, _active_mmproj

    if not LLAMA_AUTO_START:
        # .bat started it — just verify it's reachable.
        # Use LLAMA_STARTUP_TIMEOUT from config (default 240s from .bat)
        timeout = float(os.environ.get("LLAMA_STARTUP_TIMEOUT", "240"))
        print(f"⏳ Waiting for llama-server at {LLAMA_BASE_URL} (timeout {timeout:.0f}s)...")
        if _wait_for_health(timeout=timeout):
            _active_model = MODEL_PATH
            print(f"✅ llama-server already running at {LLAMA_BASE_URL}")
            return
        raise RuntimeError(
            f"llama-server at {LLAMA_BASE_URL} is not reachable after {timeout:.0f}s. "
            "Either let the .bat start it, or set LLAMA_AUTO_START=1."
        )

    cmd = server_command()
    check_build(cmd, MIN_BUILD)

    model = MODEL_PATH
    if not model or not Path(model).exists():
        raise RuntimeError(f"Model not found: {model!r}")

    # Resolve mmproj next to the model (gemma-4-* pattern match)
    mmproj = _find_mmproj(model)

    print(f"🚀 Starting llama-server with {Path(model).name} (ctx={LLAMA_CTX_SIZE})...")

    args = [
        "-m", model,
        "--host", LLAMA_HOST,
        "--port", str(LLAMA_PORT),
        "--ctx-size", str(LLAMA_CTX_SIZE),
        "--threads", str(LLAMA_THREADS),
        "--batch-size", str(LLAMA_BATCH_SIZE),
    ]
    if LLAMA_N_GPU_LAYERS:
        args += ["-ngl", LLAMA_N_GPU_LAYERS]
    if mmproj:
        args += ["--mmproj", mmproj]
    if LLAMA_EXTRA_ARGS:
        args += shlex.split(LLAMA_EXTRA_ARGS)

    # Output goes to DEVNULL — uncomment to debug llama-server itself.
    _proc = subprocess.Popen(
        cmd + args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(Path(__file__).resolve().parent),
    )

    if not _wait_for_health(timeout=240):
        _proc.terminate()
        raise RuntimeError("llama-server did not become ready in 240s")

    _active_model = model
    _active_mmproj = mmproj or ""
    print(f"✅ llama-server ready at {LLAMA_BASE_URL}")


def _wait_for_health(timeout: float) -> bool:
    """Poll /health until 200 or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _proc is not None and _proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {_proc.returncode}"
            )
        try:
            conn = _connect(timeout=2)
            conn.request("GET", "/health")
            ok = conn.getresponse().status == 200
            conn.close()
            if ok:
                return True
        except OSError:
            pass
        time.sleep(1)
    return False


def _find_mmproj(model_path: str) -> str:
    """Locate the mmproj GGUF paired with the given model. Looks in the
    same directory first, then falls back to the MODELS_DIR tree."""
    model_dir = Path(model_path).parent
    # Try sibling files with 'mmproj' in the name
    for p in model_dir.glob("*.gguf"):
        if "mmproj" in p.name.lower():
            return str(p)
    # Fallback: scan MODELS_DIR recursively
    for p in MODELS_DIR.rglob("*.gguf"):
        if "mmproj" in p.name.lower():
            return str(p)
    return ""


def stop() -> None:
    """Terminate the managed llama-server. No-op if the .bat owns it."""
    global _proc, _active_model, _active_mmproj
    if _proc is None:
        return
    try:
        _proc.terminate()
        _proc.wait(timeout=8)
    except Exception:
        try:
            _proc.kill()
        except Exception:
            pass
    _proc = None
    _active_model = ""
    _active_mmproj = ""


def _chat_body(
    messages: list,
    max_tokens: int,
    stream: bool,
    temperature: float | None = None,
    json_schema: dict | None = None,
    sampler: dict | None = None,
) -> dict:
    """Build the /v1/chat/completions payload. sampler overrides defaults."""
    sampler = sampler or {}
    body: dict = {
        "model": LLAMA_MODEL,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens > 0 else None,
        "temperature": temperature if temperature is not None
                       else sampler.get("temperature", 1.0),
        "stream": stream,
        "cache_prompt": True,  # reuse the KV-cache across turns
    }
    # Top-level sampler params — llama-server accepts them flat.
    for key in ("top_p", "top_k", "min_p", "typical_p",
                "repeat_penalty", "repeat_last_n",
                "xtc_probability", "xtc_order", "top_n_sigma",
                "mirostat", "mirostat_tau", "mirostat_eta"):
        if key in sampler:
            body[key] = sampler[key]
    if json_schema:
        # llama-server compiles the schema to a grammar: the output is
        # structurally guaranteed to parse (used by the action decider).
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"schema": json_schema},
        }
    if stream:
        # The final chunk carries usage.prompt_tokens — the REAL context
        # size, which drives history rotation (estimates drift).
        body["stream_options"] = {"include_usage": True}
    # Remove None values — llama-server rejects null max_tokens.
    return {k: v for k, v in body.items() if v is not None}


def chat_blocking(
    messages: list,
    max_tokens: int,
    temperature: float | None = None,
    json_schema: dict | None = None,
    sampler: dict | None = None,
) -> str:
    """Non-streaming request; returns the message content ('' on discard).
    Used by prime_cache (max_tokens=1) and the action decider (temp=0)."""
    conn = _connect(timeout=300)
    body = _chat_body(messages, max_tokens, stream=False,
                      temperature=temperature, json_schema=json_schema,
                      sampler=sampler)
    conn.request("POST", "/v1/chat/completions",
                 json.dumps(body),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    if "error" in data:
        raise RuntimeError(f"llama-server: {data['error']}")
    choices = data.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content") or ""


class ChatStream:
    """Streaming chat request, driven from an executor thread.

    cancel() is thread-safe and actually aborts generation server-side:
    the socket shutdown is observed by llama-server, which stops
    generating and closes the SSE stream. A simple `cancelled` flag
    wouldn't work — the reader thread would still block on readline()
    until the server finishes its current token.
    """

    def __init__(self, messages: list, max_tokens: int,
                 sampler: dict | None = None, temperature: float | None = None):
        self.body = _chat_body(messages, max_tokens, stream=True,
                               temperature=temperature, sampler=sampler)
        self.conn: http.client.HTTPConnection | None = None
        self.cancelled = False
        self.prompt_tokens: int | None = None  # real count, from usage chunk

    def run(self, on_delta) -> None:
        """Block while reading the SSE stream. Call from a thread; call
        cancel() from another thread to abort.

        on_delta(text) is invoked for each content delta — it's the
        caller's job to marshal it onto the event loop.
        """
        # self.conn is published BEFORE the request is sent, so a cancel()
        # landing mid-upload still tears the socket down.
        self.conn = _connect(timeout=300)
        self.conn.request("POST", "/v1/chat/completions",
                          json.dumps(self.body),
                          {"Content-Type": "application/json"})
        resp = self.conn.getresponse()
        if resp.status != 200:
            body = resp.read()[:300]
            self.conn.close()
            raise RuntimeError(
                f"llama-server HTTP {resp.status}: {body!r}"
            )
        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if usage and usage.get("prompt_tokens"):
                    self.prompt_tokens = usage["prompt_tokens"]
                choices = chunk.get("choices") or []
                text = None
                if choices:
                    delta = choices[0].get("delta") or {}
                    if isinstance(delta, dict):
                        text = delta.get("content")
                if text:
                    on_delta(text)
        except Exception as e:
            # Any failure here means the stream is dead — including
            # http.client's own cleanup racing a cancel() from another
            # thread (which can raise AttributeError from _close_conn).
            # Truncation is normal on abort; a genuinely dead server
            # surfaces on the next request.
            if not self.cancelled:
                print(f"LLM stream ended early: {type(e).__name__}: {e}")
        finally:
            try:
                self.conn.close()
            except OSError:
                pass

    def cancel(self) -> None:
        """Abort generation server-side. Safe to call from any thread."""
        self.cancelled = True
        try:
            if self.conn and self.conn.sock:
                self.conn.sock.shutdown(socket.SHUT_RDWR)
            if self.conn:
                self.conn.close()
        except OSError:
            pass


def prime_cache(messages: list) -> None:
    """Fire-and-discard request that pushes a prompt prefix (camera frame,
    speech chunks) through llama-server's cache while the user is talking.
    Content must be media-only appends — a trailing text block would
    diverge the prefix and kill reuse.

    Failure is not worth reporting: the turn still works, it just pays
    full prefill. Run in an executor so it never blocks the turn loop.
    """
    t0 = time.time()
    try:
        chat_blocking(messages, max_tokens=1)
        print(f"⚡ Primed cache ({time.time() - t0:.2f}s)")
    except Exception as e:
        # Swallowed on purpose — priming is best-effort.
        print(f"⚡ Cache priming failed (non-fatal): {e}")