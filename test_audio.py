"""
test_audio.py — проверка нативного аудио-входа Gemma 4.
Генерирует речь через Silero TTS -> WAV 16kHz mono -> input_audio -> llama-server.
Запускать в .venv проекта (там где torch/numpy).
"""
import base64
import io
import json
import urllib.error
import urllib.request
import wave

import numpy as np

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "gemma-4-E2B-it-qat"
API_KEY = "no-key"


def make_test_wav() -> bytes:
    """Синтезирует тестовую фразу через Silero и ресемплит в 16kHz mono."""
    try:
        import tts_silero
        print("   [Silero] синтезирую тестовую фразу...")
        backend = tts_silero.load(model_id="v5_5_ru", speaker="xenia", sample_rate=24000)
        pcm = np.asarray(backend.generate("Привет! Скажи, сколько будет два плюс два?"), dtype=np.float32)
        # 24kHz -> 16kHz (линейно, для теста достаточно)
        new_len = max(1, int(round(len(pcm) * 16000 / 24000)))
        x_old = np.linspace(0.0, 1.0, num=len(pcm), dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=new_len, dtype=np.float32)
        pcm16 = np.interp(x_new, x_old, pcm).astype(np.float32)
    except Exception as exc:
        print(f"   ⚠️ Silero недоступен ({exc}) — ставлю синус 440Гц (речи не будет)")
        t = np.linspace(0.0, 2.0, 16000 * 2, dtype=np.float32)
        pcm16 = 0.3 * np.sin(2 * np.pi * 440 * t)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes((pcm16 * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def main():
    print("🎙 Генерация тестового аудио...")
    wav_bytes = make_test_wav()
    b64 = base64.b64encode(wav_bytes).decode()
    print(f"   WAV: {len(wav_bytes)} bytes | base64: {len(b64)} chars")

    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                {"type": "text", "text": "Что сказано в этом аудио? Ответь кратко одним предложением."},
            ],
        }],
        "temperature": 0.1,
        "max_tokens": 200,
        "stream": False,
    }

    print("📤 Отправка в llama-server (input_audio)...")
    req = urllib.request.Request(
        LLAMA_URL, data=json.dumps(payload).encode("utf-8"), method="POST"
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {API_KEY}")

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        print("\n✅ ОТВЕТ МОДЕЛИ:")
        print("   " + text.strip())
        print("\n🎉 АУДИО-ВХОД РАБОТАЕТ — модель услышала речь. Можно строить пайплайн в 1 этап.")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", "replace")
        print(f"\n❌ HTTP {e.code}: {err[:1000]}")
        low = err.lower()
        if "input_audio" in low or "audio" in low or "unsupported" in low:
            print("\n→ llama-server не принял input_audio.")
            print("  Причины: старая версия llama.cpp (аудио Gemma 4 добавлено в апреле 2026)")
            print("  ИЛИ в mmproj нет аудио-энкодера (только vision).")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    main()