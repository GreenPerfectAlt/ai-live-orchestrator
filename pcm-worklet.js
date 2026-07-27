/*
pcm-worklet.js — low-latency PCM16 ring-buffer player with chunk stitching. (v18-hardened)

Protocol (port.postMessage -> worklet):
  { type: "push",  pcm: Int16Array | Float32Array }  — append audio (use transferable!)
  { type: "clear" }                                  — 8ms fade-out, then flush queue
  { type: "duck",  level: 0..1 }                     — smooth volume dip (barge-in)
  { type: "unduck" }                                 — restore volume

Protocol (worklet -> port.postMessage):
  { type: "ready" }
  { type: "underrun", count }                        — buffer starvation, throttled 1/s
  { type: "stats", latencyMs, underruns, buffered }  — periodic latency telemetry (v18)

v18 changes:
  * RING_CAPACITY reduced to 1<<18 (was 1<<20) — tighter buffer, faster reaction
  * GAIN_STEP now sample-rate-relative (always ~10ms ramp)
  * Silent preroll cushion (PREROLL_SECONDS) at the start of a fresh playback
    session — absorbs startup jitter so the first syllable is not eaten.
    Applied only after hardClear (barge-in / fresh start), NOT after mid-turn
    underruns, so it never stacks latency within a phrase.
  * Periodic latency stats posted to the main thread for diagnostics.
*/

const RING_CAPACITY = 1 << 18;              // 262,144 samples ≈ 5.4s @ 48kHz (≈10.9s @ 24kHz)
const FADE_SECONDS = 0.008;                 // 8ms click-free fade on clear
const STITCH_SECONDS = 0.004;               // 4ms fade at chunk boundaries (kills inter-phrase clicks)
const PREROLL_SECONDS = 0.15;               // silent cushion at fresh-session start (parlor-jarvis pattern)
const GAIN_STEP = 1 / (sampleRate * 0.010); // ~10ms gain ramp for duck/unduck, sample-rate independent
const STATS_INTERVAL = 0.5;                 // seconds between latency stats reports

class PCMRingPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ring = new Float32Array(RING_CAPACITY);
    this.writePos = 0;
    this.readPos = 0;
    this.available = 0;
    this.gain = 1.0;
    this.targetGain = 1.0;
    this.fadeLeft = 0;
    this.fadeTotal = 1;
    this.playing = false;
    this.underruns = 0;
    this.lastReport = 0;
    this.lastStats = 0;
    this._prerolled = false;                // v18: fresh-session preroll flag
    this.port.onmessage = (event) => this.onMessage(event.data);
    this.port.postMessage({ type: "ready" });
  }

  onMessage(msg) {
    if (!msg || !msg.type) return;
    switch (msg.type) {
      case "push": this.pushPCM(msg.pcm); break;
      case "clear": this.startClearFade(); break;
      case "duck": {
        const level = typeof msg.level === "number" ? msg.level : 0.25;
        this.targetGain = Math.max(0, Math.min(1, level));
        break;
      }
      case "unduck": this.targetGain = 1.0; break;
    }
  }

  // Short fade-out on the old tail + fade-in on the new head at a chunk
  // boundary, so back-to-back TTS chunks join without a click.
  stitchAt(boundary, tailAvail, headLen) {
    const K = Math.max(2, Math.floor(sampleRate * STITCH_SECONDS));
    const tailK = Math.min(K, tailAvail);
    const headK = Math.min(K, headLen);
    for (let j = 0; j < tailK; j++) {
      const idx = (boundary - tailK + j + RING_CAPACITY) % RING_CAPACITY;
      this.ring[idx] *= (tailK - 1 - j) / (tailK - 1 || 1);
    }
    for (let j = 0; j < headK; j++) {
      const idx = (boundary + j) % RING_CAPACITY;
      this.ring[idx] *= j / (headK - 1 || 1);
    }
  }

  // v18: write a cushion of silence at the start of a fresh playback session.
  // Gives the pipeline a small buffer so the first real chunk is not starved
  // by startup jitter. Bounded to once per hardClear — never stacks mid-turn.
  _writePreroll() {
    const n = Math.floor(sampleRate * PREROLL_SECONDS);
    if (n <= 0) return;
    // Make room if needed
    if (this.available + n > RING_CAPACITY) {
      const drop = this.available + n - RING_CAPACITY;
      this.readPos = (this.readPos + drop) % RING_CAPACITY;
      this.available -= drop;
    }
    for (let i = 0; i < n; i++) {
      this.ring[this.writePos] = 0;
      this.writePos = (this.writePos + 1) % RING_CAPACITY;
    }
    this.available += n;
  }

  pushPCM(pcm) {
    if (!pcm || !pcm.length) return;
    if (this.fadeLeft > 0) this.hardClear();

    // Capture emptiness BEFORE preroll so the first real chunk is not stitched
    // against the silence cushion (a silence->audio join needs no crossfade).
    const wasEmpty = this.available === 0;

    // v18: silent preroll on fresh playback session only
    if (!this._prerolled) {
      this._writePreroll();
      this._prerolled = true;
    }

    const boundary = this.writePos;
    const availBefore = this.available;
    let n = pcm.length;
    let src = 0;
    const isInt16 = pcm instanceof Int16Array;

    if (this.available + n > RING_CAPACITY) {
      const drop = this.available + n - RING_CAPACITY;
      this.readPos = (this.readPos + drop) % RING_CAPACITY;
      this.available -= drop;
    }

    while (n > 0) {
      const space = RING_CAPACITY - this.writePos;
      const take = Math.min(space, n);
      if (isInt16) {
        for (let i = 0; i < take; i++) this.ring[this.writePos + i] = pcm[src + i] / 32768;
      } else {
        for (let i = 0; i < take; i++) this.ring[this.writePos + i] = pcm[src + i];
      }
      this.writePos = (this.writePos + take) % RING_CAPACITY;
      src += take; n -= take; this.available += take;
    }

    if (!wasEmpty && this.fadeLeft === 0) this.stitchAt(boundary, availBefore, pcm.length);
    this.playing = true;
  }

  startClearFade() {
    if (this.available <= 0) { this.hardClear(); return; }
    const fadeSamples = Math.max(16, Math.floor(sampleRate * FADE_SECONDS));
    this.fadeTotal = Math.min(fadeSamples, this.available);
    this.fadeLeft = this.fadeTotal;
  }

  hardClear() {
    this.readPos = 0; this.writePos = 0; this.available = 0;
    this.fadeLeft = 0; this.playing = false;
    this._prerolled = false;                // v18: next session gets a fresh preroll
  }

  process(inputs, outputs) {
    const out = outputs[0];
    const frame = out[0];
    if (!frame) return true;
    const need = frame.length;
    let written = 0;

    while (written < need && this.available > 0) {
      let sample = this.ring[this.readPos];
      if (this.fadeLeft > 0) {
        sample *= this.fadeLeft / this.fadeTotal;
        this.fadeLeft--;
        this.writeSample(out, written, sample);
        written++;
        this.readPos = (this.readPos + 1) % RING_CAPACITY;
        this.available--;
        if (this.fadeLeft === 0) { this.hardClear(); break; }
      } else {
        if (this.gain !== this.targetGain) {
          const diff = this.targetGain - this.gain;
          this.gain = Math.abs(diff) <= GAIN_STEP ? this.targetGain : this.gain + Math.sign(diff) * GAIN_STEP;
        }
        this.writeSample(out, written, sample * this.gain);
        written++;
        this.readPos = (this.readPos + 1) % RING_CAPACITY;
        this.available--;
      }
    }

    if (written < need) {
      for (let ch = 0; ch < out.length; ch++) out[ch].fill(0, written);
      if (this.playing) {
        this.playing = false; this.underruns++;
        const t = currentTime;
        if (t - this.lastReport > 1.0) { this.lastReport = t; this.port.postMessage({ type: "underrun", count: this.underruns }); }
      }
    }

    // v18: periodic latency telemetry for the main thread
    if (this.playing) {
      const t = currentTime;
      if (t - this.lastStats > STATS_INTERVAL) {
        this.lastStats = t;
        this.port.postMessage({
          type: "stats",
          latencyMs: Math.round((this.available / sampleRate) * 1000),
          underruns: this.underruns,
          buffered: this.available,
        });
      }
    }

    return true;
  }

  writeSample(out, index, sample) {
    for (let ch = 0; ch < out.length; ch++) out[ch][index] = sample;
  }
}

registerProcessor("pcm-ring-player", PCMRingPlayer);