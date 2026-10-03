import { TAKE_SECONDS, TARGET_RATE } from "./config.js";

/* MediaRecorder emits compressed webm/opus, and lossy compression alters the
   fine spectral detail these features measure -- so raw samples are captured
   through Web Audio and written as uncompressed 16-bit PCM. */
export function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2), view = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  str(36, "data"); view.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (const s of samples) {
    const c = Math.max(-1, Math.min(1, s));
    view.setInt16(o, c < 0 ? c * 0x8000 : c * 0x7fff, true); o += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}

/* Cancellation. `sweep` resolves false when a cancel lands mid-animation, and
   each phase turns that into a throw, so the one `finally` below owns every
   teardown path -- completed, cancelled, or failed. */
export class Cancelled extends Error {}

function sweep(ms, from, to, onFrac, isCancelled) {
  return new Promise((done) => {
    const t0 = performance.now();
    (function step(now) {
      if (isCancelled()) return done(false);
      const k = Math.min((now - t0) / ms, 1);
      onFrac(from + (to - from) * k);
      k < 1 ? requestAnimationFrame(step) : done(true);
    })(t0);
  });
}

/**
 * Capture one take. Reports progress through `onState` so React can render the
 * countdown; the audio graph itself stays imperative because that is what the
 * Web Audio API is.
 *
 * @param onState {(s:{phase:"ready"|"recording", digit:number, frac:number}) => void}
 * @param isCancelled {() => boolean}
 * @returns {Promise<{blob:Blob, peak:number, warning:string|null}>}
 */
export async function recordTake(onState, isCancelled) {
  // The browser's default voice processing is tuned for intelligibility and
  // reshapes exactly what is being measured here -- noise suppression alters
  // harmonics-to-noise, auto gain flattens shimmer -- so all of it is off.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  const ctx = new AudioContext({ sampleRate: TARGET_RATE });
  const src = ctx.createMediaStreamSource(stream);
  const node = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  let capturing = false;
  node.onaudioprocess = (e) => {
    if (capturing) chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  src.connect(node); node.connect(ctx.destination);

  const rate = ctx.sampleRate;          // read before the context is closed
  let cd = null, tick = null;
  try {
    // Both phases drain the same way -- the ring empties as the time runs out --
    // so it reads as one clock rather than restarting each second. Colour and
    // caption carry which phase you're in, not the direction of travel.
    let n = 3;
    onState({ phase: "ready", digit: 3, frac: 1 });
    cd = setInterval(() => { if (--n >= 1) onState({ phase: "ready", digit: n, frac: null }); }, 1000);
    if (!(await sweep(3000, 1, 0, (f) => onState({ phase: "ready", frac: f }), isCancelled)))
      throw new Cancelled();
    clearInterval(cd); cd = null;

    capturing = true;
    onState({ phase: "recording", digit: TAKE_SECONDS, frac: 1 });
    const t0 = performance.now();
    let shown = TAKE_SECONDS;
    tick = setInterval(() => {
      const left = Math.max(0, Math.ceil(TAKE_SECONDS - (performance.now() - t0) / 1000));
      if (left !== shown) { shown = left; onState({ phase: "recording", digit: left, frac: null }); }
    }, 80);
    if (!(await sweep(TAKE_SECONDS * 1000, 1, 0, (f) => onState({ phase: "recording", frac: f }), isCancelled)))
      throw new Cancelled();
  } finally {
    // Every exit runs through here: finished, cancelled, or thrown. A cancel
    // that left the microphone open would keep the browser's recording
    // indicator lit with nothing listening.
    capturing = false;
    if (cd) clearInterval(cd);
    if (tick) clearInterval(tick);
    node.disconnect(); src.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    ctx.close();
  }

  const total = chunks.reduce((n, c) => n + c.length, 0);
  const flat = new Float32Array(total);
  let o = 0; for (const c of chunks) { flat.set(c, o); o += c.length; }
  const peak = flat.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  return {
    blob: encodeWav(flat, rate),
    peak,
    warning: peak < 0.02 ? "very quiet — move closer"
           : peak > 0.99 ? "clipping — move back" : null,
  };
}
