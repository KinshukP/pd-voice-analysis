import { useEffect, useRef } from "react";

const CIRC = 2 * Math.PI * 54;

/**
 * Full-screen countdown. Driven entirely by `state` from the recorder, so the
 * audio code owns the timing and this owns only the drawing.
 */
export default function Countdown({ state, onCancel }) {
  const numRef = useRef(null);
  const digit = state?.digit;

  // Replay the pop on every digit change. The class has to come off and the
  // layout be read back before it goes on again, or the animation won't re-fire.
  useEffect(() => {
    const el = numRef.current;
    if (!el) return;
    el.classList.remove("pop");
    void el.offsetWidth;
    el.classList.add("pop");
  }, [digit]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onCancel(); };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [onCancel]);

  if (!state) return null;
  const live = state.phase === "recording";

  return (
    <div className="overlay" role="status" aria-live="polite">
      <div className="dial">
        <svg viewBox="0 0 120 120" aria-hidden="true">
          <circle className="ring-bg" cx="60" cy="60" r="54" />
          <circle
            className={`ring-fg${live ? " live" : ""}`}
            cx="60" cy="60" r="54"
            strokeDasharray={CIRC}
            strokeDashoffset={CIRC * (1 - (state.frac ?? 0))}
          />
        </svg>
        <div className="ov-num" ref={numRef}>{digit}</div>
      </div>
      <div className="ov-cue">{live ? "Say “ahhh”" : "Get ready"}</div>
      <div className="ov-sub">
        {live ? "Hold it steady — same pitch, same volume"
              : "Breathe in — you'll hold one steady note"}
      </div>
      <button className="ov-cancel" onClick={onCancel}>
        Cancel <span className="kbd">Esc</span>
      </button>
    </div>
  );
}
