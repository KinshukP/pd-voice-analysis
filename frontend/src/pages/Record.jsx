import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import Layout from "../components/Layout.jsx";
import Countdown from "../components/Countdown.jsx";
import { useSession } from "../state/session.jsx";
import { recordTake, Cancelled } from "../lib/audio.js";
import { screen } from "../lib/api.js";
import { API, MIN_TAKES, RECOMMENDED, TAKE_SECONDS } from "../lib/config.js";

export default function Record() {
  const { gender, takes, addTake, removeTake, urlFor, markAnalysed, result, stale, forced } =
    useSession();
  const navigate = useNavigate();

  const [countdown, setCountdown] = useState(null);   // null while not recording
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const cancelled = useRef(false);
  const fileInput = useRef(null);

  async function record() {
    setError("");
    cancelled.current = false;
    try {
      const { blob, warning } = await recordTake(
        (s) => setCountdown((prev) => ({ ...prev, ...s })),
        () => cancelled.current
      );
      addTake(`take_${takes.length + 1}.wav`, blob, warning, "recorded");
    } catch (e) {
      if (!(e instanceof Cancelled)) setError(`Microphone unavailable: ${e.message}`);
    } finally {
      setCountdown(null);
    }
  }

  async function analyse() {
    setBusy(true); setError("");
    try {
      const res = await screen({ takes, gender, forced: forced.current });
      if (res.ok) { markAnalysed(res.data); navigate("/results"); }
      else setError(res.error);
    } catch (e) {
      setError(`Could not reach the analysis service (${e.message}). Is the backend running on ${API}?`);
    } finally {
      setBusy(false);
    }
  }

  const enough = takes.length >= MIN_TAKES;
  const remaining = MIN_TAKES - takes.length;
  const label = result && stale ? "Reanalyse" : result ? "See my result" : "Analyse recordings";

  return (
    <Layout step={2} showStartOver>
      {countdown && <Countdown state={countdown} onCancel={() => { cancelled.current = true; }} />}

      <h2 className="page">Hold a steady note</h2>
      <p className="sub">
        Say “ahhh” at a constant pitch and volume for the full {TAKE_SECONDS} seconds, somewhere
        quiet. Keep the same microphone, distance and room for every take — switching setups
        partway makes them harder to compare.
      </p>

      <div className="card">
        <div className="rec-cta">
          <button className="rec-btn" onClick={record} disabled={busy} aria-label="Record a take">
            <span className="dot" />
          </button>
          <span className="count">
            {takes.length === 0
              ? `Record ${MIN_TAKES} to get started · ${RECOMMENDED} recommended`
              : enough
                ? `${takes.length} takes — enough to analyse`
                : `${takes.length} of ${MIN_TAKES} · ${remaining} more to go`}
          </span>
          <button className="ghost" onClick={() => fileInput.current.click()} disabled={busy}>
            or upload audio files
          </button>
          <input
            ref={fileInput} type="file" accept="audio/*" multiple hidden
            onChange={(e) => {
              [...e.target.files].forEach((f) => addTake(f.name, f, null, "uploaded"));
              e.target.value = "";
            }}
          />
        </div>

        {takes.length > 0 && (
          <ul className="takes">
            {takes.map((t) => (
              <li key={t.id}>
                <span className="take-badge">{t.source === "recorded" ? "rec" : "file"}</span>
                <span className="grow" style={t.warning ? { color: "var(--warning)" } : undefined}>
                  {t.name}{t.warning ? ` — ${t.warning}` : ""}
                </span>
                <audio controls src={urlFor(t)} />
                <button className="x" title="Remove" onClick={() => removeTake(t.id)}>✕</button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="row">
        <button className="primary big" onClick={analyse} disabled={!enough || busy}>
          {busy ? "Analysing…" : label}
        </button>
        {stale && <span className="count">recordings changed since the last result</span>}
        {result && !stale && (
          <button className="ghost pushright" onClick={() => navigate("/results")}>
            Skip to result →
          </button>
        )}
      </div>

      {error && <p className="err">{error}</p>}
    </Layout>
  );
}
