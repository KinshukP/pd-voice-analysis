import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Layout from "../components/Layout.jsx";
import DivergingChart from "../components/DivergingChart.jsx";
import { useSession } from "../state/session.jsx";
import { screen } from "../lib/api.js";

const ICON = {
  pd: (
    <svg className="ico" width="19" height="19" viewBox="0 0 20 20" fill="none"
         stroke="var(--critical)" strokeWidth="1.7">
      <path d="M10 2.6 18.4 17H1.6L10 2.6Z" strokeLinejoin="round" />
      <path d="M10 8v3.6" strokeLinecap="round" />
      <circle cx="10" cy="14.4" r=".9" fill="var(--critical)" stroke="none" />
    </svg>
  ),
  unc: (
    <svg className="ico" width="19" height="19" viewBox="0 0 20 20" fill="none"
         stroke="var(--warning)" strokeWidth="1.7">
      <circle cx="10" cy="10" r="8" />
      <path d="M7.7 7.6a2.35 2.35 0 1 1 3.1 2.23c-.5.18-.8.66-.8 1.19v.4" strokeLinecap="round" />
      <circle cx="10" cy="14.3" r=".9" fill="var(--warning)" stroke="none" />
    </svg>
  ),
  hc: (
    <svg className="ico" width="19" height="19" viewBox="0 0 20 20" fill="none"
         stroke="var(--good)" strokeWidth="1.7">
      <circle cx="10" cy="10" r="8" />
      <path d="m6.4 10.3 2.4 2.4 4.8-5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
};

const KEY = { likely_pd: "pd", uncertain: "unc", likely_healthy: "hc" };
const BAND = {
  pd: ["v-pd", "Signs associated with Parkinson's detected", "A statistical signal only — this is not a diagnosis."],
  hc: ["v-hc", "No strong signs detected", "This does not rule Parkinson's out."],
  unc: ["v-unc", "Inconclusive — flagged for follow-up", "The recordings were borderline, so no confident call was made."],
};

// Always two decimals: small values collapse to "0%" without them, and large
// ones were rounding to a bare integer.
const pct = (v) => (v * 100).toFixed(2) + "%";
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

/**
 * "Download PDF" is the browser's own print pipeline, not a client-side
 * renderer: `window.print()` against a `@media print` stylesheet that hides
 * the controls and forces the light palette. That is what makes the PDF an
 * exact match for the results page instead of a second implementation of it
 * that can drift out of sync -- it prints the same DOM the screen shows.
 *
 * The only extra step is the suggested filename: browsers seed the "Save as
 * PDF" filename from document.title, so it's swapped for the duration of the
 * print and put back after. `afterprint` covers the normal case; the timeout
 * is a backup for browsers that fire it unreliably from a script-triggered
 * print.
 */
function downloadPdf() {
  const previous = document.title;
  document.title = `voice-screening-result-${new Date().toISOString().slice(0, 10)}`;
  const restore = () => { document.title = previous; };
  addEventListener("afterprint", restore, { once: true });
  setTimeout(restore, 2000);
  print();
}

export default function Results() {
  const { result: r, gender, takes, markAnalysed, forced } = useSession();
  const navigate = useNavigate();
  const [showTable, setShowTable] = useState(false);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  // Re-run with a changed override set, staying on this page.
  async function rerun(add, remove) {
    if (add) forced.current.add(add); else forced.current.delete(remove);
    setBusy(add || remove); setError("");
    try {
      const res = await screen({ takes, gender, forced: forced.current });
      if (res.ok) markAnalysed(res.data); else setError(res.error);
    } catch (e) {
      setError(`Could not reach the analysis service (${e.message}).`);
    } finally {
      setBusy("");
    }
  }

  const key = KEY[r.result];
  const band = BAND[key];
  const concepts = r.concept_breakdown.filter((c) => c.narrated);
  const adj = r.adjusted_probability;
  const d = r.confidence_detail;

  return (
    <Layout step={3} showStartOver>
      <div className="row no-print" style={{ justifyContent: "flex-end", marginBottom: 14 }}>
        <button className="ghost" onClick={downloadPdf}>
          <span aria-hidden="true">⬇</span> Download PDF
        </button>
      </div>

      <div className={`verdict ${band[0]}`}>
        {ICON[key]}
        <div><h3>{band[1]}</h3><p>{band[2]}</p></div>
      </div>

      <div className="lean">
        {cap(r.evidence.lean)} — <b>{r.evidence.strength}</b> evidence.
      </div>

      <div className="statgrid">
        <div className="stat">
          <div className="lab">Confidence in this result</div>
          <div className="hero">{r.confidence_pct}<span className="hero-unit">%</span></div>
          <div className="meter"><div style={{ width: `${Math.max(r.confidence_pct, 1.5)}%` }} /></div>
          <span className="tag">{r.confidence_level}</span>
        </div>
        <div className="stat">
          <div className="lab">Estimated chance of Parkinson's</div>
          <div className="hero">{pct(adj.probability)}</div>
          <p className="hint" style={{ marginTop: 10 }}>
            Confidence is how strong the evidence is — <b>not</b> a chance of having
            Parkinson's. This is that figure.
          </p>
        </div>
      </div>

      <div className="facets">
        <div className="facet">
          <div className="lab">
            <span>Strength of evidence</span><span>{Math.round(d.evidence_strength_norm * 100)}%</span>
          </div>
          <div className="meter">
            <div style={{ width: `${Math.max(d.evidence_strength_norm * 100, 1.5)}%` }} />
          </div>
          <p className="hint">How far this voice sits from one that says nothing either way.</p>
        </div>
        <div className="facet">
          <div className="lab">
            <span>Agreement between takes</span><span>{Math.round(d.spread_factor * 100)}%</span>
          </div>
          <div className="meter">
            <div style={{ width: `${Math.max(d.spread_factor * 100, 1.5)}%` }} />
          </div>
          <p className="hint">How closely your recordings matched each other.</p>
        </div>
      </div>

      <h4>What contributed</h4>
      <figure>
        <figcaption>Each measured pattern's pull on the result, relative to the model's baseline.</figcaption>
        <div className="legend">
          <span><i style={{ background: "var(--pos)" }} />toward Parkinson's</span>
          <span><i style={{ background: "var(--neg)" }} />toward healthy</span>
        </div>
        <DivergingChart items={concepts} />
      </figure>

      <h4>Reading</h4>
      <p style={{ fontSize: ".93rem", margin: 0, whiteSpace: "pre-line" }}>{r.explanation_text}</p>

      <h4>Recordings used</h4>
      <p className="count" style={{ margin: "0 0 9px" }}>
        {r.takes.n_used} of {r.takes.n_given} combined
        {r.spread.inconsistent ? " · results varied between takes" : ""}
      </p>
      <div className="chips">
        {r.takes.used.map((u) => {
          const k = KEY[u.result];
          const col = { pd: "var(--critical)", unc: "var(--warning)", hc: "var(--good)" }[k];
          const word = { pd: "signs", unc: "borderline", hc: "no signs" }[k];
          return (
            <span className="chip" key={u.wav}>
              <span className="sw" style={{ background: col }} />
              {u.wav} <span style={{ color: "var(--muted)" }}>· {word}</span>
            </span>
          );
        })}
      </div>

      {r.takes.excluded.map((e) => (
        <div className="excl" key={e.wav}>
          <b>{e.wav}</b> excluded <span className="why">— {e.reason}</span>
          <button className="override no-print" disabled={!!busy} onClick={() => rerun(e.wav, null)}>
            {busy === e.wav ? "including…" : "Include anyway"}
          </button>
        </div>
      ))}
      {(r.takes.forced || []).map((f) => (
        <div className="excl" key={f.wav}>
          <b>{f.wav}</b>
          <span className="why"> — included by override, despite failing the quality checks</span>
          <button className="override undo no-print" disabled={!!busy} onClick={() => rerun(null, f.wav)}>
            {busy === f.wav ? "excluding…" : "Exclude again"}
          </button>
        </div>
      ))}
      {r.takes.failed.map((e) => (
        <div className="excl" key={e.wav}>
          <b>{e.wav}</b>
          <span className="why"> — could not be analysed, so it cannot be included</span>
        </div>
      ))}
      {r.spread.warning && <p className="hint" style={{ color: "var(--warning)" }}>{r.spread.warning}</p>}
      {error && <p className="err">{error}</p>}

      <p className="no-print" style={{ margin: "20px 0 0" }}>
        <button className="linkish" onClick={() => setShowTable((v) => !v)}>
          {showTable ? "Hide the table" : "Show the numbers as a table"}
        </button>
      </p>
      {showTable && (
        <div>
          <table>
            <thead>
              <tr><th>Measured pattern</th><th>Direction</th><th className="num">Influence</th></tr>
            </thead>
            <tbody>
              {concepts.map((c) => (
                <tr key={c.concept}>
                  <td>{c.label}</td><td>{c.direction}</td>
                  <td className="num">
                    {(c.contribution > 0 ? "+" : "−") + Math.abs(c.contribution).toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <table>
            <thead><tr><th>Recording</th><th>Outcome</th></tr></thead>
            <tbody>
              {r.takes.used.map((u) => (
                <tr key={u.wav}><td>{u.wav}</td><td>{u.result.replace(/_/g, " ")}</td></tr>
              ))}
              {r.takes.excluded.map((e) => (
                <tr key={e.wav}><td>{e.wav}</td><td>excluded — {e.kind.replace(/_/g, " ")}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="hint">
            Influence values are the model's internal contribution scores; only their sign and
            relative size are meaningful.
          </p>
        </div>
      )}

      <div className="row no-print" style={{ marginTop: 28 }}>
        <button onClick={() => navigate("/record")}>Edit recordings</button>
        <span className="count">Add, remove or re-record takes, then analyse again.</span>
      </div>

      <p className="disclaimer">
        This tool estimates acoustic patterns statistically associated with Parkinson's
        disease in one research dataset. It cannot diagnose anything, it has not been
        clinically validated, and “no strong signs” does not rule Parkinson's out. Discuss
        any concern about your health with a clinician.
      </p>
    </Layout>
  );
}
