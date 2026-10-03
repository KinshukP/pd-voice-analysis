import { useNavigate } from "react-router-dom";
import Layout from "../components/Layout.jsx";
import { MIN_TAKES, RECOMMENDED, TAKE_SECONDS } from "../lib/config.js";

const STEPS = [
  ["A quick question", "Your sex sets the vocal-tract model used to measure you."],
  ["Hold a note", `${MIN_TAKES}–${RECOMMENDED} takes of a steady “ahhh”, ${TAKE_SECONDS} seconds each. Or upload files.`],
  ["See what it found", "A clear result, the reasoning behind it, and which takes it used."],
];

export default function Start() {
  const navigate = useNavigate();
  return (
    <Layout>
      <h1>Your voice may know<br />before you do.</h1>
      <p className="sub">
        Parkinson's can reshape a voice long before anything else gives it away — the
        steadiness of a held note, the texture underneath it. This takes about two minutes.
      </p>

      <div className="card">
        <ul className="steplist">
          {STEPS.map(([title, body], i) => (
            <li key={title}>
              <span className="n">{i + 1}</span>
              <span><b>{title}</b><span className="t">{body}</span></span>
            </li>
          ))}
        </ul>
      </div>

      <div className="row" style={{ justifyContent: "center" }}>
        <button className="primary big" onClick={() => navigate("/sex")}>
          Get started!
        </button>
      </div>

      <p className="disclaimer" style={{ marginTop: 32 }}>
        This is a research prototype. It estimates acoustic patterns statistically associated
        with Parkinson's disease in one research dataset — it cannot diagnose anything, it has
        not been clinically validated, and “no strong signs” does not rule Parkinson's out.
        Discuss any concern about your health with a clinician.
      </p>
    </Layout>
  );
}
