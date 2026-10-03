import { useNavigate } from "react-router-dom";
import { useSession } from "../state/session.jsx";

/** Progress bars for the three working steps. The start page passes step=0 to
 *  hide them -- there is nothing to be partway through yet. */
function Steps({ step }) {
  if (!step) return null;
  return (
    <div className="steps-row no-print" aria-label={`Step ${step} of 3`}>
      {[1, 2, 3].map((n) => (
        <i key={n} className={n === step ? "on" : n < step ? "done" : ""} />
      ))}
    </div>
  );
}

/**
 * Shell for all four pages. The header (logo + optional step row) is pinned to
 * the top and always centered; everything a page passes as `children` is
 * centered vertically in the remaining space. `safe center` (in CSS) is what
 * makes that work for a short page like Sex and a long, scrolling one like
 * Results alike -- it centers when the content fits, and falls back to
 * top-aligned rather than clipping when it doesn't.
 */
export default function Layout({ step = 0, showStartOver = false, children }) {
  const { startOver } = useSession();
  const navigate = useNavigate();

  return (
    <div className="wrap">
      <header className="topbar">
        {showStartOver && (
          <button
            className="ghost start-over no-print"
            onClick={async () => { await startOver(); navigate("/sex"); }}
          >
            Start over
          </button>
        )}
        <span className="brand">
          <span className="mark">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="#fff"
                 strokeWidth="1.9" strokeLinecap="round">
              <path d="M3 6.2v3.6M6 3.4v9.2M9 5v6M12 7v2" />
            </svg>
          </span>
          Parkinson's Disease Voice Analysis
        </span>
      </header>

      {/* Steps rides with the content, not the header -- it moves with
          .main-inner when the block is vertically centered, so it always sits
          directly above the page's own content rather than pinned to the top
          of the viewport. */}
      <main className="main">
        <div className="main-inner">
          <Steps step={step} />
          {children}
        </div>
      </main>
    </div>
  );
}
