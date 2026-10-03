import { HashRouter, Routes, Route, Navigate } from "react-router-dom";
import { SessionProvider, useSession } from "./state/session.jsx";
import Start from "./pages/Start.jsx";
import Sex from "./pages/Sex.jsx";
import Record from "./pages/Record.jsx";
import Results from "./pages/Results.jsx";

/**
 * Blocks a page whose data is missing rather than letting it crash. Reached by
 * a bookmark, a refresh, or the browser's back button after a Start over --
 * all of which are normal, so they redirect instead of erroring.
 *
 * Waits for `ready` first: the session hydrates from IndexedDB asynchronously,
 * and redirecting before that lands would bounce a returning user off their own
 * saved state.
 */
function Guard({ need, children }) {
  const { ready, gender, result } = useSession();
  if (!ready) return null;
  if (need === "gender" && !gender) return <Navigate to="/sex" replace />;
  if (need === "result" && !result) return <Navigate to="/record" replace />;
  return children;
}

export default function App() {
  return (
    <SessionProvider>
      {/* HashRouter, not BrowserRouter: this is served as static files, and
          hash routes work without a server-side rewrite for deep links. */}
      <HashRouter>
        <Routes>
          <Route path="/" element={<Start />} />
          <Route path="/sex" element={<Sex />} />
          <Route path="/record" element={<Guard need="gender"><Record /></Guard>} />
          <Route
            path="/results"
            element={
              <Guard need="gender">
                <Guard need="result"><Results /></Guard>
              </Guard>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </HashRouter>
    </SessionProvider>
  );
}
