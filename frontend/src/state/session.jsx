import { createContext, useContext, useEffect, useRef, useState, useCallback } from "react";
import { loadState, saveState, clearState } from "../lib/storage.js";

const Ctx = createContext(null);
export const useSession = () => useContext(Ctx);

/**
 * Everything the four pages share. Held in one place rather than passed down,
 * because the record and results pages both read and write most of it, and the
 * route can change under them at any point.
 *
 * Persisted to IndexedDB on every change so a refresh -- or landing straight on
 * a deep link -- is not silent data loss.
 */
export function SessionProvider({ children }) {
  const [ready, setReady] = useState(false);      // hydrated from IndexedDB yet?
  const [gender, setGenderRaw] = useState("");
  const [takes, setTakes] = useState([]);
  const [result, setResult] = useState(null);
  const [analysedSig, setAnalysedSig] = useState(null);

  // Object URLs for the audio players, keyed by take id. Kept in a ref rather
  // than state: creating one per render would leak a URL on every keystroke,
  // and they are not rendering data, they are a resource to clean up.
  const urls = useRef(new Map());

  // Filenames the user chose to include despite failing a quality check.
  // Deliberately not persisted -- an override should be a conscious act each
  // time, not a sticky setting that silently survives a reload.
  const forced = useRef(new Set());

  const seq = useRef(0);

  useEffect(() => {
    (async () => {
      const saved = await loadState();
      if (saved) {
        const t = saved.takes || [];
        setTakes(t);
        seq.current = saved.takeSeq || t.reduce((m, x) => Math.max(m, x.id || 0), 0);
        setGenderRaw(saved.gender || "");
        setResult(saved.lastResult ?? null);
        setAnalysedSig(saved.analysedSig ?? null);
      }
      setReady(true);
    })();
  }, []);

  // One writer for persistence, driven by the state it mirrors, so no call site
  // can forget to save.
  useEffect(() => {
    if (!ready) return;
    saveState({
      takeSeq: seq.current,
      analysedSig,
      lastResult: result,
      gender,
      takes: takes.map((t) => ({
        id: t.id, name: t.name, blob: t.blob, warning: t.warning, source: t.source,
      })),
    });
  }, [ready, takes, gender, result, analysedSig]);

  /* Identity of the thing that was analysed: which takes, and the sex setting
     (changing it changes the formant ceiling, so it changes the result). Ids are
     sequential rather than positional, so removing a take and adding an
     identical one still counts as a change. */
  const currentSig = useCallback(
    () => takes.map((t) => t.id).join(",") + "|" + gender,
    [takes, gender]
  );
  const stale = analysedSig !== null && analysedSig !== currentSig();

  const addTake = useCallback((name, blob, warning, source = "uploaded") => {
    setTakes((prev) => [...prev, { id: ++seq.current, name, blob, warning, source }]);
  }, []);

  const removeTake = useCallback((id) => {
    setTakes((prev) => prev.filter((t) => t.id !== id));
    const u = urls.current.get(id);
    if (u) { URL.revokeObjectURL(u); urls.current.delete(id); }
  }, []);

  const urlFor = useCallback((take) => {
    let u = urls.current.get(take.id);
    if (!u) { u = URL.createObjectURL(take.blob); urls.current.set(take.id, u); }
    return u;
  }, []);

  const setGender = useCallback((g) => setGenderRaw(g), []);

  const markAnalysed = useCallback((data) => {
    setResult(data);
    setAnalysedSig(takes.map((t) => t.id).join(",") + "|" + gender);
  }, [takes, gender]);

  /** Wipe everything and drop back to the sex question. */
  const startOver = useCallback(async () => {
    urls.current.forEach((u) => URL.revokeObjectURL(u));
    urls.current.clear();
    forced.current = new Set();
    seq.current = 0;
    setTakes([]); setResult(null); setAnalysedSig(null); setGenderRaw("");
    await clearState();
  }, []);

  return (
    <Ctx.Provider value={{
      ready, gender, setGender, takes, addTake, removeTake, urlFor,
      result, markAnalysed, setResult, stale, forced, startOver,
    }}>
      {children}
    </Ctx.Provider>
  );
}
