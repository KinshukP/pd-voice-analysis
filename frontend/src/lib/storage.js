/* IndexedDB rather than localStorage: seven 5-second WAVs are ~3 MB, and
   base64-ing them into localStorage would blow its ~5 MB quota. IndexedDB
   stores Blobs natively, so the takes go in as-is.
   Every failure path is swallowed -- losing saved state is an inconvenience,
   never a reason to break the page (private-mode browsers refuse to open a
   database at all). */
const DB = "pdvoice", STORE = "state", KEY = "current";

function openDb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

async function withStore(mode, fn) {
  const db = await openDb();
  try {
    return await new Promise((res, rej) => {
      const tx = db.transaction(STORE, mode);
      const req = fn(tx.objectStore(STORE));
      // Resolve with the request's own result, or undefined for a miss. Falling
      // back to the request object itself would make "nothing saved" look truthy.
      tx.oncomplete = () => res(req && "result" in req ? req.result : undefined);
      tx.onerror = () => rej(tx.error);
    });
  } finally { db.close(); }
}

export async function saveState(state) {
  try { await withStore("readwrite", (s) => s.put(state, KEY)); } catch { /* storage unavailable */ }
}
export async function loadState() {
  try { return await withStore("readonly", (s) => s.get(KEY)); } catch { return null; }
}
export async function clearState() {
  try { await withStore("readwrite", (s) => s.delete(KEY)); } catch { /* ignore */ }
}
