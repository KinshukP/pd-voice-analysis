import { API } from "./config.js";

/** POST the takes. Returns {ok, data} rather than throwing on a 4xx, because a
 *  422 carries an actionable message the user needs to see. */
export async function screen({ takes, gender, forced }) {
  const fd = new FormData();
  takes.forEach((t) => fd.append("files", t.blob, t.name));
  takes.forEach((t) => fd.append("sources", t.source || "uploaded"));
  fd.append("gender", gender);
  forced.forEach((n) => fd.append("force_include", n));

  const res = await fetch(`${API}/screen`, { method: "POST", body: fd });
  const data = await res.json();
  if (res.ok) return { ok: true, data };
  const d = data.detail;
  return { ok: false, error: typeof d === "string" ? d : JSON.stringify(d) };
}
