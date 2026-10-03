// Backend origin. Override without editing this file by appending ?api=... ,
// e.g. http://localhost:5173/?api=http://127.0.0.1:8001
export const API =
  new URLSearchParams(location.search).get("api") || "http://127.0.0.1:8000";

export const TAKE_SECONDS = 5;
export const TARGET_RATE = 44100;
export const MIN_TAKES = 5;
export const RECOMMENDED = 7;
