/* Short axis labels. The backend's concept labels are written for prose and run
   to ~48 characters, which overflows the chart's label column -- measured, not
   guessed. The full label rides the tooltip and the table view instead. */
const SHORT = {
  temporal_fluctuation: "Energy fluctuation",
  spectral_shape: "Tone and timbre",
  vocal_tract: "Vocal-tract resonance",
  voice_regularity: "Pitch & loudness steadiness",
  other_spectral: "Other spectral detail",
};

export const shortLabel = (c) => SHORT[c.concept] || c.label.replace(/\s*\(.*\)$/, "");

/* Diverging bar: the job is polarity (above/below a zero baseline), so the
   validated blue<->red pair with a neutral midpoint. One SVG shares one zero
   axis; bars are 16px, rounded on the data-end only. Geometry was checked
   numerically for label overflow -- longest label 186px in a 196px column. */
export default function DivergingChart({ items }) {
  const W = 640, ROW = 46, PAD_T = 8;
  const H = items.length * ROW + PAD_T + 26;
  const LAB_W = 200, X0 = LAB_W + 24, X1 = W - 6;
  const ZERO = (X0 + X1) / 2, HALF = ZERO - X0 - 46;
  const max = Math.max(...items.map((d) => Math.abs(d.contribution)), 1e-6);

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="Influence of each measured pattern on the result"
    >
      <line className="c-axis" x1={ZERO} y1={PAD_T - 2} x2={ZERO} y2={PAD_T + items.length * ROW} />
      {items.map((d, i) => {
        const y = PAD_T + i * ROW + (ROW - 16) / 2;
        const w = Math.max((Math.abs(d.contribution) / max) * HALF, 2);
        const pos = d.contribution > 0;
        const x = pos ? ZERO : ZERO - w;
        const r = 4;
        // rounded on the data-end, square against the zero baseline
        const path = pos
          ? `M${x},${y} h${w - r} a${r},${r} 0 0 1 ${r},${r} v${16 - 2 * r} a${r},${r} 0 0 1 -${r},${r} H${x} Z`
          : `M${x + w},${y} H${x + r} a${r},${r} 0 0 0 -${r},${r} v${16 - 2 * r} a${r},${r} 0 0 0 ${r},${r} H${x + w} Z`;
        const tx = pos ? x + w + 8 : x - 8;
        return (
          <g key={d.concept}>
            <text className="c-label" x={LAB_W} y={y + 11.5} textAnchor="end">
              {shortLabel(d)}
            </text>
            <path d={path} fill={pos ? "var(--pos)" : "var(--neg)"}>
              <title>
                {`${d.label}: ${d.contribution > 0 ? "+" : ""}${d.contribution.toFixed(3)} (${d.direction})`}
              </title>
            </path>
            <text className="c-val" x={tx} y={y + 11.5} textAnchor={pos ? "start" : "end"}>
              {(d.contribution > 0 ? "+" : "−") + Math.abs(d.contribution).toFixed(2)}
            </text>
          </g>
        );
      })}
      <text className="c-tick" x={ZERO - 8} y={H - 8} textAnchor="end">← toward healthy</text>
      <text className="c-tick" x={ZERO + 8} y={H - 8}>toward Parkinson's →</text>
    </svg>
  );
}
