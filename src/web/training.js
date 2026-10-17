// Training curves for our in-house ResEnc-M, read from /static/training_curves.json
// (exported from the real training log by scripts/export_training_curves.py).
const NS = 'http://www.w3.org/2000/svg';
const $ = id => document.getElementById(id);

export function niceTicks(min, max, count = 4) {
  if (!(max > min)) return [min];
  const raw = (max - min) / count, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(6));
  return ticks;
}

function lineChart(container, points, { yMin, yMax, color, yFormat = v => v, xLabel = 'step', label }) {
  container.replaceChildren();
  if (!points.length) { container.textContent = 'No data logged.'; return; }
  const W = 320, H = 120, L = 34, R = 8, T = 8, B = 22;
  const xs = points.map(p => p[0]), xMax = Math.max(...xs), xMin = 0;
  const x = v => L + (W - L - R) * (v - xMin) / (xMax - xMin || 1);
  const y = v => T + (H - T - B) * (1 - (v - yMin) / (yMax - yMin || 1));
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('width', '100%');
  const el = (tag, attrs) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); svg.append(n); return n; };
  for (const v of niceTicks(yMin, yMax)) {
    el('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), class: 'lc-grid' });
    el('text', { x: L - 5, y: y(v) + 3, 'text-anchor': 'end', class: 'lc-axis' }).textContent = yFormat(v);
  }
  for (const v of niceTicks(0, xMax, 4)) {
    el('line', { x1: x(v), x2: x(v), y1: T, y2: H - B, class: 'lc-grid' });
    el('text', { x: x(v), y: H - 7, 'text-anchor': 'middle', class: 'lc-axis' }).textContent = v >= 1000 ? `${v / 1000}k` : v;
  }
  el('line', { x1: L, x2: W - R, y1: H - B, y2: H - B, class: 'lc-baseline' });
  el('path', { d: points.map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(''), fill: 'none', stroke: color, 'stroke-width': 1.6, 'stroke-linejoin': 'round' });
  // hover: crosshair + tooltip with the nearest logged point
  const cross = el('line', { y1: T, y2: H - B, class: 'lc-cross', visibility: 'hidden' });
  const dot = el('circle', { r: 3, fill: color, visibility: 'hidden' });
  const hit = el('rect', { x: L, y: T, width: W - L - R, height: H - T - B, fill: 'transparent' });
  const tip = document.querySelector('.chart-tooltip');
  hit.addEventListener('mousemove', e => {
    const box = svg.getBoundingClientRect(), vx = (e.clientX - box.left) / box.width * W;
    const p = points.reduce((a, b) => (Math.abs(x(b[0]) - vx) < Math.abs(x(a[0]) - vx) ? b : a));
    cross.setAttribute('x1', x(p[0])); cross.setAttribute('x2', x(p[0])); cross.setAttribute('visibility', 'visible');
    dot.setAttribute('cx', x(p[0])); dot.setAttribute('cy', y(p[1])); dot.setAttribute('visibility', 'visible');
    if (tip) { tip.textContent = `${label} at ${xLabel} ${p[0]}: ${yFormat(p[1], true)}`; tip.hidden = false;
      tip.style.left = `${Math.min(innerWidth - tip.offsetWidth - 8, e.clientX + 12)}px`; tip.style.top = `${e.clientY - tip.offsetHeight - 10}px`; }
  });
  hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); if (tip) tip.hidden = true; });
  container.append(svg);
}

export async function renderTrainingCurves() {
  let data;
  try {
    const response = await fetch('/static/training_curves.json', { cache: 'no-cache' });
    if (!response.ok) throw new Error();
    data = await response.json();
  } catch { $('training-note').textContent = 'Training log not available.'; return; }
  const loss = data.loss.map(r => [r.iteration, r.loss]);
  const dice = data.val.map(r => [r.iteration, r.mean]);
  lineChart($('chart-train-loss'), loss, { yMin: 0, yMax: Math.max(1, ...loss.map(p => p[1])), color: '#3b93c9',
    yFormat: (v, exact) => exact ? v.toFixed(3) : v.toFixed(1), label: 'train_loss' });
  lineChart($('chart-val-dice'), dice, { yMin: 0, yMax: 1, color: '#b5862a',
    yFormat: (v, exact) => exact ? v.toFixed(3) : v.toFixed(1), label: 'val_mean_dice' });
  const last = data.val.at(-1);
  $('training-note').textContent = `${data.description} Final validation Dice: WT ${last.WT.toFixed(3)} · TC ${last.TC.toFixed(3)} · ET ${last.ET.toFixed(3)} (mean ${last.mean.toFixed(3)}).`;
}
