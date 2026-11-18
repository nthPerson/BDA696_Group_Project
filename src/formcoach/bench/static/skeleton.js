// Skeleton geometry (pure) and drawing (canvas). MediaPipe-33 landmark indices; "left" and
// "right" are the person's own sides. Landmarks are [x, y, visibility] normalised to the image.

// Same 12 edges as formcoach/app/overlay.py EDGES (names: formcoach/pose/skeletons.py).
// 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips, 25/26 knees, 27/28 ankles (left, right).
export const EDGES = [
  [11, 13], [13, 15], [12, 14], [14, 16], [11, 12], [23, 24],
  [11, 23], [12, 24], [23, 25], [25, 27], [24, 26], [26, 28],
];

const MIN_VISIBILITY = 0.5;
const LEFT_ARM = [[11, 13], [13, 15]];
const RIGHT_ARM = [[12, 14], [14, 16]];

/**
 * Landmarks in pixels. Points with visibility < 0.5 (or missing) are dropped; each kept point
 * carries `i`, its landmark index, and `edges` holds [i, j] landmark indices with both ends kept.
 * `mirror` flips x like a mirror: x = (1 - x) * width.
 * @returns {{points: {i:number, x:number, y:number, v:number}[], edges: number[][]}}
 */
export function layout(lm, width, height, { mirror = false } = {}) {
  const points = [];
  (lm ?? []).forEach((p, i) => {
    if (!p || p[0] == null || p[1] == null || p[2] == null || p[2] < MIN_VISIBILITY) return;
    points.push({ i, x: (mirror ? 1 - p[0] : p[0]) * width, y: p[1] * height, v: p[2] });
  });
  const kept = new Set(points.map((p) => p.i));
  return { points, edges: EDGES.filter(([a, b]) => kept.has(a) && kept.has(b)) };
}

/**
 * Draw the skeleton on a 2D canvas context (no text). An arm whose entry in `angles`
 * ({l, r}, degrees) is a number is drawn in the accent colour.
 */
export function drawSkeleton(ctx, lm, width, height, { mirror = false, angles = null } = {}) {
  const { points, edges } = layout(lm, width, height, { mirror });
  const at = new Map(points.map((p) => [p.i, p]));
  const hot = new Set();
  if (Number.isFinite(angles?.l)) LEFT_ARM.forEach((e) => hot.add(e.join()));
  if (Number.isFinite(angles?.r)) RIGHT_ARM.forEach((e) => hot.add(e.join()));
  ctx.lineCap = "round";
  for (const e of edges) {
    const [a, b] = [at.get(e[0]), at.get(e[1])];
    ctx.strokeStyle = hot.has(e.join()) ? "#7aa7ff" : "#c9c9d3";
    ctx.lineWidth = hot.has(e.join()) ? 5 : 3;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  ctx.fillStyle = "#ececf1";
  for (const p of points) {
    if (p.i < 11) continue; // face points are not drawn
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}
