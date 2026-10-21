import { sampleWind, type WindGrid } from './wind-layer';

/**
 * OSIRIS — wind as streamlines.
 *
 * The wind is drawn the way God's Eye View (MIT, github.com/bilawalsidhu/gods-eye-view)
 * draws it: not particles that wander, but a fixed set of fine curves that
 * follow the wind, baked once from the forecast. Each curve is the path an air
 * parcel takes through eight hours of the field either side of its seed, and a
 * bright comet runs along it. Seeds are spread evenly over the sphere, so the
 * map reads the same everywhere and nothing piles up at the poles.
 *
 * This file only makes the curves; wind-gl-layer draws them.
 */

/** Curves on a desktop screen, and on a narrow one. */
export const WIND_PATH_LIMIT = 7200;
export const WIND_NARROW_PATH_LIMIT = 1200;
/** Steps either side of a seed, and the field time each one covers. */
export const WIND_PATH_STEPS = 16;
export const WIND_STEP_SECONDS = 1800;
/** Floats per vertex: mercator x, mercator y, how far along its curve (0..1), the curve's phase. */
export const WIND_STRIDE = 4;

const DEGREE_METERS = 111320;
const GOLDEN_ANGLE = 137.50776405003785;
const GOLDEN_RATIO = 0.6180339887498949;
/** Below this the air is called still, and a curve stops. */
const CALM = 0.15;
/** The map ends here (web mercator's edge), so the curves do too. */
const MAX_LAT = 85.0511;
/** Seeds stay a little inside that edge. */
const SEED_LAT = 85;

const normalizeLng = (lng: number) => ((((lng + 180) % 360) + 360) % 360) - 180;
const metersPerDegreeLng = (lat: number) => DEGREE_METERS * Math.cos((lat * Math.PI) / 180);

function mercatorX(lng: number): number {
  return (lng + 180) / 360;
}
function mercatorY(lat: number): number {
  return 0.5 - Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) / (2 * Math.PI);
}

/** One midpoint step along the wind, bounded so a fast wind near a pole cannot leap. Null where the curve must end. */
function advance(field: WindGrid, lng: number, lat: number, seconds: number): [number, number] | null {
  const here = sampleWind(field, lng, lat);
  if (!here || !(Math.hypot(here[0], here[1]) >= CALM)) return null;
  const lngRate = here[0] / metersPerDegreeLng(lat);
  const latRate = here[1] / DEGREE_METERS;
  const dt = Math.sign(seconds) * Math.min(Math.abs(seconds), 0.75 / Math.max(Math.abs(lngRate), Math.abs(latRate), 1e-12));
  const midLat = lat + latRate * dt * 0.5;
  if (Math.abs(midLat) > MAX_LAT) return null;
  const mid = sampleWind(field, lng + lngRate * dt * 0.5, midLat);
  if (!mid || !Number.isFinite(mid[0] + mid[1])) return null;
  const nextLng = normalizeLng(lng + (mid[0] * dt) / metersPerDegreeLng(midLat));
  const nextLat = lat + (mid[1] * dt) / DEGREE_METERS;
  // A curve ends at the map's seam, the poles and still air rather than wrap round.
  if (Math.abs(nextLat) > MAX_LAT || Math.abs(nextLng - lng) > 180) return null;
  if (Math.abs(nextLng - lng) + Math.abs(nextLat - lat) < 1e-8) return null;
  return [nextLng, nextLat];
}

export interface WindStreamlines {
  /** WIND_STRIDE floats per vertex, curve after curve. A curve's last vertex is followed by the next one's first, whose `along` is lower. */
  vertices: Float32Array;
  /** Vertices in use. */
  count: number;
  /** Curves. */
  paths: number;
}

/**
 * Curves through the wind field, from seeds spread evenly over the globe.
 * Increasing position along a curve always follows the wind.
 */
export function bakeWindStreamlines(
  field: WindGrid,
  { count = WIND_PATH_LIMIT, steps = WIND_PATH_STEPS, stepSeconds = WIND_STEP_SECONDS }: { count?: number; steps?: number; stepSeconds?: number } = {},
): WindStreamlines {
  const budget = Math.max(0, Math.floor(Number.isFinite(count) ? count : WIND_PATH_LIMIT));
  const half = Math.min(WIND_PATH_STEPS, Math.max(1, Math.floor(steps)));
  const out = new Float32Array(budget * (2 * half + 1) * WIND_STRIDE);
  const lngs: number[] = [];
  const lats: number[] = [];
  let n = 0;
  let paths = 0;
  for (let seed = 0; seed < budget; seed++) {
    const lng = normalizeLng(seed * GOLDEN_ANGLE);
    const lat = (Math.asin((((seed + 0.5) / budget) * 2 - 1) * Math.sin((SEED_LAT * Math.PI) / 180)) * 180) / Math.PI;
    const wind = sampleWind(field, lng, lat);
    if (!wind || !(Math.hypot(wind[0], wind[1]) >= CALM)) continue;

    // Backwards from the seed, then forwards, so the curve runs with the wind.
    const back: [number, number][] = [];
    const ahead: [number, number][] = [];
    for (const [direction, into] of [[-1, back], [1, ahead]] as const) {
      let at: [number, number] = [lng, lat];
      for (let step = 0; step < half; step++) {
        const next = advance(field, at[0], at[1], direction * stepSeconds);
        if (!next) break;
        into.push(next);
        at = next;
      }
    }
    if (back.length + ahead.length + 1 < 3) continue;
    lngs.length = lats.length = 0;
    for (let i = back.length - 1; i >= 0; i--) { lngs.push(back[i][0]); lats.push(back[i][1]); }
    lngs.push(lng); lats.push(lat);
    for (const p of ahead) { lngs.push(p[0]); lats.push(p[1]); }

    // How far along each vertex is, by distance, so a comet crosses a long curve and a short one in the same time.
    const m = lngs.length;
    const start = n;
    let length = 0;
    for (let i = 0; i < m; i++) {
      const x = mercatorX(lngs[i]);
      const y = mercatorY(lats[i]);
      if (i) length += Math.hypot(x - out[(n - 1) * WIND_STRIDE], y - out[(n - 1) * WIND_STRIDE + 1]);
      const o = n * WIND_STRIDE;
      out[o] = x;
      out[o + 1] = y;
      out[o + 2] = length; // made a fraction below
      n++;
    }
    const phase = ((seed + 1) * GOLDEN_RATIO) % 1;
    for (let i = start; i < n; i++) {
      const o = i * WIND_STRIDE;
      out[o + 2] = length > 0 ? out[o + 2] / length : 0;
      out[o + 3] = phase;
    }
    paths++;
  }
  return { vertices: out, count: n, paths };
}

/** Where the camera is above the ground, in metres: MapLibre's camera sits 1.5 screen-heights from the centre. */
export function cameraHeight(zoom: number, lat: number, screenHeight: number): number {
  return 1.5 * screenHeight * 78271.517 * Math.cos((Math.max(-85, Math.min(85, lat)) * Math.PI) / 180) / 2 ** zoom;
}

const FADE_LOW_METERS = 15000;
const FADE_HIGH_METERS = 60000;

/** Streamlines fade out as the camera comes down to the ground: gone under 15 km, full from 60 km. */
export function windHeightFade(height: number): number {
  return Math.max(0, Math.min(1, (height - FADE_LOW_METERS) / (FADE_HIGH_METERS - FADE_LOW_METERS)));
}
