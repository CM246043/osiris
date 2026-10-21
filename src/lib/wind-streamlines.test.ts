import { describe, expect, it } from 'vitest';
import type { WindGrid } from './wind-layer';
import { bakeWindStreamlines, cameraHeight, windHeightFade, WIND_STRIDE } from './wind-streamlines';

/** A one-degree grid with the same wind everywhere, rows north to south. */
function uniform(u: number, v: number): WindGrid {
  const ni = 360;
  const nj = 181;
  return { ni, nj, la1: 90, lo1: 0, dx: 1, dy: 1, u: new Float32Array(ni * nj).fill(u), v: new Float32Array(ni * nj).fill(v) };
}

/** The curves, as lists of [x, y, along, phase]. */
function curves(baked: ReturnType<typeof bakeWindStreamlines>): number[][][] {
  const out: number[][][] = [];
  for (let i = 0; i < baked.count; i++) {
    const o = i * WIND_STRIDE;
    const vertex = Array.from(baked.vertices.subarray(o, o + WIND_STRIDE));
    if (i === 0 || vertex[2] < out.at(-1)!.at(-1)![2]) out.push([]);
    out.at(-1)!.push(vertex);
  }
  return out;
}

describe('bakeWindStreamlines', () => {
  it('draws no curves where the air is still', () => {
    const baked = bakeWindStreamlines(uniform(0, 0), { count: 200 });
    expect(baked.paths).toBe(0);
    expect(baked.count).toBe(0);
  });

  it('makes a curve per seed in moving air, each running from 0 to 1 along its length', () => {
    const baked = bakeWindStreamlines(uniform(5, 0), { count: 300 });
    expect(baked.paths).toBeGreaterThan(200);
    const all = curves(baked);
    expect(all).toHaveLength(baked.paths);
    for (const curve of all) {
      expect(curve.length).toBeGreaterThanOrEqual(3);
      expect(curve[0][2]).toBe(0);
      expect(curve.at(-1)![2]).toBeCloseTo(1, 5);
      // One phase per curve, and `along` only ever grows.
      expect(new Set(curve.map(v => v[3])).size).toBe(1);
      for (let i = 1; i < curve.length; i++) expect(curve[i][2]).toBeGreaterThan(curve[i - 1][2]);
    }
  });

  it('runs with the wind: an eastward wind moves a curve east', () => {
    const baked = bakeWindStreamlines(uniform(8, 0), { count: 100 });
    for (const curve of curves(baked)) {
      // Mercator x grows eastward, except where a curve is cut at the seam, which never wraps.
      for (let i = 1; i < curve.length; i++) expect(curve[i][0]).toBeGreaterThan(curve[i - 1][0]);
    }
  });

  it('runs north in a northerly wind, which is a smaller mercator y', () => {
    const baked = bakeWindStreamlines(uniform(0, 8), { count: 100 });
    for (const curve of curves(baked)) {
      for (let i = 1; i < curve.length; i++) expect(curve[i][1]).toBeLessThan(curve[i - 1][1]);
    }
  });

  it('never wraps a curve across the seam or to a pole', () => {
    const baked = bakeWindStreamlines(uniform(30, 10), { count: 400 });
    for (const curve of curves(baked)) {
      for (let i = 1; i < curve.length; i++) expect(Math.abs(curve[i][0] - curve[i - 1][0])).toBeLessThan(0.5);
      for (const v of curve) {
        expect(v[1]).toBeGreaterThan(0);
        expect(v[1]).toBeLessThan(1);
      }
    }
  });

  it('is the same every time', () => {
    const a = bakeWindStreamlines(uniform(5, 2), { count: 50 });
    const b = bakeWindStreamlines(uniform(5, 2), { count: 50 });
    expect(Array.from(a.vertices.subarray(0, a.count * WIND_STRIDE))).toEqual(Array.from(b.vertices.subarray(0, b.count * WIND_STRIDE)));
  });
});

describe('windHeightFade', () => {
  it('is gone near the ground and full from sixty kilometres', () => {
    expect(windHeightFade(5_000)).toBe(0);
    expect(windHeightFade(15_000)).toBe(0);
    expect(windHeightFade(37_500)).toBeCloseTo(0.5, 5);
    expect(windHeightFade(60_000)).toBe(1);
    expect(windHeightFade(20_000_000)).toBe(1);
  });
});

describe('cameraHeight', () => {
  it('doubles with each zoom out, and shrinks away from the equator', () => {
    expect(cameraHeight(4, 0, 800) / cameraHeight(5, 0, 800)).toBeCloseTo(2, 5);
    expect(cameraHeight(6, 60, 800)).toBeLessThan(cameraHeight(6, 0, 800));
  });
});
