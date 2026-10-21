import type { WindGrid } from './wind-layer';
import { bakeWindStreamlines } from './wind-streamlines';

/**
 * OSIRIS — the Wind layer's streamlines, made off the main thread. A full set
 * takes the best part of a tenth of a second, and while the time slider plays
 * they are remade several times a second; here that never holds up the map.
 */

type Request = { id: number; field: WindGrid; count: number };

// The DOM typings describe a window; this file runs in a worker.
const scope = self as unknown as {
  onmessage: ((event: MessageEvent<Request>) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
};

scope.onmessage = ({ data }) => {
  const { vertices, count, paths } = bakeWindStreamlines(data.field, { count: data.count });
  scope.postMessage({ id: data.id, vertices, count, paths }, [vertices.buffer]);
};
