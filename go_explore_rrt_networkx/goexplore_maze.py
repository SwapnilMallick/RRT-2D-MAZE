"""
Shared maze environment for the Go-Explore RRT family.

Exports:
    make_wall    — build a list of AABB rects for one wall segment with door gaps
    FourRoomMaze — continuous unit-square maze with 4 rooms and Liang-Barsky collision
    step         — trivial point-mass forward dynamics (pos + action)
"""

import numpy as np


def make_wall(axis, pos, span, thickness, doors):
    """Build a wall (list of axis-aligned rects) with doorway gaps.

    axis='v': vertical wall at x=pos, spanning y in `span`, gaps `doors` along y.
    axis='h': horizontal wall at y=pos, spanning x in `span`, gaps `doors` along x.
    Each rect is (xmin, ymin, xmax, ymax). doors is a list of (lo, hi) gaps.
    """
    t = thickness / 2.0
    lo, hi = span
    cuts = sorted(doors)
    rects, cursor = [], lo
    for a, b in cuts:
        if a > cursor:
            if axis == "v":
                rects.append((pos - t, cursor, pos + t, a))
            else:
                rects.append((cursor, pos - t, a, pos + t))
        cursor = max(cursor, b)
    if cursor < hi:
        if axis == "v":
            rects.append((pos - t, cursor, pos + t, hi))
        else:
            rects.append((cursor, pos - t, hi, pos + t))
    return rects


class FourRoomMaze:
    """Unit square split into 4 rooms by a cross of walls with 4 doorways,
    giving the classic cyclic four-room connectivity."""

    def __init__(self, wall_thickness=0.02):
        self.bounds = (0.0, 0.0, 1.0, 1.0)  # xmin, ymin, xmax, ymax
        self.walls = []
        # Vertical wall at x=0.5: doorways at 18-30% and 70-82% along y
        self.walls += make_wall("v", 0.5, (0.0, 1.0), wall_thickness,
                                doors=[(0.18, 0.30), (0.70, 0.82)])
        # Horizontal wall at y=0.5: doorways at 18-30% and 70-82% along x
        self.walls += make_wall("h", 0.5, (0.0, 1.0), wall_thickness,
                                doors=[(0.18, 0.30), (0.70, 0.82)])

    def in_bounds(self, p):
        x0, y0, x1, y1 = self.bounds
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def point_in_walls(self, p):
        for xmin, ymin, xmax, ymax in self.walls:
            if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax:
                return True
        return False

    def segment_free(self, p, q):
        """True if segment p->q hits no wall (exact, via Liang-Barsky clip)."""
        if not (self.in_bounds(p) and self.in_bounds(q)):
            return False
        for rect in self.walls:
            if self._segment_hits_aabb(p, q, rect):
                return False
        return True

    @staticmethod
    def _segment_hits_aabb(p, q, rect):
        """Liang-Barsky: does segment p->q intersect axis-aligned box rect?"""
        xmin, ymin, xmax, ymax = rect
        dx, dy = q[0] - p[0], q[1] - p[1]
        t0, t1 = 0.0, 1.0
        for pp, qq in ((-dx, p[0] - xmin), (dx, xmax - p[0]),
                       (-dy, p[1] - ymin), (dy, ymax - p[1])):
            if pp == 0:
                if qq < 0:           # parallel and outside this slab
                    return False
            else:
                r = qq / pp
                if pp < 0:
                    if r > t1:
                        return False
                    if r > t0:
                        t0 = r
                else:
                    if r < t0:
                        return False
                    if r < t1:
                        t1 = r
        return t0 <= t1

    def sample(self, goal, goal_bias, rng: np.random.Generator):
        if rng.random() < goal_bias:
            return np.asarray(goal, dtype=float)
        x0, y0, x1, y1 = self.bounds
        return np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])


def step(pos, action):
    """Forward dynamics of the point mass: state = position, action = displacement."""
    return pos + action
