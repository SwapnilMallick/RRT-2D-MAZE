"""
Shared maze primitives extracted from go_explore_inspired_rrt/goexplore_rrt.py.
Provides FourRoomMaze and step for use by bin_goexplore.py and bin_sweep.py.
"""

import numpy as np


def make_wall(axis, pos, span, thickness, doors):
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
    """Unit square split into 4 rooms by a cross of walls with 4 doorways."""

    def __init__(self, wall_thickness=0.02):
        self.bounds = (0.0, 0.0, 1.0, 1.0)
        self.walls = []
        self.walls += make_wall("v", 0.5, (0.0, 1.0), wall_thickness,
                                doors=[(0.18, 0.30), (0.70, 0.82)])
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
        if not (self.in_bounds(p) and self.in_bounds(q)):
            return False
        for rect in self.walls:
            if self._segment_hits_aabb(p, q, rect):
                return False
        return True

    @staticmethod
    def _segment_hits_aabb(p, q, rect):
        xmin, ymin, xmax, ymax = rect
        dx, dy = q[0] - p[0], q[1] - p[1]
        t0, t1 = 0.0, 1.0
        for pp, qq in ((-dx, p[0] - xmin), (dx, xmax - p[0]),
                       (-dy, p[1] - ymin), (dy, ymax - p[1])):
            if pp == 0:
                if qq < 0:
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

    def sample(self, goal, goal_bias, rng):
        if rng.random() < goal_bias:
            return np.asarray(goal, dtype=float)
        x0, y0, x1, y1 = self.bounds
        return np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
    
class FourRoomMazeV2:
    """Unit square split into 4 rooms by a cross of walls with 4 doorways,
    giving the classic cyclic four-room connectivity."""

    def __init__(self, wall_thickness=0.02):
        self.bounds = (0.0, 0.0, 1.0, 1.0)  # xmin, ymin, xmax, ymax
        self.walls = []
        # Vertical wall at x=0.5: doorways at 18-30% and 70-82% along y
        self.walls += make_wall("v", 0.5, (0.0, 1.0), wall_thickness,
                                doors=[(0.18, 0.30), (0.70, 0.82)])
        #new vertical walls at x=0.25 and x=0.75
        self.walls += make_wall("v", 0.25, (0.0, 1.0), wall_thickness,
                               doors=[(0.35, 0.65),])
        self.walls += make_wall("v", 0.75, (0.0, 1.0), wall_thickness,
                                doors=[(0.35, 0.65)])
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
    
class FourRoomMazeV3:
    """Unit square split into 4 rooms by a cross of walls with 4 doorways,
    giving the classic cyclic four-room connectivity."""

    def __init__(self, wall_thickness=0.02):
        self.bounds = (0.0, 0.0, 1.0, 1.0)  # xmin, ymin, xmax, ymax
        self.walls = []
        # Vertical wall at x=0.5: doorways at 18-30% and 70-82% along y
        self.walls += make_wall("v", 0.5, (0.0, 1.0), wall_thickness,
                                doors=[(0.18, 0.30), (0.70, 0.82)])
        #new vertical walls at x=0.25 and x=0.75
        self.walls += make_wall("v", 0.25, (0.0, 1.0), wall_thickness,
                                doors=[(0.0, 0.12), (0.37, 0.62), (0.87, 1.0)])
        self.walls += make_wall("v", 0.75, (0.0, 1.0), wall_thickness,
                                doors=[(0.0, 0.12), (0.37, 0.62), (0.87, 1.0)])
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

class FourRoomMazeV4:
    """Unit square split into 4 rooms by a cross of walls with 4 doorways,
    giving the classic cyclic four-room connectivity."""

    def __init__(self, wall_thickness=0.02):
        self.bounds = (0.0, 0.0, 1.0, 1.0)  # xmin, ymin, xmax, ymax
        self.walls = []
        # Vertical wall at x=0.5: doorways at 18-30% and 70-82% along y
        self.walls += make_wall("v", 0.5, (0.0, 1.0), wall_thickness,
                                doors=[(0.18, 0.30), (0.70, 0.82)])
        #new vertical walls at x=0.25 and x=0.75
        self.walls += make_wall("v", 0.25, (0.0, 1.0), 0.20,
                                doors=[(0.0, 0.12), (0.37, 0.62), (0.87, 1.0)])
        self.walls += make_wall("v", 0.75, (0.0, 1.0), 0.20,
                                doors=[(0.0, 0.12), (0.37, 0.62), (0.87, 1.0)])
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
    
class FourRoomMazeV5:
    """Unit square split into 4 rooms by a cross of walls with 4 doorways,
    giving the classic cyclic four-room connectivity."""

    def __init__(self, wall_thickness=0.02):
        self.bounds = (0.0, 0.0, 1.0, 1.0)  # xmin, ymin, xmax, ymax
        self.walls = []
        # Vertical wall at x=0.5: doorways at 18-30% and 70-82% along y
        # self.walls += make_wall("v", 0.5, (0.0, 1.0), wall_thickness,
        #                        doors=[(0.18, 0.30), (0.70, 0.82)])
        #new vertical walls at x=0.18 and x=0.82: doorways at 0-12% and 37-62% and 87-100% along y
        self.walls += make_wall("v", 0.18, (0.0, 1.0), wall_thickness,
                                doors=[(0.0, 0.12), (0.87, 1.0)])
        self.walls += make_wall("v", 0.82, (0.0, 1.0), wall_thickness,
                                doors=[(0.0, 0.12), (0.87, 1.0)])
        
        # Horizontal walls at x=0.35 and x=0.65: doorways at 0-37% and 62-100% along y
        self.walls += make_wall("h", 0.35, (0.0, 1.0), wall_thickness,
                                doors=[(0.0, 0.37), (0.63, 1.0)])
        self.walls += make_wall("h", 0.65, (0.0, 1.0), wall_thickness,
                                doors=[(0.0, 0.37), (0.63, 1.0)])
        
        # Vertical walls at x=0.37 and x=0.63: doorways at 0-33% and 66-100% along y
        self.walls += make_wall("v", 0.37, (0.0, 1.0), wall_thickness,
                                doors=[(0.0, 0.335), (0.45, 0.55), (0.665, 1.0)])
        self.walls += make_wall("v", 0.63, (0.0, 1.0), wall_thickness,
                                doors=[(0.0, 0.335), (0.45, 0.55), (0.665, 1.0)])

        # Horizontal walls at y=0.135 and y=0.855: doorways at 0-18%, 42-58%, and 82-100% along x
        self.walls += make_wall("h", 0.135, (0.0, 1.0), wall_thickness,
                                doors=[(0.0, 0.18), (0.42, 0.58), (0.82, 1.0)])
        self.walls += make_wall("h", 0.855, (0.0, 1.0), wall_thickness,
                                doors=[(0.0, 0.18), (0.42, 0.58), (0.82, 1.0)])
        # Horizontal wall at y=0.5: doorways at 18-30% and 70-82% along x
        # self.walls += make_wall("h", 0.5, (0.0, 1.0), wall_thickness,
        #                        doors=[(0.18, 0.30), (0.70, 0.82)])

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
    return pos + action
