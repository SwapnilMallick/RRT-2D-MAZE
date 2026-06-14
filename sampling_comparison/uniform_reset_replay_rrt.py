"""
Uniform reset-and-replay RRT (baseline).

Defines the reset-and-replay RRT planner and runs it with UNIFORM sampling over
a multi-seed batch, printing summary statistics. The planner also supports an
optional soft room prior (room_weights); with room_weights=None it is the plain
uniform baseline run here. The soft-prior variant reuses this planner via
run_experiment (see soft_prior_reset_replay_rrt.py).

Task: start in the top-left room, goal in the top-right room.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "go_explore_rrt_networkx"))
from goexplore_maze import FourRoomMaze  # type: ignore[import]

# ----- experiment configuration -------------------------------------------- #
MAZE = FourRoomMaze(0.02)
START = np.array([0.10, 0.90])     # top-left room
GOAL = np.array([0.90, 0.90])      # top-right room
EPS = 0.03                         # step size
GOAL_BIAS = 0.10                   # prob of sampling the goal directly
GOAL_RADIUS = 0.03
BETA = 0.8                         # prior/uniform mix (only used if room_weights given)
MAX_ITER = 60000
SEEDS = range(40)


def room_box(r):                   # r: 0=BL 1=BR 2=TL 3=TR
    x0 = 0.0 if r % 2 == 0 else 0.5
    y0 = 0.0 if r < 2 else 0.5
    return x0, x0 + 0.5, y0, y0 + 0.5


def room_of(p):
    return (0 if p[0] < 0.5 else 1) + 2 * (0 if p[1] < 0.5 else 1)


class ResetReplayRRT:
    """Reset-and-replay RRT. room_weights=None => uniform sampling."""

    def __init__(self, maze, start, goal, room_weights=None, beta=BETA, eps=EPS,
                 goal_bias=GOAL_BIAS, goal_radius=GOAL_RADIUS, max_iter=MAX_ITER,
                 seed=0):
        self.maze = maze
        self.start, self.goal = np.asarray(start, float), np.asarray(goal, float)
        self.room_weights, self.beta, self.eps = room_weights, beta, eps
        self.goal_bias, self.goal_radius, self.max_iter = goal_bias, goal_radius, max_iter
        self.rng = np.random.default_rng(seed)
        self.pos = [self.start.copy()]
        self.parent = [-1]
        self.action = [None]
        self.env_steps = self.replay_steps = self.resets = 0

    # --- sampling (uniform, or soft room prior mixed with uniform) ---
    def _prior_sample(self):
        w = np.array([self.room_weights[r] for r in range(4)], float)
        w /= w.sum()
        r = self.rng.choice(4, p=w)
        x0, x1, y0, y1 = room_box(r)
        return np.array([self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)])

    def sample(self):
        if self.rng.random() < self.goal_bias:
            return self.goal.copy()
        if self.room_weights is not None and self.rng.random() < self.beta:
            return self._prior_sample()
        return np.array([self.rng.uniform(0, 1), self.rng.uniform(0, 1)])

    # --- reset-and-replay RRT machinery ---
    def nearest(self, q):
        d2 = [np.sum((p - q) ** 2) for p in self.pos]
        return int(np.argmin(d2))

    def steer(self, q_from, q_to):
        d = q_to - q_from
        n = np.linalg.norm(d)
        u = d / n if n > 1e-12 else np.array([1.0, 0.0])
        return u * self.eps

    def replay_to(self, i):
        self.resets += 1
        d = 0
        while self.parent[i] != -1:
            d += 1
            i = self.parent[i]
        self.env_steps += d
        self.replay_steps += d

    def plan(self):
        current = 0
        for _ in range(self.max_iter):
            q_s = self.sample()
            near = self.nearest(q_s)
            if near != current:
                self.replay_to(near)
                current = near
            a = self.steer(self.pos[near], q_s)
            q_new = self.pos[near] + a
            self.env_steps += 1
            if self.maze.segment_free(self.pos[near], q_new):
                self.pos.append(q_new)
                self.parent.append(near)
                self.action.append(a)
                current = len(self.pos) - 1
                if np.linalg.norm(q_new - self.goal) <= self.goal_radius:
                    return self._result(True)
            else:
                current = near
        return self._result(False)

    def _result(self, success):
        return dict(success=success, nodes=len(self.pos),
                    replay=self.replay_steps, resets=self.resets,
                    bottom=float(np.mean([room_of(p) < 2 for p in self.pos])))


def run_experiment(room_weights, tag, seeds=SEEDS, goal_bias=GOAL_BIAS):
    """Run the planner over many seeds and print summary statistics."""
    results = []
    for s in seeds:
        r = ResetReplayRRT(MAZE, START, GOAL, room_weights=room_weights,
                           goal_bias=goal_bias, seed=s).plan()
        results.append(r)
    nodes = np.array([r["nodes"] for r in results])
    replay = np.array([r["replay"] for r in results])
    bottom = np.array([r["bottom"] for r in results])
    succ = np.mean([r["success"] for r in results])
    print(f"=== {tag} ===")
    print(f"  seeds            : {len(results)}   goal_bias {goal_bias}")
    print(f"  success          : {succ*100:.0f}%")
    print(f"  nodes to solve   : mean {nodes.mean():6.1f}   median {np.median(nodes):.0f}")
    print(f"  replay steps     : mean {replay.mean():7.0f}   median {np.median(replay):.0f}")
    print(f"  nodes in bottom  : {bottom.mean()*100:4.1f}%  (off-path exploration)")
    return results


if __name__ == "__main__":
    run_experiment(room_weights=None, tag="uniform reset-replay RRT")
