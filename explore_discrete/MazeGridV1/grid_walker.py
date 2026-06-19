"""
Discrete grid four-room maze explorer (lightweight version).

State : a cell (i,j) on an n x n grid (i=col/x, j=row/y, origin bottom-left).
Action: one of 8 unit directions (N,S,E,W,NE,NW,SE,SW) -> a neighbor cell.
Counter: per-cell visit count. Start=1, all else 0. Wall cells are frozen at 0
         (never incremented), so they are never selectable as c*.
Storage: a DiGraph (one node per cell). First arrival at a cell freezes its
         canonical replay path (parent + incoming action); later arrivals add
         edges for bookkeeping only and DO NOT change the replay path.
Collision: a move is rejected iff the destination cell is an obstacle. The agent
         does not know walls in advance (they read as count 0), so it may pick a
         wall-ward action; that move fails, position is unchanged, the wall count
         is NOT touched, a bump is recorded, and the current walk breaks.

Selection: c* = a uniformly random cell among TREE cells (reached cells) with the
           least visit count. Reach it by reset-and-replay (no teleport), then
           take up to j steps. Action rule: never step back to the parent cell;
           prefer an unvisited neighbor (count 0); if none, the least-visited
           neighbor (random tie-break).
"""

import os
import sys
import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_maze import build_grid

ACTIONS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0),
           "NE": (1, 1), "NW": (-1, 1), "SE": (1, -1), "SW": (-1, -1)}


class GridWalker:
    def __init__(self, n=21, start=(2, 2), goal=(18, 18),
                 k_seed=30, j_roll=15, n_iters=6000, seed=0):
        self.n = n
        self.obstacles, self.mid, self.lo, self.hi = build_grid(n)
        self.start, self.goal = start, goal
        self.k, self.j, self.n_iters = k_seed, j_roll, n_iters
        self.rng = np.random.default_rng(seed)

        self.G = nx.DiGraph()
        self.G.add_node(start)
        self.canonical = {start: (None, None)}   # cell -> (parent, action)
        self.count = {start: 1}                   # per-cell visit count
        self.bumps = {}                           # wall cell -> bump count

        self.moves = self.replay_steps = self.resets = self.bumps_total = 0
        self.reached = False

    def cnt(self, c):
        return self.count.get(c, 0)

    def free(self, c):
        return 0 <= c[0] < self.n and 0 <= c[1] < self.n and c not in self.obstacles

    # --- action choice: avoid parent; prefer unvisited; else least-visited ---
    def select_action(self, cell, parent):
        i, j = cell
        cands = []
        for name, (di, dj) in ACTIONS.items():
            nb = (i + di, j + dj)
            if not (0 <= nb[0] < self.n and 0 <= nb[1] < self.n):
                continue
            if nb == parent:
                continue                          # hard-exclude the parent cell
            cands.append((name, nb))
        unvisited = [(nm, nb) for nm, nb in cands if self.cnt(nb) == 0]
        pool = unvisited if unvisited else None
        if pool is None:
            m = min(self.cnt(nb) for _, nb in cands)
            pool = [(nm, nb) for nm, nb in cands if self.cnt(nb) == m]
        return pool[self.rng.integers(len(pool))]

    # --- a walk of up to `steps` moves from `cell` (came from `parent`) ---
    def walk(self, cell, parent, steps):
        cur, prev = cell, parent
        for _ in range(steps):
            name, nb = self.select_action(cur, prev)
            self.moves += 1
            if nb in self.obstacles:              # collision: no move, no count
                self.bumps[nb] = self.bumps.get(nb, 0) + 1
                self.bumps_total += 1
                break
            if nb not in self.G:                  # first arrival -> freeze path
                self.canonical[nb] = (cur, name)
            self.G.add_edge(cur, nb, action=name)  # record edge (bookkeeping)
            self.count[nb] = self.cnt(nb) + 1
            prev, cur = cur, nb
            if cur == self.goal:
                self.reached = True
                return cur
        return cur

    # --- canonical replay path (frozen at first arrival) ---
    def replay_path(self, c):
        path = [c]
        while self.canonical[path[-1]][0] is not None:
            path.append(self.canonical[path[-1]][0])
        path.reverse()
        return path

    def reset_and_replay(self, c):
        self.resets += 1
        self.replay_steps += len(self.replay_path(c)) - 1

    def run(self):
        # phase 1: seed walk from the start
        cur = self.walk(self.start, None, self.k)
        if self.reached:
            return
        # phase 2: least-visited selection + reset-replay + rollout
        for _ in range(self.n_iters):
            tree_cells = list(self.G.nodes())
            m = min(self.count[c] for c in tree_cells)
            least = [c for c in tree_cells if self.count[c] == m]
            cstar = least[self.rng.integers(len(least))]
            if cstar != cur:
                self.reset_and_replay(cstar)
                cur = cstar
            self.count[cstar] += 1                # selecting c* = visiting it
            cur = self.walk(cstar, self.canonical[cstar][0], self.j)
            if self.reached:
                return


# --------------------------------------------------------------------------- #
def render(w, fname):
    n = w.n
    img = np.ones((n, n, 3))                       # white = free unvisited
    counts = np.array([[w.cnt((i, j)) for i in range(n)] for j in range(n)])
    cmax = max(1, counts.max())
    cmap = plt.cm.magma
    for j in range(n):
        for i in range(n):
            if (i, j) in w.obstacles:
                img[j, i] = (0.42, 0.42, 0.42)     # wall
            elif w.cnt((i, j)) > 0:
                img[j, i] = cmap(0.15 + 0.85 * w.cnt((i, j)) / cmax)[:3]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, origin="lower", interpolation="nearest")
    # tile borders for every cell (no index labels)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#cfcfcf", lw=0.5)
    ax.tick_params(which="both", length=0)
    ax.set_xticks([]); ax.set_yticks([])

    # canonical tree edges (light)
    for c, (p, _) in w.canonical.items():
        if p is not None:
            ax.plot([p[0], c[0]], [p[1], c[1]], color="black", lw=0.5, zorder=2)
    # solution path (canonical, goal -> start)
    if w.reached:
        path = np.array(w.replay_path(w.goal))
        ax.plot(path[:, 0], path[:, 1], color="#d62828", lw=2.2, zorder=3)
    ax.scatter(*w.start, c="#2a9d8f", s=110, zorder=4)
    ax.scatter(*w.goal, c="#e9c46a", s=220, marker="*", edgecolor="k", zorder=4)
    ax.set_title("discrete grid explorer — tiles shaded by visit count\n"
                 "(gray=wall, white=free unvisited, red=solution path)", fontsize=11)
    fig.tight_layout(); fig.savefig(fname, dpi=130)


if __name__ == "__main__":
    w = GridWalker(seed=None)
    w.run()

    free_cells = w.n * w.n - len(w.obstacles)
    reached_cells = len(w.G)
    in_deg = dict(w.G.in_degree())
    multi = sum(1 for c, d in in_deg.items() if d > 1)
    print("--- discrete grid explorer (lightweight) ---")
    print(f"reached goal       : {w.reached}")
    print(f"cells reached      : {reached_cells}/{free_cells}  "
          f"({reached_cells/free_cells*100:.1f}% coverage)")
    print(f"graph              : {w.G.number_of_nodes()} nodes, "
          f"{w.G.number_of_edges()} edges  "
          f"(extra edges over a tree: {w.G.number_of_edges()-(reached_cells-1)})")
    print(f"multi-parent cells : {multi}  (in-degree > 1 -> the DAG part)")
    print(f"moves attempted    : {w.moves}")
    print(f"replay steps       : {w.replay_steps}  (over {w.resets} resets)")
    print(f"wall bumps         : {w.bumps_total} total, "
          f"{len(w.bumps)} distinct wall cells hit")
    out_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(out_dir, f"grid_walker_{ts}.png")
    render(w, out_path)
    print(f"saved {out_path}")
