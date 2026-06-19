"""
Run the same discrete walker (grid_walker.GridWalker) on the MODIFIED four-room
maze (4 cols x 2 rows, 8 rooms) from maze_grid_mod.build_grid_mod.

Only the grid + start/goal change; the algorithm (cell-count least-visited
selection, DiGraph storage with first-arrival-frozen replay, seed walk, rollout,
wall bumps) is inherited unchanged.
"""

import os
import sys
import datetime
import numpy as np
import networkx as nx

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)                                          # maze_grid_mod
sys.path.insert(0, os.path.join(_here, "..", "MazeGridV1"))       # grid_walker

from grid_walker import GridWalker, render
from maze_grid_8_room import build_grid_8_room


class GridWalkerMod(GridWalker):
    def __init__(self, n=29, start=(2, 26), goal=(26, 26),
                 k_seed=40, j_roll=20, n_iters=40000, seed=0):
        self.n = n
        self.obstacles, self.hrow, self.vcols = build_grid_8_room(n)
        self.start, self.goal = start, goal
        self.k, self.j, self.n_iters = k_seed, j_roll, n_iters
        self.rng = np.random.default_rng(seed)
        self.G = nx.DiGraph()
        self.G.add_node(start)
        self.canonical = {start: (None, None)}
        self.count = {start: 1}
        self.bumps = {}
        self.moves = self.replay_steps = self.resets = self.bumps_total = 0
        self.reached = False


if __name__ == "__main__":
    w = GridWalkerMod(seed=None)
    w.run()

    free_cells = w.n * w.n - len(w.obstacles)
    reached_cells = len(w.G)
    in_deg = dict(w.G.in_degree())
    multi = sum(1 for c, d in in_deg.items() if d > 1)
    print("--- discrete grid explorer on MODIFIED maze (8 rooms) ---")
    print(f"reached goal       : {w.reached}")
    print(f"cells reached      : {reached_cells}/{free_cells}  "
          f"({reached_cells/free_cells*100:.1f}% coverage)")
    print(f"graph              : {w.G.number_of_nodes()} nodes, "
          f"{w.G.number_of_edges()} edges  "
          f"(extra over a tree: {w.G.number_of_edges()-(reached_cells-1)})")
    print(f"multi-parent cells : {multi}  (in-degree > 1)")
    print(f"moves attempted    : {w.moves}")
    print(f"replay steps       : {w.replay_steps}  (over {w.resets} resets)")
    print(f"wall bumps         : {w.bumps_total} total, "
          f"{len(w.bumps)} distinct wall cells hit")
    out_dir = os.path.join(_here, "images")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(out_dir, f"grid_walker_mod_{ts}.png")
    render(w, out_path)
    print(f"saved {out_path}")
