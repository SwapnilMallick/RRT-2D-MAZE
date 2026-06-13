"""
Exploration variant #1 — score a node purely by its visit count.

Selection rule:  v* = argmin over nodes of score(node), with score(node) = visit
count (random tie-break). The chosen node's counter is incremented. Everything
else is the reset-and-replay machinery from before:
  - SEED: a chain of up to K random actions from the root.
  - Each iteration: pick v* by score, reset to root, replay root->v*, then roll
    out up to J diverse random actions (each heading kept angularly distinct
    from headings already taken at that node), collision-checked, break on wall.

`score()` is the ONLY decision-specific piece. Later variants (cell counts,
distance-to-doorway, ...) override just this method; the rest is untouched.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "go_explore_rrt_networkx"))
from goexplore_maze import FourRoomMaze, step


class Node:
    __slots__ = ("id", "pos", "parent", "action", "counter")
    def __init__(self, nid, pos, parent, action, counter=1):
        self.id, self.pos, self.parent = nid, pos, parent
        self.action, self.counter = action, counter


class CountExplorer:
    def __init__(self, maze, start, goal, eps=0.10, k_seed=15, j_rollout=6,
                 m_iters=4000, goal_radius=0.04, d_theta_deg=35, max_tries=16,
                 node_cap=3000, seed=None):
        self.maze = maze
        self.start = np.asarray(start, float)
        self.goal = np.asarray(goal, float)
        self.eps, self.k, self.j = eps, k_seed, j_rollout
        self.m, self.goal_radius = m_iters, goal_radius
        self.d_theta = np.deg2rad(d_theta_deg)
        self.max_tries, self.node_cap = max_tries, node_cap
        self.rng = np.random.default_rng(seed)

        self.nodes = [Node(0, self.start.copy(), None, None, counter=1)]
        self.taken = {0: []}          # node id -> list of action headings taken
        self.env_steps = self.replay_steps = self.ext_attempts = self.resets = 0
        self.reached = None
        self.selections = []          # ids chosen as v*, for diagnostics

    # ----- THE scoring rule (override this in later variants) -------------- #
    def score(self, node):
        return node.counter           # lower = less visited = preferred

    # ----- selection ------------------------------------------------------- #
    def select(self):
        s = [self.score(n) for n in self.nodes]
        m = min(s)
        cand = [n for n, sc in zip(self.nodes, s) if sc == m]
        v = cand[self.rng.integers(len(cand))]
        v.counter += 1                # selecting == visiting
        self.selections.append(v.id)
        return v

    # ----- helpers --------------------------------------------------------- #
    def _diverse_action(self, node_id):
        taken = self.taken.get(node_id, [])
        theta = self.rng.uniform(0, 2 * np.pi)
        for _ in range(self.max_tries):
            if all(abs((theta - t + np.pi) % (2*np.pi) - np.pi) >= self.d_theta
                   for t in taken):
                break
            theta = self.rng.uniform(0, 2 * np.pi)
        return theta, self.eps * np.array([np.cos(theta), np.sin(theta)])

    def _add(self, pos, parent, action):
        n = Node(len(self.nodes), pos, parent, action, counter=1)
        self.nodes.append(n)
        self.taken[n.id] = []
        return n

    def _pred_walk(self, node):
        chain = []
        while node.parent is not None:
            chain.append(node); node = node.parent
        chain.reverse()
        return chain

    def _replay_to(self, node):
        self.resets += 1
        pos = self.nodes[0].pos.copy()
        for nd in self._pred_walk(node):
            pos = step(pos, nd.action)
            self.env_steps += 1; self.replay_steps += 1
        assert np.allclose(pos, node.pos, atol=1e-9)
        return pos

    def _at_goal(self, pos):
        return np.linalg.norm(pos - self.goal) <= self.goal_radius

    def _rollout(self, v_star, pos):
        cur = v_star
        for _ in range(self.j):
            theta, a = self._diverse_action(cur.id)
            new_pos = step(pos, a)
            self.env_steps += 1; self.ext_attempts += 1
            if self.maze.segment_free(pos, new_pos):
                child = self._add(new_pos, cur, a)
                self.taken[cur.id].append(theta)
                cur, pos = child, new_pos
                if self._at_goal(pos):
                    self.reached = child
                    return True
            else:
                break
        return False

    # ----- driver ---------------------------------------------------------- #
    def run(self):
        # phase 1: seed chain from root
        cur, pos = self.nodes[0], self.start.copy()
        for _ in range(self.k):
            theta, a = self._diverse_action(cur.id)
            new_pos = step(pos, a)
            self.env_steps += 1; self.ext_attempts += 1
            if self.maze.segment_free(pos, new_pos):
                child = self._add(new_pos, cur, a)
                self.taken[cur.id].append(theta)
                cur, pos = child, new_pos
                if self._at_goal(pos):
                    self.reached = child; return
            else:
                break
        # phase 2: count-scored exploration
        for it in range(self.m):
            self.iters = it + 1
            v = self.select()
            self._replay_to(v)
            if self._rollout(v, v.pos.copy()):
                return
            if len(self.nodes) >= self.node_cap:
                return


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    maze = FourRoomMaze(0.03)
    ex = CountExplorer(maze, start=(0.10, 0.90), goal=(0.90, 0.90))
    ex.run()

    counts = np.array([n.counter for n in ex.nodes])
    G = 50
    occ = np.zeros((G, G), bool)
    for n in ex.nodes:
        occ[min(int(n.pos[1]*G), G-1), min(int(n.pos[0]*G), G-1)] = True

    print("--- variant #1: score = visit count ---")
    print(f"reached goal      : {ex.reached is not None}")
    print(f"iterations        : {getattr(ex, 'iters', 0)}")
    print(f"tree nodes        : {len(ex.nodes)}")
    print(f"env steps         : {ex.env_steps}  (replay {ex.replay_steps}, "
          f"attempts {ex.ext_attempts}, resets {ex.resets})")
    print(f"counter           : min {counts.min()}, max {counts.max()}, "
          f"mean {counts.mean():.2f}")
    print(f"min-score bucket  : {int((counts == counts.min()).sum())}/{len(ex.nodes)} "
          f"nodes share the minimum score (selection ~ uniform over these)")
    print(f"spatial coverage  : {occ.mean()*100:.1f}% of {G}x{G} cells")

    # plot: tree + coverage
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))
    for ax in (a1, a2):
        for xmin, ymin, xmax, ymax in maze.walls:
            ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                   facecolor="#555", zorder=3))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for n in ex.nodes:
        if n.parent is not None:
            a1.plot([n.parent.pos[0], n.pos[0]], [n.parent.pos[1], n.pos[1]],
                    color="#8fb3e0", lw=0.4, zorder=1)
    P = np.array([n.pos for n in ex.nodes])
    a1.scatter(P[:, 0], P[:, 1], s=3, c="#36506e", zorder=2)
    a1.scatter(*ex.start, c="#2a9d8f", s=80, zorder=5, label="start")
    a1.scatter(*ex.goal, c="#e9c46a", s=170, marker="*", edgecolor="k",
               zorder=5, label="goal")
    a1.legend(loc="upper left"); a1.set_title(f"tree ({len(ex.nodes)} nodes)")
    H, *_ = np.histogram2d(P[:, 0], P[:, 1], bins=40, range=[[0, 1], [0, 1]])
    a2.imshow(np.log1p(H.T), origin="lower", extent=[0, 1, 0, 1],
              cmap="magma", aspect="equal", zorder=0)
    a2.set_title("node density (log) — coverage")
    out_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "explore_count.png")
    fig.tight_layout(); fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
