"""
Continuous bin-based Go-Explore (anti-clutter via 16 bins).

State/action: continuous. Each of the 4 rooms is split into 4 geometric quadrant
bins -> 16 bins total. Two-level selection avoids piling up in one region:
  1. pick the NON-EMPTY bin with the minimum population (# nodes in it; random
     tie-break),
  2. within it pick the node with the least visitation count (random tie-break)
     = v*,
then explore from v* in Go-Explore fashion (reset-and-replay to v*, then a fixed
j-step diverse-action rollout). Bin population is the spatial saturation signal;
node visitation breaks intra-bin ties toward less-used nodes.

Deterministic point-mass dynamics f(q,a)=q+a, exact reset-and-replay. Collisions
via the maze's exact Liang-Barsky test. Reuses FourRoomMaze + step.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from goexplore_maze import FourRoomMaze, step
import datetime

class Node:
    __slots__ = ("id", "pos", "parent", "action", "bin", "visits")
    def __init__(self, nid, pos, parent, action, b):
        self.id, self.pos, self.parent, self.action = nid, pos, parent, action
        self.bin, self.visits = b, 1


class BinGoExplore:
    def __init__(self, maze, start, goal, bins_per_room_axis=3, eps=0.03,
                 k_seed=20, j_roll=12, m_iters=20000, goal_radius=0.03,
                 d_theta_deg=35, max_tries=16, node_cap=4000, seed=0):
        self.maze = maze
        self.start, self.goal = np.asarray(start, float), np.asarray(goal, float)
        self.eps, self.k, self.j = eps, k_seed, j_roll
        self.m, self.goal_radius = m_iters, goal_radius
        self.d_theta = np.deg2rad(d_theta_deg)
        self.max_tries, self.node_cap = max_tries, node_cap
        self.rng = np.random.default_rng(seed)

        # bins: bins_per_room_axis per room per axis -> the maze (2 rooms/axis)
        # is an nb x nb grid, nb = 2 * bins_per_room_axis. Default 3 -> 6x6 = 36.
        self.nb = 2 * bins_per_room_axis
        self.w = 1.0 / self.nb
        self.bin_boxes = {}
        for bi in range(self.nb):
            for bj in range(self.nb):
                b = bj * self.nb + bi
                self.bin_boxes[b] = (bi*self.w, (bi+1)*self.w, bj*self.w, (bj+1)*self.w)
        self.bin_pop = {b: 0 for b in self.bin_boxes}        # population per bin
        self.bin_nodes = {b: [] for b in self.bin_boxes}     # node ids per bin

        b0 = self.bin_of(self.start)
        self.nodes = [Node(0, self.start.copy(), None, None, b0)]
        self.bin_pop[b0] = 1
        self.bin_nodes[b0].append(0)
        self.taken = {0: []}            # node id -> headings already taken

        self.env_steps = self.replay_steps = self.ext_attempts = self.resets = 0
        self.reached = None

    def bin_of(self, p):
        bi = min(int(p[0] / self.w), self.nb - 1)
        bj = min(int(p[1] / self.w), self.nb - 1)
        return bj * self.nb + bi

    # ----- hierarchical selection: min-population bin, then least-visited node -
    def select(self):
        nonempty = [(b, self.bin_pop[b]) for b in self.bin_boxes if self.bin_pop[b] > 0]
        pmin = min(p for _, p in nonempty)
        cand_bins = [b for b, p in nonempty if p == pmin]
        b = cand_bins[self.rng.integers(len(cand_bins))]
        ids = self.bin_nodes[b]
        vmin = min(self.nodes[i].visits for i in ids)
        cand = [i for i in ids if self.nodes[i].visits == vmin]
        return self.nodes[cand[self.rng.integers(len(cand))]]

    def _diverse_action(self, node_id):
        taken = self.taken.get(node_id, [])
        theta = self.rng.uniform(0, 2*np.pi)
        for _ in range(self.max_tries):
            if all(abs((theta - t + np.pi) % (2*np.pi) - np.pi) >= self.d_theta
                   for t in taken):
                break
            theta = self.rng.uniform(0, 2*np.pi)
        return theta, self.eps * np.array([np.cos(theta), np.sin(theta)])

    def _add(self, pos, parent, action):
        b = self.bin_of(pos)
        n = Node(len(self.nodes), pos, parent, action, b)
        self.nodes.append(n)
        self.bin_pop[b] += 1
        self.bin_nodes[b].append(n.id)
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

    def run(self):
        # phase 1: seed walk from the start
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
        # phase 2: bin-based selection + reset-replay + diverse rollout
        for _ in range(self.m):
            v = self.select()
            v.visits += 1                       # selecting = visiting the node
            if not np.allclose(v.pos, cur.pos):
                self._replay_to(v)
                cur = v
            if self._rollout(v, v.pos.copy()):
                return
            cur = self.nodes[-1]
            if len(self.nodes) >= self.node_cap:
                return


def coverage(ex, G=50):
    occ = np.zeros((G, G), bool)
    for n in ex.nodes:
        occ[min(int(n.pos[1]*G), G-1), min(int(n.pos[0]*G), G-1)] = True
    return occ.mean()


if __name__ == "__main__":
    maze = FourRoomMaze(0.02)
    ex = BinGoExplore(maze, start=(0.10, 0.90), goal=(0.90, 0.90), seed=None)
    ex.run()

    pops = np.array([ex.bin_pop[b] for b in ex.bin_boxes])
    print("--- continuous bin-based Go-Explore ---")
    print(f"bin grid          : {ex.nb}x{ex.nb} = {ex.nb**2} bins "
          f"({ex.nb//2} per room per axis, {(ex.nb//2)**2} per room)")
    print(f"reached goal      : {ex.reached is not None}")
    print(f"tree nodes        : {len(ex.nodes)}")
    print(f"env steps         : {ex.env_steps}  (replay {ex.replay_steps}, "
          f"resets {ex.resets})")
    print(f"spatial coverage  : {coverage(ex)*100:.1f}% of 50x50 cells")
    print(f"bin populations   : min {pops.min()}, max {pops.max()}, "
          f"mean {pops.mean():.1f}, std {pops.std():.1f}  ({ex.nb**2} bins)")
    print(f"empty bins        : {int((pops == 0).sum())}/{ex.nb**2}")

    # figure: tree colored by bin + bin grid + per-bin population
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax in (a1, a2):
        for xmin, ymin, xmax, ymax in maze.walls:
            ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                   facecolor="#555", zorder=3))
        for t in range(1, ex.nb):
            x = t * ex.w
            ax.axvline(x, color="#9bbcdc", lw=0.6, ls="--", zorder=2)
            ax.axhline(x, color="#9bbcdc", lw=0.6, ls="--", zorder=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    for n in ex.nodes:
        if n.parent is not None:
            a1.plot([n.parent.pos[0], n.pos[0]], [n.parent.pos[1], n.pos[1]],
                    color="#8fb3e0", lw=0.4, zorder=1)
    P = np.array([n.pos for n in ex.nodes])
    a1.scatter(P[:, 0], P[:, 1], s=3, c="#36506e", zorder=4)
    a1.scatter(*ex.start, c="#2a9d8f", s=80, zorder=5)
    a1.scatter(*ex.goal, c="#e9c46a", s=160, marker="*", edgecolor="k", zorder=5)
    if ex.reached is not None:
        path = [ex.reached]
        while path[-1].parent is not None: path.append(path[-1].parent)
        pp = np.array([p.pos for p in path])
        a1.plot(pp[:, 0], pp[:, 1], color="#d62828", lw=2, zorder=6)
    a1.set_title(f"tree + {ex.nb**2} bins ({len(ex.nodes)} nodes)")

    pmax = max(1, pops.max())
    for b, (x0, x1, y0, y1) in ex.bin_boxes.items():
        v = ex.bin_pop[b]
        a2.add_patch(Rectangle((x0, y0), ex.w, ex.w,
                               facecolor=plt.cm.magma(0.15 + 0.85*v/pmax),
                               alpha=0.9, zorder=0))
        a2.text((x0+x1)/2, (y0+y1)/2, str(v), ha="center", va="center",
                fontsize=7, color="white", zorder=4)
    a2.set_title("per-bin population (anti-clutter signal)")
    images_dir = Path(__file__).parent / "images"
    images_dir.mkdir(exist_ok=True)
    out_path = images_dir / f"bin_goexplore_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    fig.tight_layout(); fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")
