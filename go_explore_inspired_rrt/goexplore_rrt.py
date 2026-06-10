"""
Voronoi-biased Go-Explore (phase 1) in a continuous four-room maze.

This is your algorithm: an RRT whose tree is identical to vanilla RRT, but the
agent is NOT allowed to teleport to an arbitrary node before extending. Instead,
the only primitives are:
    - reset to the root, and
    - roll a stored action sequence forward.
So to extend from the nearest node, the agent resets to root and *replays* the
stored root->node action sequence, then takes one extension step toward the
sample. We track the replayed steps explicitly, because that O(depth) cost per
iteration is the whole difference between this and teleport-RRT.

Node representation: each node stores its parent and the *incoming* edge action
(the displacement that produced it from its parent). Replaying the chain of
incoming actions from the root reconstructs the node's state exactly under
deterministic point-mass dynamics.

Dynamics here are a trivial point mass: state = position, action = displacement,
step(pos, a) = pos + a. That makes replay exact and bit-reproducible, which is
the regime where phase-1-only is complete (no robustification needed).

Usage
-----
    python goexplore_rrt.py [options]

Examples
--------
    python goexplore_rrt.py
    python goexplore_rrt.py --start "0.10,0.10" --goal "0.90,0.90"
    python goexplore_rrt.py --seed 42 --max_iter 12000
    python goexplore_rrt.py --noise_std 0.3 --viz animate --animate_every 10
    python goexplore_rrt.py --viz both --fps 10
"""

from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np


# --------------------------------------------------------------------------- #
# Environment: continuous four-room maze
# --------------------------------------------------------------------------- #
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
        # Vertical wall at x=0.5: doorway low (room1<->room2) and high (room3<->room4)
        self.walls += make_wall("v", 0.5, (0.0, 1.0), wall_thickness,
                                doors=[(0.18, 0.30), (0.70, 0.82)])
        # Horizontal wall at y=0.5: doorway left (room1<->room3) and right (room2<->room4)
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
        """True if the segment p->q hits no wall (exact, via Liang-Barsky clip)."""
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


# --------------------------------------------------------------------------- #
# Tree
# --------------------------------------------------------------------------- #
class Node:
    __slots__ = ("id", "pos", "parent", "action")

    def __init__(self, nid, pos, parent, action):
        self.id = nid
        self.pos = pos              # configuration (here: 2D position)
        self.parent = parent        # parent Node or None for root
        self.action = action        # incoming edge action (displacement); None at root


def step(pos, action):
    """Forward dynamics of the point mass."""
    return pos + action


# --------------------------------------------------------------------------- #
# Planner: Voronoi-biased Go-Explore (phase 1)
# --------------------------------------------------------------------------- #
class GoExploreRRT:
    def __init__(self, maze, start, goal, rng: np.random.Generator,
                 step_size=0.03, goal_bias=0.10,
                 goal_radius=0.03, noise_std=0.0, max_iter=8000, verify_replay=True):
        self.maze = maze
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.rng = rng
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_radius = goal_radius
        self.noise_std = noise_std        # >0 => "random action toward sample"
        self.max_iter = max_iter
        self.verify_replay = verify_replay

        self.nodes = [Node(0, self.start.copy(), None, None)]
        # instrumentation: this is what separates us from teleport-RRT
        self.env_steps = 0          # total forward dynamics calls (replay + extension)
        self.replay_steps = 0       # steps spent re-executing to reach a node
        self.extension_steps = 0    # steps that attempted to grow the tree
        self.resets = 0             # how many times we reset to root

        # Per-iteration traversal history for animation.
        # Each entry: {'n_nodes': int, 'traversal': ndarray(K,2), 'success': bool}
        # traversal[0] = root, traversal[-2] = near, traversal[-1] = new_pos
        self.history: list[dict] = []

    # --- tree helpers ---
    def nearest(self, point):
        # brute force O(N); a KD-tree would scale this, but the demo stays clear.
        d2 = [np.sum((n.pos - point) ** 2) for n in self.nodes]
        return self.nodes[int(np.argmin(d2))]

    def actions_root_to(self, node):
        """Collect incoming edge actions along root -> node."""
        chain = []
        while node.parent is not None:
            chain.append(node.action)
            node = node.parent
        chain.reverse()
        return chain

    def replay_to(self, node):
        """Reset to root and roll the stored action sequence forward.
        Returns the reconstructed position (== node.pos under deterministic sim)."""
        self.resets += 1
        pos = self.nodes[0].pos.copy()          # reset to root
        for a in self.actions_root_to(node):
            pos = step(pos, a)
            self.env_steps += 1
            self.replay_steps += 1
        if self.verify_replay:
            assert np.allclose(pos, node.pos, atol=1e-9), "replay did not reconstruct node"
        return pos

    def steer(self, from_pos, to_pos):
        """Action that moves from `from_pos` toward `to_pos` by one step.
        With noise_std>0 the direction is perturbed -> a 'random action toward'."""
        d = to_pos - from_pos
        n = np.linalg.norm(d)
        if n < 1e-12:
            theta = self.rng.uniform(0, 2 * np.pi)
            unit = np.array([np.cos(theta), np.sin(theta)])
        else:
            unit = d / n
            if self.noise_std > 0:
                ang = np.arctan2(unit[1], unit[0]) + self.rng.normal(0, self.noise_std)
                unit = np.array([np.cos(ang), np.sin(ang)])
        return unit * min(self.step_size, max(n, self.step_size))

    def _build_traversal(self, near: Node, new_pos: np.ndarray) -> np.ndarray:
        """Positions visited this iteration: root → ... → near → new_pos."""
        chain = []
        n = near
        while n is not None:
            chain.append(n.pos.copy())
            n = n.parent
        chain.reverse()
        chain.append(np.asarray(new_pos, dtype=float))
        return np.array(chain)  # shape (depth+2, 2)

    # --- main loop ---
    def plan(self):
        current = self.nodes[0]     # the node the agent is physically standing on
        for it in range(self.max_iter):
            s = self.maze.sample(self.goal, self.goal_bias, self.rng)
            near = self.nearest(s)

            if near.id != current.id:
                # We are NOT at the node we must extend from: pay the replay cost.
                self.replay_to(near)
                current = near
            # else: nearest happens to be where we already stand -> no replay (free).

            action = self.steer(near.pos, s)
            new_pos = step(near.pos, action)
            self.env_steps += 1
            self.extension_steps += 1

            traversal = self._build_traversal(near, new_pos)

            if self.maze.segment_free(near.pos, new_pos):
                child = Node(len(self.nodes), new_pos, near, action)
                self.nodes.append(child)
                current = child        # agent now stands at the freshly added node
                self.history.append(
                    {"n_nodes": len(self.nodes), "traversal": traversal, "success": True}
                )
                if np.linalg.norm(new_pos - self.goal) <= self.goal_radius:
                    return self._result(success=True, goal_node=child, iters=it + 1)
            else:
                # extension rejected; agent stays at `near`
                current = near
                self.history.append(
                    {"n_nodes": len(self.nodes), "traversal": traversal, "success": False}
                )

        return self._result(success=False, goal_node=None, iters=self.max_iter)

    def _result(self, success, goal_node, iters):
        path = []
        if goal_node is not None:
            n = goal_node
            while n is not None:
                path.append(n)
                n = n.parent
            path.reverse()
        return {
            "success": success,
            "iterations": iters,
            "nodes": len(self.nodes),
            "path_nodes": [n.id for n in path],
            "path_actions": [n.action for n in path if n.action is not None],
            "path_positions": np.array([n.pos for n in path]) if path else None,
            "env_steps": self.env_steps,
            "replay_steps": self.replay_steps,
            "extension_steps": self.extension_steps,
            "resets": self.resets,
        }


# --------------------------------------------------------------------------- #
# Visualization helpers
# --------------------------------------------------------------------------- #
def _draw_maze(ax, maze):
    for xmin, ymin, xmax, ymax in maze.walls:
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               facecolor="#555", edgecolor="none", zorder=2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")


def plot_static(maze, planner, result, fname):
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_maze(ax, maze)
    for n in planner.nodes:
        if n.parent is not None:
            ax.plot([n.parent.pos[0], n.pos[0]], [n.parent.pos[1], n.pos[1]],
                    color="#9ec5fe", lw=0.5, zorder=1)
    if result["path_positions"] is not None:
        p = result["path_positions"]
        ax.plot(p[:, 0], p[:, 1], color="#d62828", lw=2.2, zorder=3, label="solution path")
    ax.scatter(*planner.start, c="#2a9d8f", s=90, zorder=4, label="start")
    ax.scatter(*planner.goal, c="#e9c46a", s=180, marker="*",
               edgecolor="k", zorder=4, label="goal")
    ax.set_title("Voronoi-biased Go-Explore (phase 1) — four-room maze")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_gif(maze, planner, result, gif_path, animate_every, fps):
    """Animate per-iteration agent traversal: replay path + extension step."""
    history = planner.history
    total_iters = len(history)

    frame_indices = list(range(animate_every - 1, total_iters, animate_every))
    if not frame_indices or frame_indices[-1] < total_iters - 1:
        frame_indices.append(total_iters - 1)
    n_frames = len(frame_indices)

    if n_frames > 500:
        warnings.warn(
            f"Estimated GIF frame count = {n_frames} (> 500). "
            "Output may be large. Increase --animate_every to reduce.",
            stacklevel=2,
        )

    # Pre-build edge array for the full tree (shape: (n_nodes-1, 2, 2)).
    # Sliced to [:n_nodes-1] per frame since the tree only grows.
    edge_arr = np.array(
        [[[n.parent.pos[0], n.parent.pos[1]], [n.pos[0], n.pos[1]]]
         for n in planner.nodes[1:]],
        dtype=float,
    )

    success = result["success"]
    status  = "SUCCESS" if success else "FAILED"

    fig, ax = plt.subplots(figsize=(7, 7))

    def _update(fi):
        ax.cla()
        _draw_maze(ax, maze)

        entry    = history[frame_indices[fi]]
        n_segs   = entry["n_nodes"] - 1
        traversal = entry["traversal"]   # shape (depth+2, 2)
        iter_num = frame_indices[fi] + 1

        # Tree edges up to this iteration
        if n_segs > 0:
            ax.add_collection(LineCollection(
                edge_arr[:n_segs],
                colors="#9ec5fe", linewidths=0.5, alpha=0.5, zorder=1,
            ))

        # Replay path: root → near (all points except the last, which is new_pos)
        replay_pts = traversal[:-1]   # shape (depth+1, 2)
        if len(replay_pts) > 1:
            ax.plot(replay_pts[:, 0], replay_pts[:, 1],
                    color="#f4a261", lw=1.5, alpha=0.85, zorder=3, label="replay")

        # Extension edge: near → new_pos
        ext_color = "#2a9d8f" if entry["success"] else "#e63946"
        ax.plot(traversal[-2:, 0], traversal[-2:, 1],
                color=ext_color, lw=2.0, zorder=4,
                label="extend (ok)" if entry["success"] else "extend (blocked)")

        # Agent dot at attempted new position
        ax.scatter(traversal[-1, 0], traversal[-1, 1],
                   c="#e76f51", s=35, zorder=5)

        # Start / goal markers
        ax.scatter(*planner.start, c="#2a9d8f", s=90, zorder=6, label="start")
        ax.scatter(*planner.goal, c="#e9c46a", s=180, marker="*",
                   edgecolor="k", zorder=6, label="goal")

        # Solution path on the final frame only
        if fi == n_frames - 1 and result["path_positions"] is not None:
            p = result["path_positions"]
            ax.plot(p[:, 0], p[:, 1], color="#d62828", lw=2.2, zorder=7, label="solution")

        ax.set_title(
            f"Go-Explore RRT {status} — iter {iter_num}/{total_iters} "
            f"| nodes={entry['n_nodes']} (frame {fi + 1}/{n_frames})"
        )
        ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
        return []

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=100, blit=False, repeat=False)
    anim.save(gif_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved: {gif_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--start", default="0.10,0.10",
        help="Start position as 'x,y' in [0,1]x[0,1] (default: 0.10,0.10)",
    )
    p.add_argument(
        "--goal", default="0.90,0.90",
        help="Goal position as 'x,y' in [0,1]x[0,1] (default: 0.90,0.90)",
    )
    p.add_argument(
        "--step_size", type=float, default=0.03,
        help="RRT step size ε in maze units (default: 0.03)",
    )
    p.add_argument(
        "--goal_bias", type=float, default=0.10,
        help="Probability of sampling the goal each iteration (default: 0.10)",
    )
    p.add_argument(
        "--goal_radius", type=float, default=0.03,
        help="Distance to goal that triggers success (default: 0.03)",
    )
    p.add_argument(
        "--noise_std", type=float, default=0.0,
        help="Std-dev of angular noise added to steering direction (default: 0.0). "
             "Set >0 (e.g. 0.3) for 'random action toward sample'.",
    )
    p.add_argument(
        "--max_iter", type=int, default=8000,
        help="Maximum planner iterations (default: 8000)",
    )
    p.add_argument(
        "--wall_thickness", type=float, default=0.02,
        help="Thickness of maze walls in maze units (default: 0.02)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility. Omit for a different tree each run.",
    )
    p.add_argument(
        "--viz", choices=["static", "animate", "both", "none"], default="both",
        help="Output: 'static' PNG, 'animate' GIF, 'both', or 'none' (default: both)",
    )
    p.add_argument(
        "--animate_every", type=int, default=1,
        help="Record one GIF frame every N iterations (default: 20)",
    )
    p.add_argument(
        "--fps", type=int, default=1,
        help="GIF playback speed in frames per second (default: 1)",
    )
    p.add_argument(
        "--out", default="maze_tree",
        help="Output filename stem. A timestamp is appended automatically (default: maze_tree)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    start = np.array([float(v) for v in args.start.split(",")], dtype=float)
    goal  = np.array([float(v) for v in args.goal.split(",")],  dtype=float)

    rng = np.random.default_rng(args.seed)
    actual_seed = rng.bit_generator.state["state"]["state"] if args.seed is None else args.seed

    maze    = FourRoomMaze(wall_thickness=args.wall_thickness)
    planner = GoExploreRRT(
        maze,
        start        = start,
        goal         = goal,
        rng          = rng,
        step_size    = args.step_size,
        goal_bias    = args.goal_bias,
        goal_radius  = args.goal_radius,
        noise_std    = args.noise_std,
        max_iter     = args.max_iter,
    )

    print(f"Start  : {tuple(start)}   Goal : {tuple(goal)}")
    print(
        f"Config : step={args.step_size}  bias={args.goal_bias}  "
        f"radius={args.goal_radius}  noise={args.noise_std}  "
        f"max_iter={args.max_iter}  seed={actual_seed}"
    )
    print("Planning...")

    t0        = time.perf_counter()
    res       = planner.plan()
    plan_time = time.perf_counter() - t0

    print("\n--- result ---")
    print(f"success         : {res['success']}")
    print(f"iterations      : {res['iterations']}")
    print(f"tree nodes      : {res['nodes']}")
    print(f"solution length : {len(res['path_nodes'])} nodes")
    print(f"plan time       : {plan_time:.4f} s")
    print(f"env steps total : {res['env_steps']}")
    print(f"  replay steps  : {res['replay_steps']}")
    print(f"  extension     : {res['extension_steps']}")
    print(f"resets to root  : {res['resets']}")
    if res["nodes"] > 1:
        print(f"env_steps / node: {res['env_steps'] / res['nodes']:.1f}   "
              f"(teleport-RRT would be ~1.0)")

    if args.viz != "none":
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = Path(__file__).parent

        if args.viz in ("static", "both"):
            images_dir = script_dir / "images"
            images_dir.mkdir(exist_ok=True)
            print("Generating static image...")
            plot_static(maze, planner, res, images_dir / f"{args.out}_{ts}.png")

        if args.viz in ("animate", "both"):
            gifs_dir = script_dir / "gifs"
            gifs_dir.mkdir(exist_ok=True)
            print("Generating animation...")
            plot_gif(maze, planner, res,
                     gifs_dir / f"{args.out}_{ts}.gif",
                     animate_every=args.animate_every,
                     fps=args.fps)

    print("\nDone.")


if __name__ == "__main__":
    main()
