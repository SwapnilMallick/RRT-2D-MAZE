"""
rrt_bidirectional_maze_2d.py — Bidirectional RRT (B-RRT) path planning in a 2-D maze.

Two trees T_a (rooted at start) and T_b (rooted at goal) grow symmetrically.
Each iteration extends both trees by one epsilon step and checks whether they
have met. No goal biasing; T_b provides implicit directional pull on T_a.

Usage
-----
    python rrt_bidirectional_maze_2d.py [options]

Presets  (--maze preset:NAME)
-----------------------------
    empty    — no obstacles; trivial straight-line solution
    wall     — single vertical wall with a gap; narrow-passage navigation
    bugtrap  — U-shaped trap; start inside, goal past the closed wall (classic RRT stress test)
    rooms    — 4 rooms connected by narrow doorways

File mazes  (--maze file:PATH.npy)
-----------------------------------
    Load a boolean numpy array (True = obstacle, False = free). Shape: (H, W).

Output files  (all timestamped, written relative to this script's directory)
------------
    bidirectional_rrt/plans/<base>_<ts>.npz  — trees and path arrays for replay / analysis
    bidirectional_rrt/images/<base>_<ts>.png — static visualization (always saved)
    bidirectional_rrt/gifs/<base>_<ts>.gif   — animated exploration (--viz animate/both)

Examples
--------
    python rrt_bidirectional_maze_2d.py --maze preset:bugtrap --viz both
    python rrt_bidirectional_maze_2d.py --maze preset:rooms --step_size 3 --max_iters 8000
    python rrt_bidirectional_maze_2d.py --maze file:custom.npy --start "5,5" --goal "95,95"
    python rrt_bidirectional_maze_2d.py --maze preset:wall --seed 42 --viz static
"""

from __future__ import annotations

import argparse
import random as _pyrandom
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe; works without a display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# Nodes live in two separate lists (one per tree). tree_id=0 → T_a (start),
# tree_id=1 → T_b (goal). parent is an index into that same list.
@dataclass
class Node:
    pos:     np.ndarray       # shape (2,) — (x, y) in continuous grid units
    parent:  Optional[int]    # index into the same tree list; None for root
    tree_id: int              # 0 = T_a (start-rooted), 1 = T_b (goal-rooted)


@dataclass
class BRRTConfig:
    step_size:      float         = 5.0
    connect_tol:    Optional[float] = None   # None → resolved to step_size in main()
    max_iters:      int           = 5000
    collision_step: float         = 0.5
    seed:           Optional[int] = None


# ---------------------------------------------------------------------------
# Maze construction
# ---------------------------------------------------------------------------

def build_maze(spec: str, W: int, H: int) -> np.ndarray:
    """
    Return a boolean occupancy grid of shape (H, W).
    True = obstacle (C_obs), False = free (C_free).
    """
    if spec.startswith("file:"):
        g = np.load(spec[5:])
        return g.astype(bool)

    if not spec.startswith("preset:"):
        raise ValueError(
            f"Unknown maze spec {spec!r}. Use 'preset:NAME' or 'file:PATH.npy'."
        )

    name = spec[7:].lower()
    grid = np.zeros((H, W), dtype=bool)

    # 1-cell border walls
    grid[0, :]  = True
    grid[-1, :] = True
    grid[:, 0]  = True
    grid[:, -1] = True

    if name == "empty":
        pass

    elif name == "wall":
        cx     = W // 2
        gap_lo = H // 3
        gap_hi = 2 * H // 3
        grid[:gap_lo, cx] = True
        grid[gap_hi:,  cx] = True

    elif name == "bugtrap":
        # U-shaped trap opening to the right; start inside, goal outside.
        cx    = W // 2        # x of the right (open) edge of the trap
        cy    = H // 2        # vertical centre
        th    = H // 3        # half-height of the U in cells
        tw    = W // 3        # horizontal depth of the U in cells
        thick = 3             # wall thickness in cells
        # bottom bar
        grid[cy - th        : cy - th + thick, cx - tw : cx] = True
        # top bar
        grid[cy + th - thick: cy + th,         cx - tw : cx] = True
        # left wall
        grid[cy - th        : cy + th,         cx - tw : cx - tw + thick] = True

    elif name == "rooms":
        mh   = H // 2
        mw   = W // 2
        door = max(5, min(W, H) // 15)
        hd   = door // 2

        grid[mh, :] = True   # horizontal divider
        grid[:, mw] = True   # vertical divider

        # left rooms: door in horizontal divider at x ≈ mw/2
        dw_l = mw // 2
        grid[mh, max(1, dw_l - hd) : min(W - 1, dw_l + hd)] = False
        # right rooms: door in horizontal divider at x ≈ mw + (W-mw)/2
        dw_r = mw + (W - mw) // 2
        grid[mh, max(1, dw_r - hd) : min(W - 1, dw_r + hd)] = False
        # bottom rooms: door in vertical divider at y ≈ mh/2
        dh_b = mh // 2
        grid[max(1, dh_b - hd) : min(H - 1, dh_b + hd), mw] = False
        # top rooms: door in vertical divider at y ≈ mh + (H-mh)/2
        dh_t = mh + (H - mh) // 2
        grid[max(1, dh_t - hd) : min(H - 1, dh_t + hd), mw] = False

    else:
        raise ValueError(
            f"Unknown preset {name!r}. Choices: empty, wall, bugtrap, rooms."
        )

    return grid


# Default (start, goal) for each preset on a 100×100 grid.
# bugtrap: start inside the U-trap (x=35), goal past the closed left wall (x=10).
# The direct path start→goal is blocked by the left wall, so the robot must first
# exit rightward through the opening, loop around the top/bottom bar, and approach
# from the left — the canonical RRT stress test.
PRESET_DEFAULTS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "empty":   ((5.0,  5.0),  (95.0, 95.0)),
    "wall":    ((10.0, 50.0), (90.0, 50.0)),
    "bugtrap": ((35.0, 50.0), (10.0, 50.0)),
    "rooms":   ((12.0, 12.0), (88.0, 88.0)),
}


# ---------------------------------------------------------------------------
# Collision checking
# ---------------------------------------------------------------------------

def in_collision(p: np.ndarray, grid: np.ndarray) -> bool:
    """True if continuous point p=(x,y) is in collision or out of bounds."""
    H, W = grid.shape
    x, y = float(p[0]), float(p[1])
    if x < 0.0 or x >= W or y < 0.0 or y >= H:
        return True
    return bool(grid[int(y), int(x)])


def edge_in_collision(
    a: np.ndarray,
    b: np.ndarray,
    grid: np.ndarray,
    step: float,
) -> bool:
    """True if any sampled point along segment a→b is in collision."""
    d      = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return in_collision(a, grid)
    n = max(2, int(np.ceil(length / step)))
    for i in range(n + 1):
        if in_collision(a + (i / n) * d, grid):
            return True
    return False


# ---------------------------------------------------------------------------
# Bidirectional RRT planner
# ---------------------------------------------------------------------------

def brrt_plan(
    start: np.ndarray,
    goal:  np.ndarray,
    grid:  np.ndarray,
    cfg:   BRRTConfig,
    rng:   np.random.Generator,
) -> tuple[list[Node], list[Node], Optional[list[np.ndarray]], Optional[np.ndarray], int]:
    """
    Run Bidirectional RRT from start to goal.

    Two trees are maintained:
      trees[0]  = T_a, rooted at start  (Node.tree_id == 0)
      trees[1]  = T_b, rooted at goal   (Node.tree_id == 1)

    Each iteration:
      1. Sample q_rand uniformly (no goal bias — T_b provides directional pull).
      2. EXTEND(cur_tree, q_rand)  → q_new_cur
      3. If successful, EXTEND(oth_tree, q_new_cur) → q_new_oth
      4. If both succeeded and ||q_new_oth - q_new_cur|| < connect_tol
         and the bridge edge is collision-free → trees have met, reconstruct path.
      5. Swap cur/oth.

    Returns
    -------
    tree_a : list[Node]
        T_a, rooted at start (tree_id=0).
    tree_b : list[Node]
        T_b, rooted at goal (tree_id=1).
    path : list[np.ndarray] | None
        Ordered position waypoints from start to goal, or None on failure.
    meeting_point : np.ndarray | None
        Midpoint of the two meeting nodes, or None on failure.
    total_iters : int
        Number of loop iterations executed.
    """
    H, W = grid.shape
    connect_tol = cfg.connect_tol if cfg.connect_tol is not None else cfg.step_size

    # trees[0] = T_a (start-rooted), trees[1] = T_b (goal-rooted)
    trees: list[list[Node]] = [
        [Node(pos=start.copy(), parent=None, tree_id=0)],
        [Node(pos=goal.copy(),  parent=None, tree_id=1)],
    ]

    # Pre-allocated position arrays for vectorised nearest-neighbour search.
    # Each tree can grow by at most 1 node per iteration.
    max_nodes = cfg.max_iters + 2
    pos_arr: list[np.ndarray] = [
        np.empty((max_nodes, 2), dtype=float),
        np.empty((max_nodes, 2), dtype=float),
    ]
    pos_arr[0][0] = start
    pos_arr[1][0] = goal
    n_nodes = [1, 1]

    def extend(ti: int, q_target: np.ndarray) -> tuple[Optional[np.ndarray], int]:
        """EXTEND primitive: steer tree ti toward q_target by one step.

        Returns (q_new, new_idx) on success, (None, -1) on collision or degenerate.
        """
        ns     = n_nodes[ti]
        diff   = pos_arr[ti][:ns] - q_target
        sq     = diff[:, 0] ** 2 + diff[:, 1] ** 2
        ni     = int(np.argmin(sq))
        q_near = pos_arr[ti][ni]

        d    = q_target - q_near
        dist = float(np.linalg.norm(d))
        if dist < 1e-9:
            return None, -1
        q_new = (
            q_target.copy()
            if dist <= cfg.step_size
            else q_near + cfg.step_size * d / dist
        )

        if edge_in_collision(q_near, q_new, grid, cfg.collision_step):
            return None, -1

        new_idx          = n_nodes[ti]
        pos_arr[ti][new_idx] = q_new
        n_nodes[ti]     += 1
        trees[ti].append(Node(pos=q_new.copy(), parent=ni, tree_id=ti))
        return q_new, new_idx

    # cur / oth index into trees[]. Swap at the end of every iteration so both
    # trees take turns leading.
    cur, oth = 0, 1

    for total_iters in range(1, cfg.max_iters + 1):
        # 1. Sample uniformly — no goal bias needed; T_b provides implicit pull.
        q_rand = rng.uniform([0.0, 0.0], [float(W), float(H)])

        # 2. Extend the current (leading) tree toward q_rand.
        q_new_cur, cur_idx = extend(cur, q_rand)

        if q_new_cur is not None:
            # 3. Extend the other tree toward the new node just added.
            q_new_oth, oth_idx = extend(oth, q_new_cur)

            if q_new_oth is not None:
                dist_meet = float(np.linalg.norm(q_new_oth - q_new_cur))
                if (dist_meet < connect_tol
                        and not edge_in_collision(
                            q_new_cur, q_new_oth, grid, cfg.collision_step)):
                    # Trees have met. Identify which index belongs to which tree.
                    if cur == 0:
                        a_meet_idx, b_meet_idx = cur_idx, oth_idx
                    else:
                        a_meet_idx, b_meet_idx = oth_idx, cur_idx

                    meeting_point = (q_new_cur + q_new_oth) * 0.5

                    # Walk T_a from meeting point up to start, then reverse.
                    path_a: list[np.ndarray] = []
                    idx: Optional[int] = a_meet_idx
                    while idx is not None:
                        path_a.append(trees[0][idx].pos.copy())
                        idx = trees[0][idx].parent
                    path_a.reverse()   # now: start → ... → q_new_a

                    # Walk T_b from meeting point up to goal (already goal-direction).
                    path_b: list[np.ndarray] = []
                    idx = b_meet_idx
                    while idx is not None:
                        path_b.append(trees[1][idx].pos.copy())
                        idx = trees[1][idx].parent
                    # path_b: q_new_b → ... → goal

                    full_path = path_a + path_b
                    return trees[0], trees[1], full_path, meeting_point, total_iters

        # 4. Swap so the other tree leads next iteration.
        cur, oth = oth, cur

    return trees[0], trees[1], None, None, cfg.max_iters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def path_length_pos(path: list[np.ndarray]) -> float:
    """Sum of Euclidean distances along a path given as a list of positions."""
    return sum(
        float(np.linalg.norm(path[i] - path[i - 1]))
        for i in range(1, len(path))
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _draw_base(
    ax: plt.Axes,
    grid: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
) -> None:
    """Draw the occupancy grid, start marker, and goal marker."""
    H, W = grid.shape
    ax.imshow(
        grid,
        cmap="binary",
        origin="lower",
        extent=[0, W, 0, H],
        interpolation="nearest",
        zorder=0,
    )
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.plot(start[0], start[1], "gs", markersize=10, zorder=5, label="Start")
    ax.plot(goal[0],  goal[1],  "y*", markersize=14, zorder=5, label="Goal")


def visualize(
    tree_a:        list[Node],
    tree_b:        list[Node],
    path:          Optional[list[np.ndarray]],
    meeting_point: Optional[np.ndarray],
    grid:          np.ndarray,
    start:         np.ndarray,
    goal:          np.ndarray,
    out_png:       str,
    out_gif:       str,
    viz_mode:      str,
    animate_every: int,
    plan_time:     float,
    fps:           int,
) -> None:
    """Save static PNG and/or animated GIF depending on viz_mode."""
    success = path is not None
    p_len   = path_length_pos(path) if path else 0.0
    status  = "SUCCESS" if success else "FAILED"
    na, nb  = len(tree_a), len(tree_b)

    # ---------------------------------------------------------------- static
    if viz_mode in ("static", "both"):
        fig, ax = plt.subplots(figsize=(8, 8))
        _draw_base(ax, grid, start, goal)

        # T_a edges in blue, T_b edges in orange
        for node in tree_a:
            if node.parent is not None:
                par = tree_a[node.parent]
                ax.plot(
                    [par.pos[0], node.pos[0]],
                    [par.pos[1], node.pos[1]],
                    color="steelblue", linewidth=0.5, alpha=0.5, zorder=1,
                )
        for node in tree_b:
            if node.parent is not None:
                par = tree_b[node.parent]
                ax.plot(
                    [par.pos[0], node.pos[0]],
                    [par.pos[1], node.pos[1]],
                    color="darkorange", linewidth=0.5, alpha=0.5, zorder=1,
                )

        # Solution path in thick red (on top)
        if path:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, "-r", linewidth=2.5, zorder=3, label="Path")

        # Meeting point in magenta
        if meeting_point is not None:
            ax.plot(
                meeting_point[0], meeting_point[1],
                "mo", markersize=9, zorder=4, label="Meeting pt",
            )

        # Legend proxy patches for the two trees
        from matplotlib.lines import Line2D
        handles, _ = ax.get_legend_handles_labels()
        handles += [
            Line2D([0], [0], color="steelblue",  linewidth=1.5, label="T_a (start)"),
            Line2D([0], [0], color="darkorange", linewidth=1.5, label="T_b (goal)"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8)

        n_waypoints = len(path) if path else 0
        ax.set_title(
            f"Bidirectional RRT: |T_a|={na}, |T_b|={nb}, "
            f"path={n_waypoints} waypoints ({p_len:.1f} units), t={plan_time:.2f}s",
            fontsize=9,
        )
        out = f"{out_png}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

    # -------------------------------------------------------------- animate
    if viz_mode in ("animate", "both"):
        # Both trees grow at roughly equal rates (they alternate per iteration).
        # We animate up to index k in each tree simultaneously.
        max_k = max(na, nb)
        frame_sizes: list[int] = list(range(animate_every, max_k, animate_every))
        if not frame_sizes or frame_sizes[-1] < max_k:
            frame_sizes.append(max_k)
        n_frames = len(frame_sizes)

        total_nodes = na + nb
        if n_frames > 500:
            warnings.warn(
                f"Estimated GIF frame count = {n_frames} (> 500). "
                "Output may be very large. Increase --animate_every to reduce.",
                stacklevel=2,
            )

        # Pre-build edge segment arrays (shape (n-1, 2, 2)) for both trees so
        # _update can slice cheaply and pass a single LineCollection per tree
        # rather than one plot() call per edge.
        def _edge_array(tree: list[Node]) -> np.ndarray:
            if len(tree) <= 1:
                return np.empty((0, 2, 2), dtype=float)
            return np.array(
                [[[tree[nd.parent].pos[0], tree[nd.parent].pos[1]],
                  [nd.pos[0],              nd.pos[1]]]
                 for nd in tree[1:]],
                dtype=float,
            )

        edge_arr_a = _edge_array(tree_a)  # shape (na-1, 2, 2)
        edge_arr_b = _edge_array(tree_b)  # shape (nb-1, 2, 2)

        fig_a, ax_a = plt.subplots(figsize=(7, 7))

        # Each _update call clears and redraws the axes from scratch so that
        # every saved GIF frame is a self-contained image. The incremental
        # "accumulate artists" approach produces frames that look correct on
        # first play, but GIF viewers that loop ignore repeat=False and use
        # the "do not dispose" delta-patch method, causing the path drawn on
        # the last frame to bleed into subsequent loop iterations.
        def _update(frame_idx: int) -> list:
            ax_a.cla()
            _draw_base(ax_a, grid, start, goal)

            k = frame_sizes[frame_idx]
            n_a_segs = max(0, min(k, na) - 1)
            n_b_segs = max(0, min(k, nb) - 1)

            if n_a_segs > 0:
                ax_a.add_collection(LineCollection(
                    edge_arr_a[:n_a_segs],
                    colors="steelblue", linewidths=0.4, alpha=0.35, zorder=1,
                ))
            if n_b_segs > 0:
                ax_a.add_collection(LineCollection(
                    edge_arr_b[:n_b_segs],
                    colors="darkorange", linewidths=0.4, alpha=0.35, zorder=1,
                ))

            # Path and meeting point only on the final frame, once both trees
            # are fully drawn and the connection is visually complete.
            if frame_idx == n_frames - 1:
                if path:
                    px = [p[0] for p in path]
                    py = [p[1] for p in path]
                    ax_a.plot(px, py, "-r", linewidth=2.5, zorder=3)
                if meeting_point is not None:
                    ax_a.plot(
                        meeting_point[0], meeting_point[1],
                        "mo", markersize=9, zorder=4,
                    )

            shown = min(k, na) + min(k, nb)
            ax_a.set_title(
                f"B-RRT {status} — {shown}/{total_nodes} nodes "
                f"(frame {frame_idx + 1}/{n_frames})"
            )
            return []

        anim = FuncAnimation(
            fig_a, _update,
            frames=n_frames, interval=100, blit=False, repeat=False,
        )
        out = f"{out_gif}.gif"
        anim.save(out, writer="pillow", fps=fps)
        plt.close(fig_a)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--maze", default="preset:bugtrap",
        help=(
            "Maze spec: 'preset:NAME' (empty/wall/bugtrap/rooms) or "
            "'file:PATH.npy' for a precomputed boolean grid. "
            "Default: preset:bugtrap"
        ),
    )
    p.add_argument(
        "--width", type=int, default=100,
        help="Grid width W in cells (default: 100)",
    )
    p.add_argument(
        "--height", type=int, default=100,
        help="Grid height H in cells (default: 100)",
    )
    p.add_argument(
        "--start", default=None,
        help="Start position as 'x,y' (float, continuous coords). "
             "Default: preset-dependent.",
    )
    p.add_argument(
        "--goal", default=None,
        help="Goal position as 'x,y' (float, continuous coords). "
             "Default: preset-dependent.",
    )
    p.add_argument(
        "--step_size", type=float, default=5.0,
        help="RRT step size ε in grid units (default: 5.0)",
    )
    p.add_argument(
        "--connect_tol", type=float, default=None,
        help=(
            "Meeting-distance tolerance for declaring the two trees connected "
            "(default: step_size). Increase if trees rarely meet."
        ),
    )
    p.add_argument(
        "--max_iters", type=int, default=5000,
        help="Maximum B-RRT loop iterations (default: 5000)",
    )
    p.add_argument(
        "--collision_step", type=float, default=0.5,
        help="Edge collision-check resolution in grid units (default: 0.5)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility. Omit for a different tree each run.",
    )
    p.add_argument(
        "--viz", choices=["static", "animate", "both"], default="both",
        help="Visualization output mode (default: both)",
    )
    p.add_argument(
        "--animate_every", type=int, default=1,
        help="Record one animation frame every N nodes added per tree (default: 20)",
    )
    p.add_argument(
        "--fps", type=int, default=5,
        help="GIF playback speed in frames per second (default: 5). Lower = slower.",
    )
    p.add_argument(
        "--out", default="maze_bidirectional_rrt",
        help="Base name for output files. A timestamp is appended automatically. "
             "(default: maze_bidirectional_rrt)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Build maze
    grid = build_maze(args.maze, args.width, args.height)
    H, W = grid.shape

    # Resolve start / goal with preset-aware defaults
    preset_name = args.maze[7:] if args.maze.startswith("preset:") else "empty"
    s_def, g_def = PRESET_DEFAULTS.get(
        preset_name, ((5.0, 5.0), (float(W - 5), float(H - 5)))
    )
    start = np.array(
        [float(v) for v in args.start.split(",")] if args.start else s_def,
        dtype=float,
    )
    goal = np.array(
        [float(v) for v in args.goal.split(",")] if args.goal else g_def,
        dtype=float,
    )

    if in_collision(start, grid):
        raise SystemExit(f"ERROR: start {tuple(start)} is in collision or out of bounds.")
    if in_collision(goal, grid):
        raise SystemExit(f"ERROR: goal {tuple(goal)} is in collision or out of bounds.")

    # Resolve connect_tol: default to step_size if not specified.
    connect_tol = args.connect_tol if args.connect_tol is not None else args.step_size

    cfg = BRRTConfig(
        step_size      = args.step_size,
        connect_tol    = connect_tol,
        max_iters      = args.max_iters,
        collision_step = args.collision_step,
        seed           = args.seed,
    )
    _pyrandom.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    actual_seed = rng.bit_generator.state["state"]["state"] if cfg.seed is None else cfg.seed

    print(f"Maze   : {args.maze}  ({W}x{H})")
    print(f"Start  : {tuple(start)}   Goal : {tuple(goal)}")
    print(
        f"Config : step={cfg.step_size}  connect_tol={cfg.connect_tol}  "
        f"max_iters={cfg.max_iters}  seed={actual_seed}"
    )
    print("Planning...")

    # Plan (only this block is timed)
    t0 = time.perf_counter()
    tree_a, tree_b, path, meeting_point, total_iters = brrt_plan(
        start, goal, grid, cfg, rng
    )
    plan_time = time.perf_counter() - t0

    # Console summary
    success = path is not None
    p_len   = path_length_pos(path) if path else 0.0
    na, nb  = len(tree_a), len(tree_b)

    print("\n=== Bidirectional RRT Summary ===")
    print(f"  seed           : {actual_seed}")
    print(f"  maze           : {args.maze}")
    print(f"  start          : {tuple(start)}")
    print(f"  goal           : {tuple(goal)}")
    print(f"  iterations     : {total_iters}")
    print(f"  |T_a| (start)  : {na} nodes")
    print(f"  |T_b| (goal)   : {nb} nodes")
    print(f"  total nodes    : {na + nb}")
    print(f"  plan time      : {plan_time:.4f} s")
    print(f"  success        : {success}")
    if success:
        wps = [tuple(p.round(2).tolist()) for p in path]
        print(f"  path length    : {p_len:.2f}")
        print(f"  waypoints ({len(wps):>3}): {wps}")
        print(f"  meeting point  : {tuple(meeting_point.round(2).tolist())}")
    else:
        print("  No path found within max_iters.")

    # Save plan.npz
    save_kw: dict = dict(
        tree_a_positions=np.array([nd.pos for nd in tree_a]),
        tree_a_parents=np.array(
            [nd.parent if nd.parent is not None else -1 for nd in tree_a],
            dtype=np.int32,
        ),
        tree_b_positions=np.array([nd.pos for nd in tree_b]),
        tree_b_parents=np.array(
            [nd.parent if nd.parent is not None else -1 for nd in tree_b],
            dtype=np.int32,
        ),
        success=np.bool_(success),
    )
    if path is not None:
        save_kw["path"]          = np.array(path)
        save_kw["meeting_point"] = meeting_point
    # Build timestamped output paths, rooted at this script's directory so
    # outputs land in the right place regardless of the working directory.
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    base     = f"{args.out}_{ts}"
    root     = Path(__file__).parent
    plan_dir = root / "plans"
    img_dir  = root / "images"
    gif_dir  = root / "gifs"
    for d in (plan_dir, img_dir, gif_dir):
        d.mkdir(parents=True, exist_ok=True)

    plan_path = plan_dir / f"{base}.npz"
    out_png   = str(img_dir / base)   # visualize appends .png
    out_gif   = str(gif_dir / base)   # visualize appends .gif

    np.savez(plan_path, **save_kw)
    print(f"\n  Saved: {plan_path}")

    # Visualize
    print("Generating visualizations...")
    visualize(
        tree_a        = tree_a,
        tree_b        = tree_b,
        path          = path,
        meeting_point = meeting_point,
        grid          = grid,
        start         = start,
        goal          = goal,
        out_png       = out_png,
        out_gif       = out_gif,
        viz_mode      = args.viz,
        animate_every = args.animate_every,
        plan_time     = plan_time,
        fps           = args.fps,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
