"""
rrt_maze_2d.py — RRT path planning in a 2-D maze with matplotlib visualization.

Usage
-----
    python rrt_maze_2d.py [options]

Presets  (--maze preset:NAME)
-----------------------------
    empty    — no obstacles; trivial straight-line solution
    wall     — single vertical wall with a gap; narrow-passage navigation
    bugtrap  — U-shaped trap; start inside, goal past the closed wall (classic RRT stress test)
    rooms    — 4 rooms connected by narrow doorways

File mazes  (--maze file:PATH.npy)
-----------------------------------
    Load a boolean numpy array (True = obstacle, False = free). Shape: (H, W).

Output files  (timestamped, written relative to the script directory)
----------------------------------------------------------------------
    plans/<stem>_<ts>.npz    — tree and path arrays for replay / analysis
    images/<stem>_<ts>.png   — static visualization (always saved)
    gifs/<stem>_<ts>.gif     — animated exploration (when --viz includes animate)

    <ts>   = YYYYMMDD_HHMMSS timestamp of the run
    <stem> = value of --out (default: maze_rrt)

Examples
--------
    python rrt_maze_2d.py --maze preset:bugtrap --viz both
    python rrt_maze_2d.py --maze preset:rooms --step_size 3 --max_iters 8000
    python rrt_maze_2d.py --maze file:custom.npy --start "5,5" --goal "95,95"
    python rrt_maze_2d.py --maze preset:wall --seed 42 --viz static
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
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Node:
    pos: np.ndarray        # shape (2,) — (x, y) in continuous grid units
    parent: Optional[int]  # index into the tree list; None for the root


@dataclass
class RRTConfig:
    step_size:      float = 5.0
    goal_threshold: float = 5.0
    goal_bias:      float = 0.05
    max_iters:      int   = 5000
    collision_step: float = 0.5
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
# RRT planner
# ---------------------------------------------------------------------------

def rrt_plan(
    start: np.ndarray,
    goal:  np.ndarray,
    grid:  np.ndarray,
    cfg:   RRTConfig,
    rng:   np.random.Generator,
) -> tuple[list[Node], Optional[list[int]], int]:
    """
    Run RRT from start to goal.

    Returns
    -------
    tree : list[Node]
        Full tree (all nodes) built during planning.
    path_indices : list[int] | None
        Indices into `tree` from root to goal, or None on failure.
    total_iters : int
        Total number of RRT loop iterations executed (includes rejected ones).
    """
    H, W = grid.shape

    # Pre-allocate position array for vectorised nearest-neighbour search.
    pos = np.empty((cfg.max_iters + 2, 2), dtype=float)
    pos[0] = start
    n_nodes = 1

    tree: list[Node] = [Node(pos=start.copy(), parent=None)]
    total_iters = 0

    for _ in range(cfg.max_iters):
        total_iters += 1

        # 1. Sample q_rand
        if rng.random() < cfg.goal_bias:
            q_rand = goal
        else:
            q_rand = rng.uniform([0.0, 0.0], [float(W), float(H)])

        # 2. Nearest neighbour (Euclidean, vectorised)
        diff    = pos[:n_nodes] - q_rand
        sq_dist = diff[:, 0] ** 2 + diff[:, 1] ** 2
        ni      = int(np.argmin(sq_dist))
        q_near  = pos[ni]

        # 3. Steer
        direction = q_rand - q_near
        dist      = float(np.linalg.norm(direction))
        if dist < 1e-9:
            continue
        q_new = (
            q_rand.copy()
            if dist <= cfg.step_size
            else q_near + cfg.step_size * direction / dist
        )

        # 4. Collision check edge q_near → q_new
        if edge_in_collision(q_near, q_new, grid, cfg.collision_step):
            continue

        # 5. Add node to tree
        new_idx         = n_nodes
        pos[new_idx]    = q_new
        n_nodes        += 1
        tree.append(Node(pos=q_new.copy(), parent=ni))

        # 6. Goal check
        if float(np.linalg.norm(q_new - goal)) < cfg.goal_threshold:
            if not edge_in_collision(q_new, goal, grid, cfg.collision_step):
                goal_idx = n_nodes
                tree.append(Node(pos=goal.copy(), parent=new_idx))
                # Trace path back to root
                path: list[int] = []
                idx: Optional[int] = goal_idx
                while idx is not None:
                    path.append(idx)
                    idx = tree[idx].parent
                path.reverse()
                return tree, path, total_iters

    return tree, None, total_iters


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def path_length(tree: list[Node], indices: list[int]) -> float:
    """Sum of Euclidean distances along the path."""
    return sum(
        float(np.linalg.norm(tree[indices[i]].pos - tree[indices[i - 1]].pos))
        for i in range(1, len(indices))
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
    tree:            list[Node],
    path_indices:    Optional[list[int]],
    grid:            np.ndarray,
    start:           np.ndarray,
    goal:            np.ndarray,
    img_path:        Path,
    gif_path:        Path,
    viz_mode:        str,
    animate_every:   int,
    plan_time:       float,
    total_iters:     int,
    fps:             int,
) -> None:
    """Save static PNG and/or animated GIF depending on viz_mode."""
    success = path_indices is not None
    p_len   = path_length(tree, path_indices) if path_indices else 0.0
    status  = "SUCCESS" if success else "FAILED"

    # ---------------------------------------------------------------- static
    if viz_mode in ("static", "both"):
        fig, ax = plt.subplots(figsize=(8, 8))
        _draw_base(ax, grid, start, goal)

        # All tree edges in thin gray
        for node in tree:
            if node.parent is not None:
                par = tree[node.parent]
                ax.plot(
                    [par.pos[0], node.pos[0]],
                    [par.pos[1], node.pos[1]],
                    color="gray", linewidth=0.5, alpha=0.4, zorder=1,
                )

        # Solution path in thick red
        if path_indices:
            px = [tree[i].pos[0] for i in path_indices]
            py = [tree[i].pos[1] for i in path_indices]
            ax.plot(px, py, "-r", linewidth=2.5, zorder=3, label="Path")

        ax.set_title(
            f"RRT {status} | iters={total_iters} | nodes={len(tree)} | "
            f"path_len={p_len:.1f} | time={plan_time:.3f}s",
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=8)
        out = img_path
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

    # -------------------------------------------------------------- animate
    if viz_mode in ("animate", "both"):
        # Build the list of tree sizes at which to capture frames.
        frame_sizes: list[int] = list(range(animate_every, len(tree), animate_every))
        if not frame_sizes or frame_sizes[-1] < len(tree):
            frame_sizes.append(len(tree))
        n_frames = len(frame_sizes)

        if n_frames > 500:
            warnings.warn(
                f"Estimated GIF frame count = {n_frames} (> 500). "
                "Output may be very large. Increase --animate_every to reduce.",
                stacklevel=2,
            )

        fig_a, ax_a = plt.subplots(figsize=(7, 7))
        _draw_base(ax_a, grid, start, goal)
        title_h = ax_a.set_title("")

        # Mutable counter so _update can draw only *new* edges each frame.
        drawn = [0]

        def _update(frame_idx: int) -> list:
            n = frame_sizes[frame_idx]
            for i in range(max(1, drawn[0]), n):
                nd = tree[i]
                if nd.parent is not None:
                    pp = tree[nd.parent]
                    ax_a.plot(
                        [pp.pos[0], nd.pos[0]],
                        [pp.pos[1], nd.pos[1]],
                        color="steelblue", linewidth=0.4, alpha=0.35, zorder=1,
                    )
            drawn[0] = n

            # Final frame: overlay solution path
            if frame_idx == n_frames - 1 and path_indices:
                px = [tree[i].pos[0] for i in path_indices]
                py = [tree[i].pos[1] for i in path_indices]
                ax_a.plot(px, py, "-r", linewidth=2.5, zorder=3)

            title_h.set_text(
                f"RRT {status} — {n}/{len(tree)} nodes "
                f"(frame {frame_idx + 1}/{n_frames})"
            )
            return [title_h]

        anim = FuncAnimation(
            fig_a, _update,
            frames=n_frames, interval=100, blit=False, repeat=False,
        )
        out = gif_path
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
        "--goal_threshold", type=float, default=5.0,
        help="Distance to goal that triggers a direct connection (default: 5.0)",
    )
    p.add_argument(
        "--goal_bias", type=float, default=0.05,
        help="Probability of sampling the goal directly each iteration (default: 0.05)",
    )
    p.add_argument(
        "--max_iters", type=int, default=5000,
        help="Maximum RRT loop iterations (default: 5000)",
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
        help="Record one animation frame every N added nodes (default: 20)",
    )
    p.add_argument(
        "--fps", type=int, default=5,
        help="GIF playback speed in frames per second (default: 5). Lower = slower.",
    )
    p.add_argument(
        "--out", default="maze_rrt",
        help="Base filename stem (default: maze_rrt). A timestamp is appended automatically.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Timestamped output paths, relative to the script's own directory
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = Path(__file__).parent
    plans_dir  = script_dir / "plans"
    images_dir = script_dir / "images"
    gifs_dir   = script_dir / "gifs"
    for d in (plans_dir, images_dir, gifs_dir):
        d.mkdir(exist_ok=True)
    stem      = args.out
    plan_path = plans_dir  / f"{stem}_{ts}.npz"
    img_path  = images_dir / f"{stem}_{ts}.png"
    gif_path  = gifs_dir   / f"{stem}_{ts}.gif"

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

    # Config + seeds
    cfg = RRTConfig(
        step_size      = args.step_size,
        goal_threshold = args.goal_threshold,
        goal_bias      = args.goal_bias,
        max_iters      = args.max_iters,
        collision_step = args.collision_step,
        seed           = args.seed,
    )
    _pyrandom.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    # Retrieve the actual seed used so it can be printed and reproduced later.
    actual_seed = rng.bit_generator.state["state"]["state"] if cfg.seed is None else cfg.seed

    print(f"Maze   : {args.maze}  ({W}x{H})")
    print(f"Start  : {tuple(start)}   Goal : {tuple(goal)}")
    print(
        f"Config : step={cfg.step_size}  threshold={cfg.goal_threshold}  "
        f"bias={cfg.goal_bias}  max_iters={cfg.max_iters}  seed={actual_seed}"
    )
    print("Planning...")

    # Plan (only this block is timed)
    t0 = time.perf_counter()
    tree, path_indices, total_iters = rrt_plan(start, goal, grid, cfg, rng)
    plan_time = time.perf_counter() - t0

    # Console summary
    success = path_indices is not None
    p_len   = path_length(tree, path_indices) if path_indices else 0.0

    print("\n=== RRT Summary ===")
    print(f"  seed           : {actual_seed}")
    print(f"  maze           : {args.maze}")
    print(f"  start          : {tuple(start)}")
    print(f"  goal           : {tuple(goal)}")
    print(f"  iterations     : {total_iters}")
    print(f"  tree size      : {len(tree)} nodes")
    print(f"  plan time      : {plan_time:.4f} s")
    print(f"  success        : {success}")
    if success:
        wps = [tuple(tree[i].pos.round(2).tolist()) for i in path_indices]
        print(f"  path length    : {p_len:.2f}")
        print(f"  waypoints ({len(wps):>3}): {wps}")
    else:
        print("  No path found within max_iters.")

    # Save plan.npz
    t_pos = np.array([nd.pos for nd in tree])
    t_par = np.array(
        [nd.parent if nd.parent is not None else -1 for nd in tree],
        dtype=np.int32,
    )
    save_kw: dict = dict(tree_positions=t_pos, tree_parents=t_par)
    if path_indices is not None:
        save_kw["path"] = np.array([tree[i].pos for i in path_indices])
    np.savez(plan_path, **save_kw)
    print(f"\n  Saved: {plan_path}")

    # Visualize
    print("Generating visualizations...")
    visualize(
        tree          = tree,
        path_indices  = path_indices,
        grid          = grid,
        start         = start,
        goal          = goal,
        img_path      = img_path,
        gif_path      = gif_path,
        viz_mode      = args.viz,
        animate_every = args.animate_every,
        plan_time     = plan_time,
        total_iters   = total_iters,
        fps           = args.fps,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
