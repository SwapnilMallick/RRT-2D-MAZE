# RRT Maze Planner

2D path planning using **Rapidly-exploring Random Trees (RRT)** and **Go-Explore**-inspired exploration, with multiple algorithm variants, maze layouts, and matplotlib visualization (static PNG + animated GIF).

## Algorithms

| Directory | Algorithm | State space | Maze |
|-----------|-----------|-------------|------|
| `rrt/` | Standard RRT | Continuous (occupancy grid) | Discrete 100×100 grid |
| `bidirectional_rrt/` | Bidirectional RRT | Continuous (occupancy grid) | Discrete 100×100 grid |
| `go_explore_inspired_rrt/` | Go-Explore RRT (no teleport) | Continuous | Unit square, four-room |
| `explore_discrete/MazeGridV1/` | Go-Explore on discrete grid, 4-room | Discrete cells | 21×21 grid |
| `explore_discrete/MazeGridV2/` | Go-Explore on discrete grid, 8-room | Discrete cells | 29×29 grid |
| `continuous_bin/` | Bin-based Go-Explore (anti-clutter) | Continuous | Unit square, five maze layouts |
| `explore/v1_count_nodes/` | Count-based node scorer (base for variants) | Continuous | Unit square, five maze layouts |
| `sampling_comparison/` | Uniform vs. soft-prior reset-replay RRT | Continuous | Unit square, four-room |
| `go_explore_rrt_networkx/` | NetworkX-backed Go-Explore RRT | Continuous | Unit square | ⚠ WIP |

## Requirements

```bash
pip install numpy matplotlib pillow        # all planners
pip install networkx                       # explore_discrete/, go_explore_rrt_networkx/
```

## Quick start

```bash
# Standard RRT — bugtrap preset, static + animated output
cd rrt && python rrt_maze_2d.py

# Bidirectional RRT — rooms preset
cd bidirectional_rrt && python rrt_bidirectional_maze_2d.py --maze preset:rooms --viz both

# Go-Explore RRT — continuous four-room maze (no teleport)
cd go_explore_inspired_rrt && python goexplore_rrt.py

# Discrete grid explorer — 21×21 four-room maze
cd explore_discrete/MazeGridV1 && python grid_walker.py

# Discrete grid explorer — 29×29 eight-room maze
cd explore_discrete/MazeGridV2 && python grid_walker_8_room.py

# Continuous bin-based Go-Explore (default: maze V1)
cd continuous_bin && python bin_goexplore.py

# Multi-seed bin-resolution sweep (prints stats, no images)
cd continuous_bin && python bin_sweep.py

# Count-based node scorer
cd explore/v1_count_nodes && python explore_count_nodes.py

# Sampling comparison: uniform vs. soft-prior reset-replay RRT
cd sampling_comparison && python uniform_reset_replay_rrt.py
cd sampling_comparison && python soft_prior_reset_replay_rrt.py
```

## Maze layouts

### Discrete presets (RRT and B-RRT only)

| Name | Description | Default start | Default goal |
|------|-------------|---------------|--------------|
| `empty` | No obstacles; trivial straight-line path | (5, 5) | (95, 95) |
| `wall` | Single vertical wall with a gap | (10, 50) | (90, 50) |
| `bugtrap` | U-shaped trap; start inside, goal outside | (35, 50) | (10, 50) |
| `rooms` | 4 rooms connected by narrow doorways | (12, 12) | (88, 88) |

Use `--maze preset:NAME` or `--maze file:PATH.npy` (custom boolean numpy array, `True` = obstacle).

### Continuous maze layouts (Go-Explore variants)

All layouts are a unit square [0,1]×[0,1] with axis-aligned walls; collision is exact via Liang-Barsky segment-AABB clipping. Doorway positions are encoded as fractions of the wall span.

| Version | Structure |
|---------|-----------|
| V1 | Classic four-room cross; doorways at 18–30% and 70–82% of each wall |
| V2 | V1 plus inner vertical walls at x=0.25 and x=0.75, with a central doorway |
| V3 | V1 plus inner vertical walls with three doorways each (tighter passages) |
| V4 | V3 but the inner walls are thick (0.20 units) instead of thin |
| V5 | Dense nested structure: outer vertical guards at x=0.18/0.82, inner ring of horizontal and vertical walls; no central cross wall |

Select with `--mazeVersion 1`–`5` in `bin_goexplore.py` and `explore_count_nodes.py`.

## Usage

### Standard RRT

```bash
cd rrt
python rrt_maze_2d.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--maze` | `preset:bugtrap` | `preset:NAME` or `file:PATH.npy` |
| `--width` | 100 | Grid width in cells |
| `--height` | 100 | Grid height in cells |
| `--start` | preset-dependent | Start position `x,y` |
| `--goal` | preset-dependent | Goal position `x,y` |
| `--step_size` | 5.0 | RRT step size ε (grid units) |
| `--goal_threshold` | 5.0 | Distance to goal that triggers direct connection |
| `--goal_bias` | 0.05 | Probability of sampling the goal each iteration |
| `--max_iters` | 5000 | Maximum RRT iterations |
| `--collision_step` | 0.5 | Edge collision-check resolution (grid units) |
| `--seed` | None | Random seed for reproducibility |
| `--viz` | `both` | `static`, `animate`, or `both` |
| `--animate_every` | 1 | Record one GIF frame every N nodes added |
| `--fps` | 5 | GIF playback speed |
| `--out` | `maze_rrt` | Output filename stem |

### Bidirectional RRT

Same options as Standard RRT, plus:

| Option | Default | Description |
|--------|---------|-------------|
| `--connect_tol` | `step_size` | Max distance between trees to declare connection |

`--goal_bias` and `--goal_threshold` are unused — the goal tree replaces them.

### Go-Explore RRT

```bash
cd go_explore_inspired_rrt
python goexplore_rrt.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--start` | `0.10,0.10` | Start `x,y` in [0,1]×[0,1] |
| `--goal` | `0.90,0.90` | Goal `x,y` in [0,1]×[0,1] |
| `--step_size` | 0.03 | RRT step size ε |
| `--goal_bias` | 0.10 | Probability of sampling the goal each iteration |
| `--goal_radius` | 0.03 | Distance to goal triggering success |
| `--noise_std` | 0.0 | Std-dev of angular noise on steering (0 = exact) |
| `--max_iter` | 8000 | Maximum iterations (singular, not `--max_iters`) |
| `--wall_thickness` | 0.02 | Maze wall thickness in maze units |
| `--seed` | None | Random seed |
| `--viz` | `both` | `static`, `animate`, `both`, or `none` |
| `--animate_every` | 1 | Record one GIF frame every N iterations |
| `--fps` | 1 | GIF playback speed |
| `--out` | `maze_tree` | Output filename stem |

### Bin-based Go-Explore

```bash
cd continuous_bin
python bin_goexplore.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--mazeVersion` | 1 | Maze layout 1–5 |
| `--start` | `0.10 0.10` | Two floats `x y` |
| `--goal` | `0.90 0.90` | Two floats `x y` |
| `--wall_thickness` | 0.02 | Wall thickness in maze units |

Outputs a static PNG (two panels: tree colored by bin + per-bin population heatmap) to `FourRoomMaze{Vn}/` relative to the script.

### Discrete grid explorer

```bash
cd explore_discrete/MazeGridV1 && python grid_walker.py   # 21x21, 4-room
cd explore_discrete/MazeGridV2 && python grid_walker_8_room.py   # 29x29, 8-room
```

No flags; edit `GridWalker.__init__` / `GridWalkerMod.__init__` to change `k_seed`, `j_roll`, `n_iters`, or `seed`. Outputs a single PNG shaded by visit count to `images/` in each script's directory.

### Sampling comparison

```bash
cd sampling_comparison
python uniform_reset_replay_rrt.py       # uniform baseline, 50 seeds
python soft_prior_reset_replay_rrt.py    # room-prior variant, same 50 seeds
```

No flags; both scripts print multi-seed summary statistics (success rate, iterations to goal, env steps) and produce no image files.

## Examples

```bash
# Bugtrap with animation
cd rrt && python rrt_maze_2d.py --maze preset:bugtrap --viz both

# Rooms with more iterations
cd rrt && python rrt_maze_2d.py --maze preset:rooms --step_size 3 --max_iters 8000

# Reproducible bidirectional run
cd bidirectional_rrt && python rrt_bidirectional_maze_2d.py --maze preset:wall --seed 42

# Go-Explore, noisy steering, every-10th-frame GIF
cd go_explore_inspired_rrt && python goexplore_rrt.py --seed 42 --noise_std 0.3 --animate_every 10

# Bin-based Go-Explore on the dense V5 maze
cd continuous_bin && python bin_goexplore.py --mazeVersion 5 --start 0.05 0.05 --goal 0.95 0.95
```

## Output files

All outputs are timestamped and written relative to the script's directory.

```
rrt/
  plans/   maze_rrt_<ts>.npz       — tree + path arrays (numpy)
  images/  maze_rrt_<ts>.png
  gifs/    maze_rrt_<ts>.gif

bidirectional_rrt/
  plans/   maze_bidirectional_rrt_<ts>.npz
  images/  maze_bidirectional_rrt_<ts>.png
  gifs/    maze_bidirectional_rrt_<ts>.gif

go_explore_inspired_rrt/
  images/  maze_tree_<ts>.png
  gifs/    maze_tree_<ts>.gif

continuous_bin/
  FourRoomMaze{V1-V5}/  bin_goexplore_<ts>.png

explore_discrete/MazeGridV1/images/  grid_walker_<ts>.png
explore_discrete/MazeGridV2/images/  grid_walker_mod_<ts>.png
explore/v1_count_nodes/images/        explore_count_nodes_<ts>.png
```

The `.npz` files (RRT and B-RRT only) contain:
- **RRT**: `tree_positions`, `tree_parents`, `path` (if found)
- **B-RRT**: `tree_a_positions`, `tree_a_parents`, `tree_b_positions`, `tree_b_parents`, `path`, `meeting_point`, `success`

Go-Explore variants print replay statistics to stdout but write no `.npz`.

## Visualization

- **Static PNG**: full tree, solution path in red, start (green dot) and goal (yellow star).
- **Animated GIF**: incremental tree growth with solution path on the final frame.
- **B-RRT colors**: T_a (start tree) blue, T_b (goal tree) orange, meeting point magenta.
- **Go-Explore GIF**: each frame shows the agent's physical traversal — orange = replay path root→nearest, green = accepted extension, red = blocked extension. Makes the O(depth) replay cost visible.
- **Bin heatmap** (`bin_goexplore.py`): right panel shows per-bin node count colored by `magma` colormap; left panel shows the tree with the bin grid overlaid.
- **Discrete grid** (`grid_walker.py`): tiles shaded by visit count (white = unvisited, gray = wall, red = solution path).

## Architecture notes

### The no-teleport constraint

The Go-Explore variants enforce a key constraint absent from standard RRT: **the agent cannot teleport to an arbitrary tree node**. To extend from `nearest`, it must reset to the root and replay the stored action chain (root → nearest) before taking one new step. Replay cost is O(depth) per iteration.

The printed metric `env_steps / node` quantifies this overhead; standard teleport-RRT scores ~1.0. Deeper trees and harder mazes push this ratio higher.

### Hierarchical node selection (`continuous_bin/`)

`BinGoExplore` avoids the spatial clutter problem (nodes piling up in one region) by a two-level selection:
1. Pick the non-empty bin with minimum node population (spatial saturation signal).
2. Within that bin, pick the least-visited node (intra-bin tie-break).

The bin grid is `nb × nb` where `nb = 2 × bins_per_room_axis` (default 3 → 6×6 = 36 bins, ~9 per room).

### Discrete grid exploration (`explore_discrete/`)

`GridWalker` selects the globally least-visited cell, resets to root, replays the first-arrival-frozen canonical path to that cell, then rolls out up to `j` steps. Move rule: avoid parent, prefer unvisited neighbors, else least-visited. Wall bumps break the walk without incrementing the wall's count. The exploration graph is a `nx.DiGraph`; only the first arrival at each cell freezes its canonical replay path.
