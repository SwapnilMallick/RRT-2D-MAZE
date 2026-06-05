# RRT Maze Planner

2D path planning in occupancy-grid mazes using **Rapidly-exploring Random Trees (RRT)** and **Bidirectional RRT (B-RRT)**, with matplotlib visualization (static PNG + animated GIF).

## Algorithms

| Script | Algorithm | Trees | Goal bias |
|--------|-----------|-------|-----------|
| `rrt/rrt_maze_2d.py` | Standard RRT | 1 (start-rooted) | 5 % toward goal |
| `bidirectional_rrt/rrt_bidirectional_maze_2d.py` | B-RRT | 2 (start + goal) | None — goal tree provides implicit pull |

## Requirements

```
numpy
matplotlib
pillow   # for GIF output
```

Install with:

```bash
pip install numpy matplotlib pillow
```

## Quick start

```bash
# Standard RRT — bugtrap preset, static + animated output
cd rrt
python rrt_maze_2d.py

# Bidirectional RRT — rooms preset
cd bidirectional_rrt
python rrt_bidirectional_maze_2d.py --maze preset:rooms --viz both
```

## Maze presets

| Name | Description | Default start | Default goal |
|------|-------------|---------------|--------------|
| `empty` | No obstacles; trivial straight-line path | (5, 5) | (95, 95) |
| `wall` | Single vertical wall with a gap | (10, 50) | (90, 50) |
| `bugtrap` | U-shaped trap; start inside, goal outside (classic stress test) | (35, 50) | (10, 50) |
| `rooms` | 4 rooms connected by narrow doorways | (12, 12) | (88, 88) |

Use `--maze preset:NAME` to select a preset, or `--maze file:PATH.npy` to load a custom boolean numpy array (`True` = obstacle).

## Usage

### Standard RRT

```bash
python rrt_maze_2d.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--maze` | `preset:bugtrap` | Maze spec: `preset:NAME` or `file:PATH.npy` |
| `--width` | 100 | Grid width in cells |
| `--height` | 100 | Grid height in cells |
| `--start` | preset-dependent | Start position as `x,y` |
| `--goal` | preset-dependent | Goal position as `x,y` |
| `--step_size` | 5.0 | RRT step size ε (grid units) |
| `--goal_threshold` | 5.0 | Distance to goal that triggers direct connection |
| `--goal_bias` | 0.05 | Probability of sampling the goal each iteration |
| `--max_iters` | 5000 | Maximum RRT iterations |
| `--collision_step` | 0.5 | Edge collision-check resolution (grid units) |
| `--seed` | None | Random seed for reproducibility |
| `--viz` | `both` | Output mode: `static`, `animate`, or `both` |
| `--animate_every` | 1 | Record one GIF frame every N nodes added |
| `--fps` | 5 | GIF playback speed |
| `--out` | `maze_rrt` | Output filename stem |

### Bidirectional RRT

```bash
python rrt_bidirectional_maze_2d.py [options]
```

Same options as above, plus:

| Option | Default | Description |
|--------|---------|-------------|
| `--connect_tol` | `step_size` | Max distance between trees to declare connection |

`--goal_bias` and `--goal_threshold` are not used — the goal tree replaces them.

## Examples

```bash
# Bugtrap with animation
python rrt_maze_2d.py --maze preset:bugtrap --viz both

# Rooms with more iterations and larger step
python rrt_maze_2d.py --maze preset:rooms --step_size 3 --max_iters 8000

# Custom maze, explicit start/goal
python rrt_maze_2d.py --maze file:custom.npy --start "5,5" --goal "95,95"

# Reproducible run
python rrt_maze_2d.py --maze preset:wall --seed 42 --viz static

# Bidirectional, rooms
python rrt_bidirectional_maze_2d.py --maze preset:rooms --step_size 3 --max_iters 8000
```

## Output files

All outputs are timestamped (`YYYYMMDD_HHMMSS`) and written relative to the script's directory:

```
rrt/
  plans/   maze_rrt_<ts>.npz   — tree and path arrays (numpy)
  images/  maze_rrt_<ts>.png   — static visualization
  gifs/    maze_rrt_<ts>.gif   — animated tree growth

bidirectional_rrt/
  plans/   maze_bidirectional_rrt_<ts>.npz
  images/  maze_bidirectional_rrt_<ts>.png
  gifs/    maze_bidirectional_rrt_<ts>.gif
```

The `.npz` files contain:
- **RRT**: `tree_positions`, `tree_parents`, `path` (if found)
- **B-RRT**: `tree_a_positions`, `tree_a_parents`, `tree_b_positions`, `tree_b_parents`, `path`, `meeting_point`, `success`

## Visualization

- **Static PNG**: full tree in gray, solution path in red, start (green square) and goal (yellow star).
- **Animated GIF**: incremental tree growth; solution path overlaid on the final frame.
- **B-RRT colors**: T_a (start tree) in blue, T_b (goal tree) in orange, meeting point in magenta.

## Repository structure

```
RRT_MAZE/
├── rrt/
│   ├── rrt_maze_2d.py
│   ├── plans/
│   ├── images/
│   └── gifs/
└── bidirectional_rrt/
    ├── rrt_bidirectional_maze_2d.py
    ├── plans/
    ├── images/
    └── gifs/
```
