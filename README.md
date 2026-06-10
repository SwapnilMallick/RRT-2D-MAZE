# RRT Maze Planner

2D path planning using **Rapidly-exploring Random Trees (RRT)**, with three variants and matplotlib visualization (static PNG + animated GIF).

## Algorithms

| Script | Algorithm | Trees | Goal bias | Maze |
|--------|-----------|-------|-----------|------|
| `rrt/rrt_maze_2d.py` | Standard RRT | 1 (start-rooted) | 5 % toward goal | Occupancy grid, 100×100 cells |
| `bidirectional_rrt/rrt_bidirectional_maze_2d.py` | B-RRT | 2 (start + goal) | None — goal tree provides implicit pull | Occupancy grid, 100×100 cells |
| `go_explore_inspired_rrt/goexplore_rrt.py` | Go-Explore RRT (phase 1) | 1 (start-rooted) | 10 % toward goal | Continuous four-room maze, unit square |

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

# Go-Explore RRT — continuous four-room maze
cd go_explore_inspired_rrt
python goexplore_rrt.py
```

## Maze presets (RRT and B-RRT only)

| Name | Description | Default start | Default goal |
|------|-------------|---------------|--------------|
| `empty` | No obstacles; trivial straight-line path | (5, 5) | (95, 95) |
| `wall` | Single vertical wall with a gap | (10, 50) | (90, 50) |
| `bugtrap` | U-shaped trap; start inside, goal outside (classic stress test) | (35, 50) | (10, 50) |
| `rooms` | 4 rooms connected by narrow doorways | (12, 12) | (88, 88) |

Use `--maze preset:NAME` to select a preset, or `--maze file:PATH.npy` to load a custom boolean numpy array (`True` = obstacle).

The Go-Explore RRT uses a fixed continuous four-room maze (unit square, coordinates in [0, 1]×[0, 1]) — there is no `--maze` flag.

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

### Go-Explore RRT

```bash
python goexplore_rrt.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--start` | `0.10,0.10` | Start position as `x,y` in [0, 1]×[0, 1] |
| `--goal` | `0.90,0.90` | Goal position as `x,y` in [0, 1]×[0, 1] |
| `--step_size` | 0.03 | RRT step size ε (maze units) |
| `--goal_bias` | 0.10 | Probability of sampling the goal each iteration |
| `--goal_radius` | 0.03 | Distance to goal that triggers success |
| `--noise_std` | 0.0 | Std-dev of angular noise on steering direction (0 = exact, >0 = random action toward sample) |
| `--max_iter` | 8000 | Maximum planner iterations |
| `--wall_thickness` | 0.02 | Maze wall thickness in maze units |
| `--seed` | None | Random seed for reproducibility |
| `--viz` | `both` | Output mode: `static`, `animate`, `both`, or `none` |
| `--animate_every` | 1 | Record one GIF frame every N iterations |
| `--fps` | 1 | GIF playback speed |
| `--out` | `maze_tree` | Output filename stem |

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

# Go-Explore RRT, default four-room maze
python goexplore_rrt.py

# Go-Explore with fixed seed and noisy steering
python goexplore_rrt.py --seed 42 --noise_std 0.3 --viz animate --animate_every 10

# Go-Explore, static output only
python goexplore_rrt.py --start "0.10,0.10" --goal "0.90,0.90" --viz static
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

go_explore_inspired_rrt/
  images/  maze_tree_<ts>.png  — static visualization
  gifs/    maze_tree_<ts>.gif  — per-iteration traversal animation
```

The `.npz` files (RRT and B-RRT only) contain:
- **RRT**: `tree_positions`, `tree_parents`, `path` (if found)
- **B-RRT**: `tree_a_positions`, `tree_a_parents`, `tree_b_positions`, `tree_b_parents`, `path`, `meeting_point`, `success`

The Go-Explore RRT does not write `.npz` files; replay statistics are printed to stdout.

## Visualization

- **Static PNG**: full tree, solution path in red, start (green) and goal (yellow star).
- **Animated GIF**: incremental tree growth; solution path overlaid on the final frame.
- **B-RRT colors**: T_a (start tree) in blue, T_b (goal tree) in orange, meeting point in magenta.
- **Go-Explore GIF**: each frame shows the agent's physical traversal for that iteration — replay path root→nearest in orange, extension step in green (accepted) or red (blocked). This makes the O(depth) replay cost visible per iteration.

## Go-Explore vs standard RRT

The key constraint in the Go-Explore variant is that the agent **cannot teleport** to an arbitrary tree node. The only primitives are reset-to-root and replay-action-sequence. To extend from the nearest node, the agent resets and re-executes the full `root → nearest` action chain before taking one new step.

The printed metric `env_steps / node` measures this overhead; standard teleport-RRT scores ~1.0 since it never replays. Deeper trees and harder mazes push this ratio higher.

## Repository structure

```
RRT_MAZE/
├── rrt/
│   ├── rrt_maze_2d.py
│   ├── plans/
│   ├── images/
│   └── gifs/
├── bidirectional_rrt/
│   ├── rrt_bidirectional_maze_2d.py
│   ├── plans/
│   ├── images/
│   └── gifs/
└── go_explore_inspired_rrt/
    ├── goexplore_rrt.py
    ├── images/
    └── gifs/
```
