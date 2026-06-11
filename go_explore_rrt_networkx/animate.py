"""
Full reset-and-replay animation. Loads a saved plan from plans/ and emits one
frame for EVERY physical step (each reset, each replay step, each extension).
Uses persistent artists (LineCollection + updated scatters) so ~3.6k frames
render in reasonable time.

Usage:
    python animate.py               # uses the latest plan in plans/
    python animate.py plans/foo.pkl # uses a specific plan
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import imageio.v2 as imageio_v2


def latest_plan(plans_dir: Path) -> Path:
    pkls = sorted(plans_dir.glob("goexplore_rrt_*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            f"No saved plans in {plans_dir}. Run goexplore_rrt_nx.py first.")
    return pkls[-1]


plans_dir = Path(__file__).parent / "plans"
plan_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_plan(plans_dir)
print(f"Loading: {plan_path}")

with open(plan_path, "rb") as fh:
    data = pickle.load(fh)

START       = data["start"]
GOAL        = data["goal"]
maze_walls  = data["maze_walls"]
res         = data["result"]
frames      = res["frames"]
edges_order = res["edges_order"]

print(f"solved              : {res['success']} at iteration {res['iterations']}")
print(f"tree nodes          : {res['nodes']}")
print(f"frames (all steps)  : {len(frames)}")
print(f"replay vs extend    : {res['replay_steps'] + res['resets']} / {res['extension_steps']}")

# ----- render with persistent artists -------------------------------------- #
PHASE = {
    "reset":       ("#e76f51", "RESET  -> jump to root"),
    "replay":      ("#e76f51", "REPLAY -> retrace stored actions"),
    "extend":      ("#2a9d8f", "EXTEND -> new node added"),
    "extend_fail": ("#d62828", "EXTEND -> blocked (collision)"),
}

fig, ax = plt.subplots(figsize=(5, 5))
for xmin, ymin, xmax, ymax in maze_walls:
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           facecolor="#555", edgecolor="none", zorder=1))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])

tree_lc  = LineCollection([], colors="#9ec5fe", linewidths=0.6, zorder=2)
ax.add_collection(tree_lc)
aim_line, = ax.plot([], [], color="#bbb", lw=0.8, ls=":", zorder=3)
hi_line,  = ax.plot([], [], color="#e76f51", lw=2.0, alpha=0.85, zorder=4)
sample_sc = ax.scatter([], [], facecolors="none", edgecolors="#888", s=55, zorder=3)
ax.scatter(*START, c="#264653", s=70, zorder=5)
ax.scatter(*GOAL,  c="#e9c46a", s=160, marker="*", edgecolor="k", zorder=5)
agent_sc  = ax.scatter([], [], s=120, zorder=6, edgecolor="k", linewidths=0.5)
counter   = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=8,
                    color="#444", va="bottom")

# running replay-step tally per frame, for the on-screen counter
cum_so_far = []
_r = 0
for f in frames:
    if f["phase"] in ("reset", "replay"):
        _r += 1
    cum_so_far.append(_r)


def draw(k):
    f = frames[k]
    tree_lc.set_segments(edges_order[:f["n_edges"]])
    if f["sample"] is not None:
        sample_sc.set_offsets([f["sample"]])
        aim_line.set_data([f["vstar"][0], f["sample"][0]],
                          [f["vstar"][1], f["sample"][1]])
    else:
        sample_sc.set_offsets(np.empty((0, 2)))
        aim_line.set_data([], [])
    if f["hi"] is not None:
        hi_line.set_data(f["hi"][:, 0], f["hi"][:, 1])
    else:
        hi_line.set_data([], [])
    color, label = PHASE[f["phase"]]
    agent_sc.set_offsets([f["agent"]])
    agent_sc.set_color(color)
    ax.set_title(label, color=color, fontsize=11, fontweight="bold")
    counter.set_text(f"nodes {f['n_edges'] + 1}   replay steps {cum_so_far[k]}")


# stream frames directly into GIF (constant memory, no temp dir or ffmpeg needed)
fps = 50
gifs_dir = Path(__file__).parent / "gifs"
gifs_dir.mkdir(exist_ok=True)
out = gifs_dir / f"{plan_path.stem}.gif"

with imageio_v2.get_writer(out, mode="I", fps=fps, loop=0) as writer:
    for k in range(len(frames)):
        draw(k)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(buf)
        if k % 500 == 0:
            print(f"  rendered {k}/{len(frames)}")

print(f"saved {out}")
