"""
Static tree visualization. Loads a saved plan from plans/ and writes a PNG.

Usage:
    python nx_viz.py               # uses the latest plan in plans/
    python nx_viz.py plans/foo.pkl # uses a specific plan
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx


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

G          = data["G"]
start      = data["start"]
goal       = data["goal"]
root       = data["root"]
res        = data["result"]
maze_walls = data["maze_walls"]

# --- (1) draw the tree with networkx, positioned at TRUE coordinates ---
node_pos = {n: tuple(p) for n, p in G.nodes(data="pos")}
path_nodes = res["path_nodes"]
path_edges = set(zip(path_nodes, path_nodes[1:]))

fig, ax = plt.subplots(figsize=(7, 7))
for xmin, ymin, xmax, ymax in maze_walls:
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           facecolor="#555", edgecolor="none", zorder=1))

# node colour = depth from root (cost-to-come / step)
depth = nx.shortest_path_length(G, source=root)
node_color = [depth[n] for n in G.nodes()]

nx.draw_networkx_edges(G, node_pos, ax=ax, edge_color="#9ec5fe", width=0.5, arrows=False)
nx.draw_networkx_edges(G, node_pos, ax=ax, edgelist=list(path_edges),
                       edge_color="#d62828", width=2.2, arrows=False)
nx.draw_networkx_nodes(G, node_pos, ax=ax, node_size=8, node_color=node_color,
                       cmap="viridis")
ax.scatter(*start, c="#2a9d8f", s=90, zorder=5, label="start")
ax.scatter(*goal,  c="#e9c46a", s=180, marker="*", edgecolor="k", zorder=5, label="goal")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_title("tree at true coords (node color = depth from root)")
ax.legend(loc="lower right")

images_dir = Path(__file__).parent / "images"
images_dir.mkdir(exist_ok=True)
out_img = images_dir / f"{plan_path.stem}.png"
fig.tight_layout()
fig.savefig(out_img, dpi=130)
plt.close(fig)
print(f"saved {out_img}")

# --- (2) serialization sanity: round-trip through JSON node-link ---
H = G.copy()
for n, p in H.nodes(data="pos"):
    H.nodes[n]["pos"] = list(map(float, p))
for u, v, a in H.edges(data="action"):
    H[u][v]["action"] = list(map(float, a))
json_data = nx.node_link_data(H, link="edges")
back = nx.node_link_graph(json_data, link="edges")
print("round-trip ok:", back.number_of_nodes() == G.number_of_nodes()
      and back.number_of_edges() == G.number_of_edges())
