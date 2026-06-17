"""
Runner for the networkx Go-Explore RRT.

Runs the planner and saves the result to plans/ as a pickle so that
nx_viz.py (static image) and animate.py (step-by-step animation) can load
the same tree without re-running the planner.
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import networkx as nx
from goexplore_nx import GoExploreRRTGraph
from goexplore_maze import FourRoomMaze
import argparse

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="random seed (omit for a random run)")
    args = ap.parse_args()

    maze = FourRoomMaze(wall_thickness=0.02)
    planner = GoExploreRRTGraph(maze, start=(0.10, 0.10), goal=(0.90, 0.90), seed=args.seed)
    actual_seed = planner.rng.bit_generator.state["state"]["state"]
    res = planner.plan()

    print("--- networkx Go-Explore RRT ---")
    print(f"success         : {res['success']}")
    print(f"nodes / edges   : {res['nodes']} / {res['edges']}  (tree: edges = nodes-1)")
    print(f"solution length : {len(res['path_nodes'])} nodes")
    print(f"env steps total : {res['env_steps']}  (replay {res['replay_steps']}, "
          f"extension {res['extension_steps']}, resets {res['resets']})")
    print(f"env_steps / node: {res['env_steps'] / res['nodes']:.1f}")
    print(f"seed            : {actual_seed}")
    print(f"frames collected: {len(res['frames'])}")

    print(f"is tree (DAG, 1 parent each): "
          f"{nx.is_tree(planner.G) and nx.is_directed_acyclic_graph(planner.G)}")

    g = res["goal_node"]
    a1 = planner.actions_root_to(g)
    a2 = planner.actions_root_to_cheapest(g)
    same = len(a1) == len(a2) and all(np.allclose(x, y) for x, y in zip(a1, a2))
    print(f"predecessor-walk == shortest-path replay: {same}  ({len(a1)} actions)")

    plans_dir = Path(__file__).parent / "plans"
    plans_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = plans_dir / f"goexplore_rrt_{ts}.pkl"

    with open(save_path, "wb") as fh:
        pickle.dump({
            "G":           planner.G,
            "start":       planner.start,
            "goal":        planner.goal,
            "root":        planner.root,
            "result":      res,
            "maze_walls":  maze.walls,
            "maze_bounds": maze.bounds,
            "seed":        actual_seed,
        }, fh)
    print(f"\nPlan saved: {save_path}")
    print("Run nx_viz.py for a static image, animate.py for the full step animation.")
