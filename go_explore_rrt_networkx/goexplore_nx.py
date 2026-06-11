"""
NetworkX-backed Go-Explore RRT planner (phase 1).

The tree lives in a nx.DiGraph instead of parent pointers; planner logic is
identical to the array version in go_explore_inspired_rrt/goexplore_rrt.py.

Node attribute  'pos'   : configuration q_v
Edge (parent->child)    : 'action' = displacement, 'weight' = step cost

plan() also collects a per-physical-step frame history and the ordered edge
list so that animate.py can render every reset/replay/extension step without
re-running the planner.
"""

import numpy as np
import networkx as nx
from goexplore_maze import FourRoomMaze, step  # noqa: F401  (re-exported for callers)

class GoExploreRRTGraph:
    def __init__(self, maze, start, goal, step_size=0.03, goal_bias=0.10,
                 goal_radius=0.03, noise_std=0.0, max_iter=8000, verify_replay=True,
                 seed=None):
        self.maze = maze
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_radius = goal_radius
        self.seed = seed
        self.noise_std = noise_std
        self.max_iter = max_iter
        self.verify_replay = verify_replay

        self.rng = np.random.default_rng(seed)

        self.G = nx.DiGraph()
        self.root = 0
        self.G.add_node(self.root, pos=self.start.copy())
        self._next_id = 1

        self.env_steps = self.replay_steps = self.extension_steps = self.resets = 0

    # --- spatial query: O(N) brute force (networkx has no spatial index) ---
    def nearest(self, q):
        best, best_d2 = None, np.inf
        for n, p in self.G.nodes(data="pos"):
            d2 = float(np.sum((p - q) ** 2))
            if d2 < best_d2:
                best_d2, best = d2, n
        return best

    def _path_root_to(self, n):
        """Node IDs from root to n (inclusive), via predecessor walk."""
        path = []
        cur = n
        while True:
            path.append(cur)
            preds = list(self.G.predecessors(cur))
            if not preds:
                break
            cur = preds[0]
        path.reverse()
        return path

    # --- replay utilities (kept for external use / goexplore_rrt_nx.py checks) ---
    def actions_root_to(self, n):
        actions = []
        cur = n
        while True:
            preds = list(self.G.predecessors(cur))
            if not preds:
                break
            p = preds[0]
            actions.append(self.G[p][cur]["action"])
            cur = p
        actions.reverse()
        return actions

    def actions_root_to_cheapest(self, n):
        """Shortest-path replay — equals predecessor walk on a pure tree."""
        path = nx.shortest_path(self.G, self.root, n, weight="weight")
        return [self.G[u][v]["action"] for u, v in zip(path, path[1:])]

    def replay_to(self, n):
        self.resets += 1
        pos = self.G.nodes[self.root]["pos"].copy()
        for a in self.actions_root_to(n):
            pos = step(pos, a)
            self.env_steps += 1
            self.replay_steps += 1
        if self.verify_replay:
            assert np.allclose(pos, self.G.nodes[n]["pos"], atol=1e-9)
        return pos

    def steer(self, q_from, q_s):
        d = q_s - q_from
        nrm = np.linalg.norm(d)
        if nrm < 1e-12:
            theta = self.rng.uniform(0, 2 * np.pi)
            u = np.array([np.cos(theta), np.sin(theta)])
        else:
            u = d / nrm
            if self.noise_std > 0:
                ang = np.arctan2(u[1], u[0]) + self.rng.normal(0, self.noise_std)
                u = np.array([np.cos(ang), np.sin(ang)])
        return u * self.step_size

    def sample(self):
        if self.rng.random() < self.goal_bias:
            return self.goal.copy()
        x0, y0, x1, y1 = self.maze.bounds
        return np.array([self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)])

    def plan(self):
        current = self.root
        frames = []       # one entry per physical step, for animate.py
        edges_order = []  # edges in the order they were added, for animate.py

        for it in range(self.max_iter):
            q_s = self.sample()
            v_star = self.nearest(q_s)

            if v_star != current:
                path = self._path_root_to(v_star)
                hi = np.array([self.G.nodes[n]["pos"] for n in path])
                q_root  = self.G.nodes[self.root]["pos"]
                q_vstar = self.G.nodes[v_star]["pos"]

                # reset frame: agent jumps back to root
                frames.append(dict(
                    agent=q_root.copy(), phase="reset",
                    n_edges=len(edges_order),
                    sample=q_s.copy(), vstar=q_vstar.copy(), hi=hi,
                ))
                self.resets += 1

                # replay frames: one per edge along root -> v_star
                walk = q_root.copy()
                for u, v in zip(path[:-1], path[1:]):
                    a = self.G[u][v]["action"]
                    walk = step(walk, a)
                    self.env_steps += 1
                    self.replay_steps += 1
                    frames.append(dict(
                        agent=walk.copy(), phase="replay",
                        n_edges=len(edges_order),
                        sample=q_s.copy(), vstar=q_vstar.copy(), hi=hi,
                    ))
                current = v_star

            q_near = self.G.nodes[v_star]["pos"]
            a = self.steer(q_near, q_s)
            q_new = step(q_near, a)
            self.env_steps += 1
            self.extension_steps += 1

            if self.maze.segment_free(q_near, q_new):
                w = self._next_id
                self._next_id += 1
                self.G.add_node(w, pos=q_new)
                self.G.add_edge(v_star, w, action=a, weight=self.step_size)
                edges_order.append(np.array([q_near, q_new]))
                current = w
                frames.append(dict(
                    agent=q_new.copy(), phase="extend",
                    n_edges=len(edges_order),
                    sample=q_s.copy(), vstar=q_near.copy(), hi=None,
                ))
                if np.linalg.norm(q_new - self.goal) <= self.goal_radius:
                    return self._result(True, w, it + 1, frames, edges_order)
            else:
                current = v_star
                frames.append(dict(
                    agent=q_near.copy(), phase="extend_fail",
                    n_edges=len(edges_order),
                    sample=q_s.copy(), vstar=q_near.copy(), hi=None,
                ))

        return self._result(False, None, self.max_iter, frames, edges_order)

    def _result(self, success, goal_node, iters, frames, edges_order):
        path_nodes, path_actions = [], []
        if goal_node is not None:
            path_nodes = nx.shortest_path(self.G, self.root, goal_node)
            path_actions = [self.G[u][v]["action"]
                            for u, v in zip(path_nodes, path_nodes[1:])]
        return {
            "success": success, "iterations": iters,
            "nodes": self.G.number_of_nodes(), "edges": self.G.number_of_edges(),
            "path_nodes": path_nodes, "path_actions": path_actions,
            "env_steps": self.env_steps, "replay_steps": self.replay_steps,
            "extension_steps": self.extension_steps, "resets": self.resets,
            "goal_node": goal_node,
            "frames": frames,
            "edges_order": edges_order,
        }
