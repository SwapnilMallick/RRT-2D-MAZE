"""
Bin-resolution sweep: vary bins-per-room and measure whether the agent escapes
the start rooms. Reuses BinGoExplore but overrides the bin grid to b_per_room^2
bins per room (so total bins = 4 * b_per_room^2). Multi-seed.
"""
import numpy as np
from goexplore_maze import FourRoomMaze
from bin_goexplore import BinGoExplore, coverage, Node


class BinGoExploreRes(BinGoExplore):
    """Same algorithm, configurable bin resolution (bins per maze axis = nb)."""
    def __init__(self, maze, nb_axis, **kw):
        self._nb_axis = nb_axis
        self.maze = maze
        self.start = np.asarray(kw.get("start", (0.10, 0.10)), float)
        self.goal = np.asarray(kw.get("goal", (0.90, 0.90)), float)
        self.eps = kw.get("eps", 0.03); self.k = kw.get("k_seed", 20)
        self.j = kw.get("j_roll", 12); self.m = kw.get("m_iters", 20000)
        self.goal_radius = kw.get("goal_radius", 0.03)
        self.d_theta = np.deg2rad(kw.get("d_theta_deg", 35))
        self.max_tries = kw.get("max_tries", 16)
        self.node_cap = kw.get("node_cap", 4000)
        self.rng = np.random.default_rng(kw.get("seed", 0))

        nb = nb_axis
        self.nb = nb; self.w = 1.0/nb
        self.bin_boxes = {}
        for bi in range(nb):
            for bj in range(nb):
                b = bj*nb + bi
                self.bin_boxes[b] = (bi*self.w, (bi+1)*self.w, bj*self.w, (bj+1)*self.w)
        self.bin_pop = {b: 0 for b in self.bin_boxes}
        self.bin_nodes = {b: [] for b in self.bin_boxes}

        b0 = self.bin_of(self.start)
        self.nodes = [Node(0, self.start.copy(), None, None, b0)]
        self.bin_pop[b0] = 1; self.bin_nodes[b0].append(0)
        self.taken = {0: []}
        self.env_steps = self.replay_steps = self.ext_attempts = self.resets = 0
        self.reached = None

    def bin_of(self, p):
        bi = min(int(p[0]/self.w), self.nb-1)
        bj = min(int(p[1]/self.w), self.nb-1)
        return bj*self.nb + bi


def room_of(p):
    return (0 if p[0] < 0.5 else 1) + 2*(0 if p[1] < 0.5 else 1)

# bins-per-room r -> bins per axis = 2r  (maze is 2 rooms across)
configs = [(1, 2), (2, 4), (3, 6), (4, 8)]   # (bins/room/axis, total bins/axis)
SEEDS = range(8)

print(f"{'bins/room':>10} {'total bins':>11} {'escape%':>8} {'reachGoal%':>11} "
      f"{'rooms':>7} {'coverage%':>10}")
for r, axis in configs:
    esc=[]; goal=[]; rooms=[]; cov=[]
    for s in SEEDS:
        ex = BinGoExploreRes(FourRoomMaze(0.02), nb_axis=axis, seed=s,
                             start=(0.10,0.10), goal=(0.90,0.90),
                             m_iters=20000, node_cap=4000)
        ex.run()
        rset = {room_of(n.pos) for n in ex.nodes}
        esc.append(len(rset) >= 2)
        goal.append(ex.reached is not None)
        rooms.append(len(rset))
        cov.append(coverage(ex))
    print(f"{r:>10} {axis*axis:>11} {np.mean(esc)*100:>7.0f} "
          f"{np.mean(goal)*100:>10.0f} {np.mean(rooms):>7.1f} {np.mean(cov)*100:>9.1f}")
