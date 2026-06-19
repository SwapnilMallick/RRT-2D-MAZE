"""
Discrete (occupancy-model) version of the MODIFIED four-room layout:
a 4-column x 2-row arrangement of 8 rooms. Vertical walls in the top and bottom
halves are offset and their doorways staggered, matching the uploaded continuous
layout. Walls are CELLS (one cell thick); doorways are free cells on a wall line.
"""
import numpy as np

def build_grid_8_room(n=29):
    """n x n grid. Returns obstacles (set of (i,j)), plus wall-line indices and
    doorway info for reference. (i=col/x, j=row/y, origin bottom-left)."""
    obstacles = set()
    # boundary
    for i in range(n):
        for j in range(n):
            if i in (0, n-1) or j in (0, n-1):
                obstacles.add((i, j))

    # one horizontal wall line (2 room-rows)
    hrow = n // 2
    # three vertical wall lines (4 room-cols), evenly placed
    vcols = [n // 4, n // 2, 3 * n // 4]

    def band(c0, c1):              # doorway gap as a half-open cell band
        return set(range(c0, c1))

    # doorway gaps on the horizontal wall: one per column-segment, staggered
    # segments split by the vertical columns: [1,v0],[v0,v1],[v1,v2],[v2,n-1]
    seg_x = [(1, vcols[0]), (vcols[0]+1, vcols[1]), (vcols[1]+1, vcols[2]), (vcols[2]+1, n-1)]
    h_doors = set()
    for k, (a, b) in enumerate(seg_x):
        c = a + int((b - a) * [0.55, 0.30, 0.65, 0.40][k])   # staggered door centers
        h_doors |= band(c, c + 3)                            # 3-cell doorway
    for i in range(n):
        if i not in h_doors:
            obstacles.add((i, hrow))

    # vertical walls: TOP half (rows hrow..n-1) and BOTTOM half (rows 0..hrow),
    # offset by using different door positions per half so they look staggered.
    def add_vwall(col, j0, j1, door_frac):
        cd = j0 + int((j1 - j0) * door_frac)
        doors = band(cd, cd + 3)
        for j in range(j0, j1):
            if j not in doors and (col, j) not in obstacles:
                obstacles.add((col, j))

    # top-half vertical walls (above the horizontal wall)
    add_vwall(vcols[0], hrow+1, n-1, 0.30)
    add_vwall(vcols[1], hrow+1, n-1, 0.65)
    add_vwall(vcols[2], hrow+1, n-1, 0.45)
    # bottom-half vertical walls (below) -- offset door fractions => staggered
    add_vwall(vcols[0], 1, hrow, 0.55)
    add_vwall(vcols[1], 1, hrow, 0.30)
    add_vwall(vcols[2], 1, hrow, 0.60)

    return obstacles, hrow, vcols


if __name__ == "__main__":
    import os
    import datetime
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from collections import deque
    n = 29
    obstacles, hrow, vcols = build_grid_mod(n)
    start, goal = (2, n-3), (n-3, n-3)          # top-left room, top-right room
    assert start not in obstacles and goal not in obstacles

    def cell_type(i, j):
        if (i, j) == start: return "start"
        if (i, j) == goal:  return "goal"
        if (i, j) in obstacles: return "wall"
        if i in vcols or j == hrow: return "door"
        return "room"
    COL = {"wall":"#6b6b6b","room":"#ffffff","door":"#bfe3c6",
           "start":"#2a9d8f","goal":"#e9c46a"}

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((i-0.5, j-0.5), 1, 1,
                                   facecolor=COL[cell_type(i, j)],
                                   edgecolor="#cfcfcf", lw=0.5))
    ax.scatter(*start, c="#2a9d8f", s=130, zorder=5)
    ax.scatter(*goal, c="#e9c46a", s=240, marker="*", edgecolor="k", zorder=5)
    ax.set_xlim(-0.5, n-0.5); ax.set_ylim(-0.5, n-0.5); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    handles = [Rectangle((0,0),1,1, facecolor=COL[k], edgecolor="#cfcfcf")
               for k in ["room","wall","door","start","goal"]]
    ax.legend(handles, ["room (free)","wall (obstacle)","doorway (free)","start","goal"],
              loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_title(f"Modified four-room maze (discrete, {n}x{n}) — 4 cols x 2 rows, 8 rooms",
                 fontsize=12)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(out_dir, f"maze_grid_mod_{ts}.png")
    fig.tight_layout(); fig.savefig(out_path, dpi=120)

    # solvability check (8-connected BFS)
    free = lambda c: 0 <= c[0] < n and 0 <= c[1] < n and c not in obstacles
    seen = {start}; dq = deque([start]); reach = False
    while dq:
        i, j = dq.popleft()
        if (i, j) == goal: reach = True; break
        for di in (-1,0,1):
            for dj in (-1,0,1):
                if di or dj:
                    c = (i+di, j+dj)
                    if free(c) and c not in seen: seen.add(c); dq.append(c)
    print(f"grid {n}x{n} | obstacles {len(obstacles)} | free {n*n-len(obstacles)}")
    print(f"start {start} goal {goal} | goal reachable: {reach}")
    print(f"saved {out_path}")
