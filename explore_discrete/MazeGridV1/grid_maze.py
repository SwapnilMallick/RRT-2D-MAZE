"""Discrete grid four-room maze: build the obstacle set, doorways, start/goal cells."""
import numpy as np

def build_grid(n=21):
    """n x n grid. Walls on the central cross with 2 doorways per wall.
    Returns: obstacles (set of (i,j)), and the geometry for plotting.
    Convention: cell (i,j) -> i is column (x), j is row (y). (0,0) bottom-left."""
    obstacles = set()
    mid = n // 2                      # central wall row/col index
    # doorway cell-bands (in grid index), placed off-center like the continuous maze
    lo = (int(0.18*n), int(0.30*n))   # lower/left doorway band
    hi = (int(0.70*n), int(0.82*n))   # upper/right doorway band
    def in_band(k, band): return band[0] <= k < band[1]

    # outer boundary walls
    for i in range(n):
        for j in range(n):
            if i == 0 or j == 0 or i == n-1 or j == n-1:
                obstacles.add((i, j))
    # vertical wall at column = mid (separates left/right); doorways in lower & upper bands (y=j)
    for j in range(n):
        if not (in_band(j, lo) or in_band(j, hi)):
            obstacles.add((mid, j))
    # horizontal wall at row = mid (separates bottom/top); doorways in left & right bands (x=i)
    for i in range(n):
        if not (in_band(i, lo) or in_band(i, hi)):
            obstacles.add((i, mid))
    return obstacles, mid, lo, hi

if __name__ == "__main__":
    import os
    import datetime
    import matplotlib.pyplot as plt
    n = 21
    obstacles, mid, lo, hi = build_grid(n)
    start = (2, 2)            # bottom-left room
    goal  = (n-3, n-3)        # top-right room
    assert start not in obstacles and goal not in obstacles

    grid = np.zeros((n, n))
    for (i, j) in obstacles: grid[j, i] = 1   # note: grid[row=j, col=i]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, origin="lower", cmap="Greys", vmin=0, vmax=1.5)
    # mark doorway cells in the walls (free cells on the wall lines)
    door_cells = []
    for j in range(n):
        if (mid, j) not in obstacles and 0 < j < n-1: door_cells.append((mid, j))
    for i in range(n):
        if (i, mid) not in obstacles and 0 < i < n-1: door_cells.append((i, mid))
    if door_cells:
        dc = np.array(door_cells)
        ax.scatter(dc[:,0], dc[:,1], marker="s", s=60, facecolors="none",
                   edgecolors="#1f6f3f", linewidths=1.5, label="doorway cells")
    ax.scatter(*start, c="#2a9d8f", s=120, label="start")
    ax.scatter(*goal, c="#e9c46a", s=200, marker="*", edgecolor="k", label="goal")
    ax.set_xticks(range(0, n, 2)); ax.set_yticks(range(0, n, 2))
    ax.grid(True, color="#ccc", lw=0.4); ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{n}x{n} discrete four-room maze  "
                 f"({len(obstacles)} obstacle cells, {len(door_cells)} doorway cells)")
    out_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(out_dir, f"grid_maze_{ts}.png")
    fig.tight_layout(); fig.savefig(out_path, dpi=130)
    print(f"grid {n}x{n}: {len(obstacles)} obstacles, doorways at bands lo={lo} hi={hi}")
    print(f"start {start} obstacle? {start in obstacles} | goal {goal} obstacle? {goal in obstacles}")
    # connectivity sanity: BFS start->goal over free cells (8-connected)
    from collections import deque
    free = lambda c: (0<=c[0]<n and 0<=c[1]<n and c not in obstacles)
    seen={start}; dq=deque([start]); reach=False
    while dq:
        i,j=dq.popleft()
        if (i,j)==goal: reach=True; break
        for di in(-1,0,1):
            for dj in(-1,0,1):
                if di==0 and dj==0: continue
                c=(i+di,j+dj)
                if free(c) and c not in seen: seen.add(c); dq.append(c)
    print(f"goal reachable from start (8-connected BFS): {reach}")
