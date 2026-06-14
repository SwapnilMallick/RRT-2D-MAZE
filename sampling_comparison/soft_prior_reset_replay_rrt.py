"""
Hand-coded soft-prior reset-and-replay RRT.

Same planner and task as the uniform baseline (start top-left, goal top-right),
but non-goal samples are drawn from a SOFT room prior mixed with uniform:

    sample ~ (1 - beta) * Uniform([0,1]^2)  +  beta * RoomPrior

RoomPrior down-weights the two bottom rooms (off-path for this start/goal) but
keeps their weight > 0, and beta < 1 keeps a uniform floor -- so a wrong prior
costs efficiency, never completeness. These hand-set weights encode the known
route (top rooms), i.e. the best case a perfect VLM could produce here.

Runs the same multi-seed experiment as the baseline and prints statistics.
"""

from uniform_reset_replay_rrt import run_experiment

# room id: 0=BL 1=BR 2=TL 3=TR ; top rooms (on-path) high, bottom rooms low-but-nonzero
TOP_PRIOR = {0: 0.10, 1: 0.10, 2: 1.0, 3: 1.0}


if __name__ == "__main__":
    run_experiment(room_weights=TOP_PRIOR,
                   tag="hand-coded soft-prior reset-replay RRT")
