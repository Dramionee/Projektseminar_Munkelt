import heapq
import math
from copy import deepcopy


class Node:
    def __init__(self, job_next_op, machine_ready, job_ready, schedule, makespan, lb):
        self.job_next_op = job_next_op          # next operation index per job [j0_op, j1_op, j2_op]
        self.machine_ready = machine_ready      # time when each machine becomes free
        self.job_ready = job_ready              # time when each job becomes ready for its next op
        self.schedule = schedule                # list of scheduled operations
        self.makespan = makespan                # current makespan
        self.lb = lb                            # lower bound

    def __lt__(self, other):
        return self.lb < other.lb



def compute_lower_bound(jobs, node):
    n_jobs = len(jobs)
    n_machines = len(jobs[0])

    # Remaining processing time per job
    job_remaining = []
    for j in range(n_jobs):
        rem = 0
        for op in range(node.job_next_op[j], n_machines):
            rem += jobs[j][op][1]
        job_remaining.append(node.job_ready[j] + rem)

    # Remaining processing time per machine
    machine_remaining = []
    for m in range(n_machines):
        rem = 0
        for j in range(n_jobs):
            op = node.job_next_op[j]
            if op < n_machines and jobs[j][op][0] == m:
                rem += jobs[j][op][1]
        machine_remaining.append(node.machine_ready[m] + rem)

    return max(max(job_remaining), max(machine_remaining))


# ------------------------------------------------------------
#  Branch & Bound Algorithm
# ------------------------------------------------------------

def branch_and_bound_jssp(jobs):
    n_jobs = len(jobs)
    n_ops = len(jobs[0])
    n_machines = n_ops

    best_makespan = math.inf
    best_schedule = None

    # Initial root node
    root = Node(
        job_next_op=[0]*n_jobs,
        machine_ready=[0]*n_machines,
        job_ready=[0]*n_jobs,
        schedule=[],
        makespan=0,
        lb=0
    )
    root.lb = compute_lower_bound(jobs, root)

    pq = []
    heapq.heappush(pq, root)

    while pq:
        node = heapq.heappop(pq)

        # prune
        if node.lb >= best_makespan:
            continue

        # Check if all operations done
        if all(node.job_next_op[j] == n_ops for j in range(n_jobs)):
            if node.makespan < best_makespan:
                best_makespan = node.makespan
                best_schedule = node.schedule
            continue

        # Branching: choose jobs with available next operation
        for j in range(n_jobs):
            op = node.job_next_op[j]
            if op >= n_ops:
                continue

            machine, dur = jobs[j][op]

            # Earliest possible start time
            start = max(node.machine_ready[machine], node.job_ready[j])
            end = start + dur

            # Create child node
            child = Node(
                job_next_op=deepcopy(node.job_next_op),
                machine_ready=deepcopy(node.machine_ready),
                job_ready=deepcopy(node.job_ready),
                schedule=node.schedule + [(j, op, machine, start, end)],
                makespan=max(node.makespan, end),
                lb=0
            )

            # Update state
            child.job_next_op[j] += 1
            child.machine_ready[machine] = end
            child.job_ready[j] = end

            # Compute LB
            child.lb = compute_lower_bound(jobs, child)

            # Only push if promising
            if child.lb < best_makespan:
                heapq.heappush(pq, child)

    return best_makespan, best_schedule


# ------------------------------------------------------------
# Example 3×3 JSSP instance
# (taken from standard JSSP toy examples)
# Format: (machine, duration)
# ------------------------------------------------------------

jobs = [
[(2,1), (0,3), (1,6), (3,7), (2,3), (4,6)],
[(1,8), (2,5), (4,10), (5,10), (0,10), (3,4)],
[(2,5), (3,4), (5,8), (0,9), (1,1), (4,7)],
[(1,5), (0,5), (2,5), (3,3), (4,8), (5,9)],
[(0,9), (3,3), (4,5), (5,4), (2,3), (1,1)],
[(2,3), (3,3), (4,9), (1,10), (0,4), (5,1)]
]

best_makespan, best_schedule = branch_and_bound_jssp(jobs)

print("Optimal Makespan:", best_makespan)
print("Schedule:")
for op in best_schedule:
    print(op)
