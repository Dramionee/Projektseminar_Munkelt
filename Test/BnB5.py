# komische chat methode, kommt auch auf 960 nach ca 2h

import time
import matplotlib.pyplot as plt

# ==========================
# Problemdefinition
# ==========================
jobs = [
    [(0,29),(1,78),(2,9),(3,36),(4,49),(5,11),(6,62),(7,56),(8,44),(9,21)],
    [(0,43),(2,90),(4,75),(9,11),(3,69),(1,28),(6,46),(5,46),(7,72),(8,30)],
    [(1,91),(0,85),(3,39),(2,74),(8,90),(5,10),(7,12),(6,89),(9,45),(4,33)],
    [(1,81),(2,95),(0,71),(4,99),(6,9),(8,52),(7,85),(3,98),(9,22),(5,43)],
    [(2,14),(0,6),(1,22),(5,61),(3,26),(4,69),(8,21),(7,49),(9,72),(6,53)],
    [(2,84),(1,2),(5,52),(3,95),(8,48),(9,72),(0,47),(6,65),(4,6),(7,25)],
    [(1,46),(0,37),(3,61),(2,13),(6,32),(5,21),(9,32),(8,89),(7,30),(4,55)],
    [(2,31),(0,86),(1,46),(5,74),(4,32),(6,88),(8,19),(9,48),(7,36),(3,79)],
    [(0,76),(1,69),(3,76),(5,51),(2,85),(9,11),(6,40),(7,89),(4,26),(8,74)],
    [(1,85),(0,13),(2,61),(6,7),(8,64),(9,76),(5,47),(3,52),(4,90),(7,45)]
]

n_jobs = len(jobs)
n_machines = 1 + max(m for j in jobs for m, _ in j)

# Restzeiten je Job
rest_time = [
    [sum(d for _, d in job[i:]) for i in range(len(job)+1)]
    for job in jobs
]

# ==========================
# Branch & Bound
# ==========================
best_ub = float("inf")
best_plan = []

def branch_and_bound(za, zm, idx, plan, ms):
    global best_ub, best_plan

    # Untere Schranke
    lb_jobs = max(za[j] + rest_time[j][idx[j]] for j in range(n_jobs))
    lb_mach = max(
        zm[m] + sum(
            d for j in range(n_jobs)
            for mm, d in jobs[j][idx[j]:] if mm == m
        )
        for m in range(n_machines)
    )
    lb = max(ms, lb_jobs, lb_mach)

    if lb >= best_ub:
        return

    # Fertige Lösung?
    if all(idx[j] == len(jobs[j]) for j in range(n_jobs)):
        if ms < best_ub:
            best_ub = ms
            best_plan = plan[:]
            print(f"Neue Lösung: {best_ub}")
        return

    # ==========================
    # Konfliktanalyse
    # ==========================
    candidates = []
    earliest = float("inf")

    for j in range(n_jobs):
        if idx[j] < len(jobs[j]):
            m, d = jobs[j][idx[j]]
            s = max(za[j], zm[m])
            if s < earliest:
                earliest = s
                candidates = [(j, m, d, s)]
            elif s == earliest:
                candidates.append((j, m, d, s))

    # Gruppieren nach Maschine
    by_machine = {}
    for c in candidates:
        by_machine.setdefault(c[1], []).append(c)

    # ==========================
    # Kein Konflikt → deterministisch
    # ==========================
    for m, ops in by_machine.items():
        if len(ops) == 1:
            j, m, d, s = ops[0]
            za2, zm2, idx2 = za[:], zm[:], idx[:]
            za2[j] = zm2[m] = s + d
            idx2[j] += 1
            branch_and_bound(
                za2, zm2, idx2,
                plan + [(j, m, s, d)],
                max(ms, s + d)
            )
            return

    # ==========================
    # Konflikt → Branching (binär)
    # ==========================
    m, ops = next(iter(by_machine.items()))
    (j1, _, d1, s1), (j2, _, d2, s2) = ops[:2]

    # Branch 1: j1 vor j2
    for (j, d) in [(j1, d1), (j2, d2)]:
        za2, zm2, idx2 = za[:], zm[:], idx[:]
        s = max(za2[j], zm2[m])
        za2[j] = zm2[m] = s + d
        idx2[j] += 1
        branch_and_bound(
            za2, zm2, idx2,
            plan + [(j, m, s, d)],
            max(ms, s + d)
        )

# ==========================
# Start
# ==========================
start = time.time()
branch_and_bound(
    za=[0]*n_jobs,
    zm=[0]*n_machines,
    idx=[0]*n_jobs,
    plan=[],
    ms=0
)
print(f"\nOPTIMUM: {best_ub}")
print(f"Laufzeit: {time.time()-start:.2f}s")
