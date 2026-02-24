import time
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# -------------------------
# Job-Shop-Instanz (10x10)
# -------------------------

jobs = [
[(2,1), (0,3), (1,6), (3,7), (2,3), (4,6)],
[(1,8), (2,5), (4,10), (5,10), (0,10), (3,4)],
[(2,5), (3,4), (5,8), (0,9), (1,1), (4,7)],
[(1,5), (0,5), (2,5), (3,3), (4,8), (5,9)],
[(0,9), (3,3), (4,5), (5,4), (2,3), (1,1)],
[(2,3), (3,3), (4,9), (1,10), (0,4), (5,1)]
]

num_jobs = len(jobs)
num_machines = 1 + max(m for job in jobs for m,_ in job)
machine_colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:olive',
                  'tab:purple','tab:grey','tab:cyan','tab:pink','tab:brown']


# -------------------------
# Basis-Hilfsfunktionen
# -------------------------

def all_operations():
    for j in range(num_jobs):
        for op in range(len(jobs[j])):
            yield (j, op)

def proc_time(op):
    j, k = op
    return jobs[j][k][1]

def machine_of(op):
    j, k = op
    return jobs[j][k][0]


# -------------------------
# Giffler–Thompson-Heuristik
# -------------------------

def giffler_thompson(jobs):
    n = len(jobs)
    k = len(jobs[0])
    job_end = [0]*n
    machine_end = [0]*num_machines
    idx = [0]*n

    schedule = []
    finished_ops = 0

    while finished_ops < n*k:
        candidates = []
        for j in range(n):
            if idx[j] < k:
                m, d = jobs[j][idx[j]]
                start = max(job_end[j], machine_end[m])
                candidates.append((j, idx[j], m, d, start))

        j, op_idx, m, d, start = min(candidates, key=lambda x: x[4])
        schedule.append((j, op_idx, m, start, d))
        job_end[j] = start + d
        machine_end[m] = start + d
        idx[j] += 1
        finished_ops += 1

    makespan = max(job_end)
    return schedule, makespan


# -------------------------
# Disjunktivgraph und kritischer Pfad
# -------------------------
# G(S'): Knoten = Operationen, Kanten:
#  - Job-Reihenfolge
#  - bereits fixierte Maschinenreihenfolgen aus path
# LB = Länge des längsten Pfades von "Start" nach "Ende".[web:9][web:10]

def build_graph_from_partial_schedule(path):
    """
    path: Liste von (j, op_idx, m, start, dur) = bereits festgelegte Operationen.
    Wir verwenden nur die Reihenfolge auf Maschinen (sortiert nach start),
    NICHT die konkreten Zeiten, für den Bound.[web:9]
    """
    succ = defaultdict(list)
    pred = defaultdict(list)

    # Job-Kanten (fix)
    for j in range(num_jobs):
        ops = len(jobs[j])
        for op in range(ops - 1):
            a = (j, op)
            b = (j, op + 1)
            succ[a].append(b)
            pred[b].append(a)

    # Maschinenkanten aus path: pro Maschine nach Startzeit sortieren
    per_machine = defaultdict(list)
    for (j, op_idx, m, s, d) in path:
        per_machine[m].append((s, j, op_idx))
    for m in per_machine:
        per_machine[m].sort(key=lambda x: x[0])  # nach Startzeit
        order = [(j, op) for s, j, op in per_machine[m]]
        for (j1, op1), (j2, op2) in zip(order, order[1:]):
            a = (j1, op1)
            b = (j2, op2)
            succ[a].append(b)
            pred[b].append(a)

    return succ, pred

def longest_path_lower_bound(path):
    """
    Berechnet einen graphbasierten LB:
    - baue G(S') aus path
    - hänge virtuellen Start-Knoten s und End-Knoten t dran
    - longest path von s nach t = LB (kritischer Pfad).[web:10]
    """
    succ, pred = build_graph_from_partial_schedule(path)

    s = "__SOURCE__"
    t = "__SINK__"

    # Quelle verbindet zu allen Operationen ohne Vorgänger
    for op in all_operations():
        if len(pred[op]) == 0:
            succ[s].append(op)
            pred[op].append(s)
    # Senke von allen Operationen ohne Nachfolger
    for op in all_operations():
        if len(succ[op]) == 0:
            succ[op].append(t)
            pred[t].append(op)

    # Topologische Sortierung (Graph muss azyklisch sein, sonst LB = +inf).[web:9]
    indeg = defaultdict(int)
    nodes = set([s, t])
    for u in succ:
        nodes.add(u)
        for v in succ[u]:
            nodes.add(v)
            indeg[v] += 1

    Q = deque([u for u in nodes if indeg[u] == 0])
    topo = []
    while Q:
        u = Q.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                Q.append(v)

    if len(topo) != len(nodes):
        # Zyklus => partieller Plan ist inkonsistent, Bound = +inf
        return float("inf")

    # Längste Pfade von s
    dist = {u: float("-inf") for u in nodes}
    dist[s] = 0
    for u in topo:
        if u == t:
            continue
        du = dist[u]
        if du == float("-inf"):
            continue
        for v in succ[u]:
            if v == t:
                w = 0
            elif isinstance(v, tuple):
                w = proc_time(v)
            else:
                w = 0
            if dist[v] < du + w:
                dist[v] = du + w

    return dist[t]


# -------------------------
# Branch and Bound
# -------------------------

def branch_and_bound(initial_ub=float("inf")):
    # Startlösung über Giffler-Thompson
    initial_plan, initial_ms = giffler_thompson(jobs)
    best = min(initial_ms, initial_ub)
    plan = initial_plan

    print(f">>> Initial solution (Giffler–Thompson): Makespan={best}")

    # Zustand: (g_bound, job_end[], machine_end[], idx[], path, ms)
    # path: (j, op_idx, m, start, dur)
    job_end0 = [0]*num_jobs
    machine_end0 = [0]*num_machines
    idx0 = [0]*num_jobs
    q = [(0, job_end0, machine_end0, idx0, [], 0)]

    visited = {}
    start_time = time.time()
    last_status = start_time
    nodes = 0
    cut_count = 0

    def state_key(idx, job_end, machine_end):
        return tuple(idx), tuple(machine_end), tuple(job_end)

    def path_str(p):
        return " > ".join(f"J{j+1}-O{op+1}-M{m+1}@{s}" for j, op, m, s, d in p)

    while q:
        g, job_end, machine_end, idx, p, ms = q.pop()
        nodes += 1

        now = time.time()
        if now - last_status > 10:
            print(f"[{now-start_time:.1f}s] Nodes: {nodes}, Best: {best}, Queue: {len(q)}, Cut: {cut_count}")
            last_status = now

        if g >= best:
            cut_count += 1
            continue

        sk = state_key(idx, job_end, machine_end)
        if sk in visited and visited[sk] <= g:
            cut_count += 1
            continue
        visited[sk] = g

        # Vollständiger Plan?
        if all(idx[j] == len(jobs[j]) for j in range(num_jobs)):
            if ms < best:
                best, plan = ms, p
                print(f"[DONE] New best: {best}, Path: {path_str(p)}")
            continue

        # Branching: jede mögliche nächste Operation
        for j in range(num_jobs):
            if idx[j] < len(jobs[j]):
                op_idx = idx[j]
                m, d = jobs[j][op_idx]
                start = max(job_end[j], machine_end[m])

                job_end2 = job_end[:]
                machine_end2 = machine_end[:]
                idx2 = idx[:]

                job_end2[j] = start + d
                machine_end2[m] = start + d
                idx2[j] += 1
                ms2 = max(ms, start + d)

                # Einfache Restzeit-Schranken (wie bei dir)
                rb_job = max(
                    job_end2[x] + sum(dd for _, dd in jobs[x][idx2[x]:])
                    for x in range(num_jobs)
                )
                rb_machine = max(
                    machine_end2[i] +
                    sum(dd for x in range(num_jobs)
                        for mi, dd in jobs[x][idx2[x]:] if mi == i)
                    for i in range(num_machines)
                )
                simple_lb = max(ms2, rb_job, rb_machine)

                # Konfliktsensitiver Graph-LB (kritischer Pfad mit Maschinenkanten aus p)
                p2 = p + [(j, op_idx, m, start, d)]
                graph_lb = longest_path_lower_bound(p2)

                g2 = max(simple_lb, graph_lb)
                #print(f"ms2={ms2}, simple_lb={simple_lb}, graph_lb={graph_lb} => g2={g2}")
                if g2 < best:
                    q.append((g2, job_end2, machine_end2, idx2, p2, ms2))
                else:
                    cut_count += 1

    return plan, best, nodes, time.time() - start_time

# -------------------------
# Hauptprogramm
# -------------------------

if __name__ == "__main__":
    plan, makespan, nodes, runtime = branch_and_bound()

    print("\n=== Branch and Bound Result ===")
    print(f"Makespan: {makespan}")
    print(f"Nodes:    {nodes}")
    print(f"Runtime:  {runtime:.1f}s\n")

    fig, ax = plt.subplots()
    for j, op_idx, m, s, d in plan:
        ax.broken_barh([(s, d)], (j*10, 9),
                       facecolors=machine_colors[m % len(machine_colors)])
        ax.text(s + d/2, j*10 + 4.5, f"M{m+1}",
                ha='center', va='center', color='white')

    ax.set_yticks([j*10 + 4.5 for j in range(num_jobs)])
    ax.set_yticklabels([f"J{j+1}" for j in range(num_jobs)])
    ax.set_xlabel('Time')
    ax.set_title(f"Makespan = {makespan}")
    plt.tight_layout()
    plt.show()
