# Baseline Branch & Bound „so wie aktuell“, Big-M = 1000, keine besonderen Parameter

import matplotlib.pyplot as plt
from pyscipopt import Model
import psutil
import os
import time
import threading

# --------------------
# Jobdaten
# --------------------
drei = {
    0: [(0,2),(1,5),(2,4)],
    1: [(1,2),(2,3),(0,5)],
    2: [(2,4),(0,2),(1,3)]
}

sechs = [
    [(2,1), (0,3), (1,6), (3,7), (2,3), (4,6)],
    [(1,8), (2,5), (4,10), (5,10), (0,10), (3,4)],
    [(2,5), (3,4), (5,8), (0,9), (1,1), (4,7)],
    [(1,5), (0,5), (2,5), (3,3), (4,8), (5,9)],
    [(0,9), (3,3), (4,5), (5,4), (2,3), (1,1)],
    [(2,3), (3,3), (4,9), (1,10), (0,4), (5,1)]
]

zehn = [
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

# Daten auswählen
jobs = {j: zehn[j] for j in range(10)}
machines = set(m for job in jobs.values() for m, _ in job)

M = 1000  # Big-M

# Operationen
ops = []
for j, job in jobs.items():
    for k, (m, p) in enumerate(job):
        ops.append((j, k, m, p))

# Modell
model = Model("JobShop_BranchAndBound")

S = {}
for j, k, m, p in ops:
    S[j, k] = model.addVar(lb=0, vtype="C", name=f"S_{j}_{k}")

Cmax = model.addVar(lb=0, vtype="C", name="Cmax")

# Job-Reihenfolge
for j, job in jobs.items():
    for k in range(len(job) - 1):
        _, p = job[k]
        model.addCons(S[j, k+1] >= S[j, k] + p)

# Maschinenkonflikte
Y = {}
for m in machines:
    ops_m = [(j, k, p) for j, k, mm, p in ops if mm == m]
    for i in range(len(ops_m)):
        for j in range(i + 1, len(ops_m)):
            j1, k1, p1 = ops_m[i]
            j2, k2, p2 = ops_m[j]

            y = model.addVar(vtype="B", name=f"y_{j1}_{k1}_{j2}_{k2}")
            Y[(j1, k1, j2, k2)] = y

            model.addCons(S[j1, k1] + p1 <= S[j2, k2] + M * (1 - y))
            model.addCons(S[j2, k2] + p2 <= S[j1, k1] + M * y)

# Makespan
for j, k, m, p in ops:
    model.addCons(Cmax >= S[j, k] + p)

model.setObjective(Cmax, "minimize")

# Cuts: Standard (an)
model.setParam("separating/maxrounds", -1)
model.setParam("separating/maxroundsroot", -1)
# Heuristiken: Standard (an)
model.setHeuristics(True)
# Presolve / Propagation: Standard
model.setParam("presolving/maxrounds", -1)
model.setParam("propagating/maxrounds", -1)
# Konflikte: Standard
model.setParam("conflict/enable", True)

# Speicher-Monitoring
process = psutil.Process(os.getpid())
memory_log = []
stop_monitor = False

def monitor_memory():
    global stop_monitor
    start_time = time.time()
    while not stop_monitor:
        t = time.time() - start_time
        mem = process.memory_info().rss
        memory_log.append((t, mem))
        time.sleep(0.1)

monitor_thread = threading.Thread(target=monitor_memory, daemon=True)

# Lösen + Messung
time_start = time.time()
monitor_thread.start()

model.optimize()

stop_monitor = True
monitor_thread.join()
time_end = time.time()

elapsed_time = time_end - time_start
peak_memory = max(m for _, m in memory_log)
peak_memory_mb = peak_memory / (1024 ** 2)

total_time = memory_log[-1][0]
area = 0.0

for i in range(len(memory_log) - 1):
    t0, m0 = memory_log[i]
    t1, _ = memory_log[i + 1]
    dt = t1 - t0
    area += m0 * dt

avg_memory = area / total_time
avg_memory_mb = avg_memory / (1024 ** 2)

# Ergebnisse
print("\n==============================")
print("ERGEBNISSE")
print("==============================")
print("Optimaler Makespan:", model.getObjVal())
print(f"Laufzeit: {elapsed_time:.2f} Sekunden")
print(f"Peak RAM-Verbrauch: {peak_memory_mb:.2f} MB")
print(f"Ø RAM (zeitgewichtet): {avg_memory_mb:.2f} MB")

# Gantt-Diagramm
solution_ops = []
for j, k, m, p in ops:
    solution_ops.append({
        "job": j,
        "op": k,
        "machine": m,
        "start": model.getVal(S[j, k]),
        "duration": p
    })

def plot_gantt(operations):
    fig, ax = plt.subplots(figsize=(10, 5))
    machines = sorted(set(op["machine"] for op in operations))
    machine_y = {m: i for i, m in enumerate(machines)}
    colors = plt.cm.tab10.colors

    for op in operations:
        y = machine_y[op["machine"]]
        ax.barh(y, op["duration"], left=op["start"],
                color=colors[op["job"] % len(colors)],
                edgecolor="black")
        ax.text(op["start"] + op["duration"] / 2, y,
                f"J{op['job']}", ha="center", va="center",
                color="white", fontsize=9)

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f"Maschine {m}" for m in machines])
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Maschine")
    ax.set_title("Job-Shop Gantt-Diagramm")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("Diagramm.png", dpi=300)
    print("Diagramm gespeichert als Diagramm.png")

plot_gantt(solution_ops)

def plot_memory_usage(memory_log):
    times = [t for t, _ in memory_log]
    mem_mb = [m / (1024 ** 2) for _, m in memory_log]

    plt.figure(figsize=(10, 4))
    plt.plot(times, mem_mb, linewidth=2)
    plt.xlabel("Zeit [s]")
    plt.ylabel("RAM [MB]")
    plt.title("Speicherverbrauch über die Zeit (Branch & Bound)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("memory_usage.png", dpi=300)
    print("Speicherverlauf gespeichert als: memory_usage.png")

plot_memory_usage(memory_log)
