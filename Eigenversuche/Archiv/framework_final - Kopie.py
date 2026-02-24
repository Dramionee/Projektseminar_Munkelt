# ==========================================================
#  SCIP Job-Shop (Big-M) + Gantt im Style deines Freundes
#  Basis: framework copy 2 (SCIP MIP) + Gantt/Memory aus TXT
# ==========================================================

import os
import time
import tracemalloc
import psutil
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyscipopt import Model

# --------------------
# Jobdaten (FT10)
# --------------------
zehn = [
[(2,1), (0,3), (1,6), (3,7), (2,3), (4,6)],
[(1,8), (2,5), (4,10), (5,10), (0,10), (3,4)],
[(2,5), (3,4), (5,8), (0,9), (1,1), (4,7)],
[(1,5), (0,5), (2,5), (3,3), (4,8), (5,9)],
[(0,9), (3,3), (4,5), (5,4), (2,3), (1,1)],
[(2,3), (3,3), (4,9), (1,10), (0,4), (5,1)]
]

# In deinem Framework: jobs als Dict
jobs = {j: zehn[j] for j in range(6)}
NUM_JOBS = len(jobs)
OPS_PER_JOB = len(jobs[0])
NUM_MACHINES = OPS_PER_JOB  # FT10: 10 Maschinen

machines = set(m for job in jobs.values() for m, _ in job)

# Global sinnvolles M (wie in framework copy 2)
M = sum(p for job in jobs.values() for _, p in job)

# Operationen flach machen
ops = []
for j, job in jobs.items():
    for k, (m, p) in enumerate(job):
        ops.append((j, k, m, p))


# ==========================================================
#  GANTT (Style deines Freundes)
# ==========================================================
def create_graphical_gantt_from_scip(solution_ops, title: str, filename: str):
    """
    solution_ops: List[dict] mit keys: job, op, machine, start, duration
    Stil übernommen aus Code - Kopie.txt (Farben + Layout + Legende).
    """
    # Farbpalette für Jobs (10 verschiedene Farben) – exakt wie in TXT
    colors = [
        '#5DADE2',  # Blau
        '#F1948A',  # Rosa
        '#F8B878',  # Orange
        '#85C1E2',  # Hellblau
        '#58D68D',  # Hellgrün
        '#48C9B0',  # Türkis
        '#BB8FCE',  # Lila
        '#AF7AC5',  # Violett
        '#8B4513',  # Braun
        '#FF8C00',  # Dunkelorange
    ]

    # Max time (Makespan) aus Lösung
    max_time = 0.0
    for op in solution_ops:
        end_time = op["start"] + op["duration"]
        if end_time > max_time:
            max_time = end_time

    fig, ax = plt.subplots(figsize=(16, 8))

    # Gruppiere Operationen nach Maschine
    machine_ops = {m: [] for m in range(NUM_MACHINES)}
    for op in solution_ops:
        machine_ops[int(op["machine"])].append(op)

    # Zeichne Balken
    for m in range(NUM_MACHINES):
        # optional: sortieren für sauberes Bild
        machine_ops[m].sort(key=lambda x: x["start"])
        for op in machine_ops[m]:
            color = colors[int(op["job"]) % len(colors)]
            ax.barh(
                m,
                op["duration"],
                left=op["start"],
                height=0.6,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )

    # Maschinenlabels wie TXT: M00..M09
    ax.set_yticks(range(NUM_MACHINES))
    ax.set_yticklabels([f"M{i:02d}" for i in range(NUM_MACHINES)])
    ax.set_ylabel("Maschinen", fontsize=12, fontweight="bold")

    ax.set_xlabel("Zeit (in Minuten)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max_time)

    ax.set_title(f"{title}\nFT10 Makespan = {int(max_time)}", fontsize=14, fontweight="bold")

    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Legende rechts
    legend_patches = [
        mpatches.Patch(color=colors[i], label=f"Job {i:02d}:{i:04d}")
        for i in range(NUM_JOBS)
    ]
    ax.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Job",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Gantt-Diagramm gespeichert: {filename}")


# ==========================================================
#  SCIP Modell (wie framework copy 2)
# ==========================================================
model = Model("JobShop_BranchAndBound")

# Startzeiten
S = {}
for j, k, m, p in ops:
    S[j, k] = model.addVar(lb=0, vtype="C", name=f"S_{j}_{k}")

# Makespan
Cmax = model.addVar(lb=0, vtype="C", name="Cmax")

# Job-Reihenfolge (Technologie-Reihenfolge)
for j, job in jobs.items():
    for k in range(len(job) - 1):
        _, p = job[k]
        model.addCons(S[j, k + 1] >= S[j, k] + p)

# Maschinenkonflikte (Big-M Disjunktionsconstraints)
Y = {}
for m in machines:
    ops_m = [(j, k, p) for j, k, mm, p in ops if mm == m]
    for i in range(len(ops_m)):
        for jj in range(i + 1, len(ops_m)):
            j1, k1, p1 = ops_m[i]
            j2, k2, p2 = ops_m[jj]

            y = model.addVar(vtype="B", name=f"y_{j1}_{k1}_{j2}_{k2}")
            Y[(j1, k1, j2, k2)] = y

            model.addCons(S[j1, k1] + p1 <= S[j2, k2] + M * (1 - y))
            model.addCons(S[j2, k2] + p2 <= S[j1, k1] + M * y)

# Makespan-Definition
for j, k, m, p in ops:
    model.addCons(Cmax >= S[j, k] + p)

model.setObjective(Cmax, "minimize")

# "Standard" Einstellungen wie in framework copy 2
model.setParam("separating/maxrounds", -1)
model.setParam("separating/maxroundsroot", -1)
model.setHeuristics(True)
model.setParam("presolving/maxrounds", -1)
model.setParam("propagating/maxrounds", -1)
model.setParam("conflict/enable", True)

# (Empfehlung für faire Vergleiche)
# model.setIntParam("parallel/maxnthreads", 1)


# ==========================================================
#  Lösen + Zeit + Speichermessung (wie im TXT)
# ==========================================================
print("\n" + "=" * 80)
print("STARTE SCIP (MIP / Branch-and-Cut) FT10")
print("=" * 80)
print(f"Globales Big-M: {M}")

overall_start = time.time()

tracemalloc.start()
process = psutil.Process(os.getpid())
memory_start_mb = process.memory_info().rss / 1024 / 1024  # MB

model.optimize()

overall_end = time.time()
total_seconds = overall_end - overall_start

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

memory_end_mb = process.memory_info().rss / 1024 / 1024  # MB
memory_used_mb = memory_end_mb - memory_start_mb

# ==========================================================
#  Ergebnisse
# ==========================================================
print("\n" + "#" * 80)
print("ERGEBNISSE")
print("#" * 80)

# Achtung: Wenn Zeitlimit/Abbruch, kann ObjVal evtl. nicht verfügbar sein
status = model.getStatus()
print(f"Solver-Status:         {status}")

if model.getNSols() > 0:
    print(f"Optimaler Makespan:     {model.getObjVal()}")
else:
    print("Keine Lösung gefunden (bis Abbruch).")

print(f"\n--- PERFORMANCE METRIKEN ---")
print(f"Gesamtzeit:             {total_seconds:.2f} Sekunden (~ {total_seconds/60:.2f} Minuten)")

print(f"\n--- SPEICHER-NUTZUNG ---")
print(f"Aktueller tracemalloc:  {current / 1024 / 1024:.2f} MB")
print(f"Peak tracemalloc:       {peak / 1024 / 1024:.2f} MB")
print(f"RSS Start:              {memory_start_mb:.2f} MB")
print(f"RSS Ende:               {memory_end_mb:.2f} MB")
print(f"RSS Änderung:           {memory_used_mb:.2f} MB")

# ==========================================================
#  Gantt erzeugen (nur wenn Lösung existiert)
# ==========================================================
if model.getNSols() > 0:
    solution_ops = []
    for j, k, m, p in ops:
        solution_ops.append(
            {
                "job": j,
                "op": k,
                "machine": m,
                "start": float(model.getVal(S[j, k])),
                "duration": p,
            }
        )

    gantt_folder = "gantt_charts"
    os.makedirs(gantt_folder, exist_ok=True)

    gantt_filename = os.path.join(gantt_folder, "gantt_SCIP_FT10.png")
    create_graphical_gantt_from_scip(
        solution_ops,
        title="SCIP (Branch-and-Cut) – FT10",
        filename=gantt_filename,
    )

    print(f"\nGantt gespeichert unter: {gantt_filename}")
