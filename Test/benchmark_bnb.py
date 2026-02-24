#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_bnb.py
Einheitliches Benchmark-Gerüst für Branch-and-Bound beim Job-Shop Scheduling (JSSP).

Ziele:
- Vergleichbarkeit: gleiche Instanz, gleiche Metriken, gleiche Time-Limits
- Mehrere Varianten (DFS/LIFO, Heap/Best-First, Ordering, UB-Heuristik, GT-Branching)
- Saubere Ausgabe für Tabellen in 4.2.2

Aufrufbeispiele:
  python benchmark_bnb.py --instance toy3x3 --time_limit 10 --repeats 3
  python benchmark_bnb.py --instance ft06   --time_limit 60 --repeats 3
  python benchmark_bnb.py --instance abz6   --time_limit 60 --repeats 1
  python benchmark_bnb.py --instance abz6   --time_limit 120 --variants heap_ub_gt heap_gt_branch

Hinweis:
- "Optimum bewiesen" ist nur dann True, wenn die Suche den Queue vollständig leert (exakt abgeschlossen),
  bevor das Time-Limit erreicht ist.
"""

import argparse
import heapq
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

Job = List[Tuple[int, int]]  # [(machine, duration), ...]
Instance = List[Job]

# ---------------------------
# Instanzen (deine Beispiele)
# ---------------------------

TOY_3x3: Instance = [
    [(0, 2), (1, 5), (2, 4)],
    [(1, 2), (2, 3), (0, 5)],
    [(2, 4), (0, 2), (1, 3)],
]

# ft06 (6x6), bekanntes Optimum: 55 (je nach Quelle/Definition; du hattest 51 für deine Variante/Instanz)
# Achtung: In deinen eigenen Messungen war die 6x6-Instanz mit Optimum 51. Das war wahrscheinlich eine andere 6x6.
# Diese ft06 hier ist genau die, die du früher gepostet hast.
FT06: Instance = [
    [(2, 1), (0, 3), (1, 6), (3, 7), (2, 3), (4, 6)],
    [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
    [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
    [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
    [(0, 9), (3, 3), (4, 5), (5, 4), (2, 3), (1, 1)],
    [(2, 3), (3, 3), (4, 9), (1, 10), (0, 4), (5, 1)],
]

# abz6 (10x10) aus deinem Code (Optimum 943 in Kommentar) – das war aber eine andere Instanz.
# Hier packe ich deine 10x10 aus Code 4/5 rein (die, bei der du ~960 als best found hattest).
ABZ6_LIKE_10x10: Instance = [
    [(0, 29), (1, 78), (2, 9), (3, 36), (4, 49), (5, 11), (6, 62), (7, 56), (8, 44), (9, 21)],
    [(0, 43), (2, 90), (4, 75), (9, 11), (3, 69), (1, 28), (6, 46), (5, 46), (7, 72), (8, 30)],
    [(1, 91), (0, 85), (3, 39), (2, 74), (8, 90), (5, 10), (7, 12), (6, 89), (9, 45), (4, 33)],
    [(1, 81), (2, 95), (0, 71), (4, 99), (6, 9), (8, 52), (7, 85), (3, 98), (9, 22), (5, 43)],
    [(2, 14), (0, 6), (1, 22), (5, 61), (3, 26), (4, 69), (8, 21), (7, 49), (9, 72), (6, 53)],
    [(2, 84), (1, 2), (5, 52), (3, 95), (8, 48), (9, 72), (0, 47), (6, 65), (4, 6), (7, 25)],
    [(1, 46), (0, 37), (3, 61), (2, 13), (6, 32), (5, 21), (9, 32), (8, 89), (7, 30), (4, 55)],
    [(2, 31), (0, 86), (1, 46), (5, 74), (4, 32), (6, 88), (8, 19), (9, 48), (7, 36), (3, 79)],
    [(0, 76), (1, 69), (3, 76), (5, 51), (2, 85), (9, 11), (6, 40), (7, 89), (4, 26), (8, 74)],
    [(1, 85), (0, 13), (2, 61), (6, 7), (8, 64), (9, 76), (5, 47), (3, 52), (4, 90), (7, 45)],
]

INSTANCES: Dict[str, Instance] = {
    "toy3x3": TOY_3x3,
    "ft06": FT06,
    "abz6": ABZ6_LIKE_10x10,
}

# ---------------------------
# Benchmark/Stats
# ---------------------------

@dataclass
class RunStats:
    variant: str
    instance: str
    time_limit_s: float
    runtime_s: float
    nodes_expanded: int
    pushed: int
    pruned_ub: int
    pruned_memo: int
    max_queue: int
    visited_size: int
    best_makespan: int
    proven_optimal: bool
    finished: bool  # finished == queue emptied before timeout
    note: str = ""


# ---------------------------
# Heuristiken (UB) und Branching
# ---------------------------

def greedy_ub_spt(inst: Instance) -> int:
    """Greedy UB: wähle stets die verfügbare nächste Operation mit kürzester Dauer (SPT), tie-break earliest start."""
    n = len(inst)
    mcount = 1 + max(m for job in inst for m, _ in job)
    idx = [0] * n
    za = [0] * n
    zm = [0] * mcount

    total_ops = sum(len(job) for job in inst)
    done = 0
    while done < total_ops:
        candidates = []
        for j in range(n):
            if idx[j] < len(inst[j]):
                m, d = inst[j][idx[j]]
                s = max(za[j], zm[m])
                candidates.append((d, s, j, m))
        d, s, j, m = min(candidates)  # shortest duration, then earliest start
        za[j] = s + d
        zm[m] = s + d
        idx[j] += 1
        done += 1
    return max(za)


def greedy_ub_earliest_start(inst: Instance) -> int:
    """Greedy UB: wähle stets die verfügbare nächste Operation mit kleinstem earliest-start s."""
    n = len(inst)
    mcount = 1 + max(m for job in inst for m, _ in job)
    idx = [0] * n
    za = [0] * n
    zm = [0] * mcount

    total_ops = sum(len(job) for job in inst)
    done = 0
    while done < total_ops:
        candidates = []
        for j in range(n):
            if idx[j] < len(inst[j]):
                m, d = inst[j][idx[j]]
                s = max(za[j], zm[m])
                candidates.append((s, -d, j, m, d))
        s, nd, j, m, d = min(candidates)  # earliest start, tie-break longer first (optional)
        za[j] = s + d
        zm[m] = s + d
        idx[j] += 1
        done += 1
    return max(za)


def compute_rest_job(inst: Instance) -> List[List[int]]:
    """rest_job[j][i] = Summe der durations ab i (inkl i) im Job j; rest_job[j][len(job)] = 0."""
    rest = []
    for job in inst:
        L = len(job)
        arr = [0] * (L + 1)
        s = 0
        for i in range(L - 1, -1, -1):
            s += job[i][1]
            arr[i] = s
        arr[L] = 0
        rest.append(arr)
    return rest


def compute_total_machine_load(inst: Instance, mcount: int) -> List[int]:
    tot = [0] * mcount
    for job in inst:
        for m, d in job:
            tot[m] += d
    return tot


# ---------------------------
# Kern: Branch & Bound
# ---------------------------

@dataclass
class Node:
    # state
    idx: List[int]
    za: List[int]
    zm: List[int]
    ms: int
    # bookkeeping (optional)
    g: int


def state_key(idx: List[int], za: List[int], zm: List[int], mode: str):
    if mode == "idx":
        return tuple(idx)
    if mode == "idx_zm":
        return (tuple(idx), tuple(zm))
    if mode == "idx_za_zm":
        return (tuple(idx), tuple(za), tuple(zm))
    raise ValueError(f"Unknown memo key mode: {mode}")


def lower_bound_simple(
    inst: Instance,
    idx: List[int],
    za: List[int],
    zm: List[int],
    ms: int,
    rest_job: List[List[int]],
    mach_remaining: List[int],
) -> int:
    """
    Simple LB:
      ra = max_j ( za[j] + rest_job[j][idx[j]] )
      rm = max_m ( zm[m] + mach_remaining[m] )
      LB = max(ms, ra, rm)
    mach_remaining[m] = Summe durations der noch nicht eingeplanten Ops auf Maschine m (inkrementell gepflegt).
    """
    ra = 0
    for j in range(len(inst)):
        v = za[j] + rest_job[j][idx[j]]
        if v > ra:
            ra = v
    rm = 0
    for m in range(len(zm)):
        v = zm[m] + mach_remaining[m]
        if v > rm:
            rm = v
    return max(ms, ra, rm)


def expand_candidates_all(
    inst: Instance,
    idx: List[int],
    za: List[int],
    zm: List[int],
) -> List[Tuple[int, int, int, int]]:
    """Alle nächsten Operationen je Job: (job, machine, dur, earliest_start)."""
    cand = []
    n = len(inst)
    for j in range(n):
        k = idx[j]
        if k < len(inst[j]):
            m, d = inst[j][k]
            s = max(za[j], zm[m])
            cand.append((j, m, d, s))
    return cand


def expand_candidates_gt_conflict(
    inst: Instance,
    idx: List[int],
    za: List[int],
    zm: List[int],
) -> List[Tuple[int, int, int, int]]:
    """
    GT-inspiriertes Konflikt-Branching:
      - bilde alle Kandidaten (nächste Ops)
      - wähle Kandidat mit kleinstem earliest start s0
      - branch nur über Kandidaten auf derselben Maschine wie dieser Kandidat
    """
    cand = expand_candidates_all(inst, idx, za, zm)
    if not cand:
        return []
    j0, m0, d0, s0 = min(cand, key=lambda x: x[3])
    conflict = [x for x in cand if x[1] == m0]
    return conflict


def order_candidates(cand: List[Tuple[int, int, int, int]], mode: str) -> List[Tuple[int, int, int, int]]:
    """
    mode:
      - "none": keine Sortierung
      - "earliest": sortiert nach (s, -d)
      - "spt": sortiert nach (d, s)
    """
    if mode == "none":
        return cand
    if mode == "earliest":
        return sorted(cand, key=lambda x: (x[3], -x[2]))
    if mode == "spt":
        return sorted(cand, key=lambda x: (x[2], x[3]))
    raise ValueError(f"Unknown ordering mode: {mode}")


def bnb_solve(
    inst: Instance,
    variant_name: str,
    time_limit_s: float,
    queue_mode: str,          # "lifo" oder "heap"
    memo_mode: str,           # "idx", "idx_zm", "idx_za_zm"
    candidate_mode: str,      # "all" oder "gt_conflict"
    ordering_mode: str,       # "none", "earliest", "spt"
    initial_ub_mode: str,     # "inf", "greedy_spt", "greedy_earliest"
) -> RunStats:
    n = len(inst)
    mcount = 1 + max(m for job in inst for m, _ in job)
    rest_job = compute_rest_job(inst)
    total_mach = compute_total_machine_load(inst, mcount)

    # initial state
    idx0 = [0] * n
    za0 = [0] * n
    zm0 = [0] * mcount
    ms0 = 0

    # mach_remaining initial = total load
    mach_rem0 = total_mach[:]

    # initial UB
    if initial_ub_mode == "inf":
        best = 10**18
        note = "UB=inf"
    elif initial_ub_mode == "greedy_spt":
        best = greedy_ub_spt(inst)
        note = f"UB=greedy_spt({best})"
    elif initial_ub_mode == "greedy_earliest":
        best = greedy_ub_earliest_start(inst)
        note = f"UB=greedy_earliest({best})"
    else:
        raise ValueError(f"Unknown initial_ub_mode: {initial_ub_mode}")

    # root LB
    g0 = lower_bound_simple(inst, idx0, za0, zm0, ms0, rest_job, mach_rem0)

    # queue init
    start = time.time()
    pushed = 0
    nodes = 0
    pruned_ub = 0
    pruned_memo = 0
    max_queue = 0

    visited: Dict[Any, int] = {}

    if queue_mode == "heap":
        heap = []
        counter = 0
        heapq.heappush(heap, (g0, counter, idx0, za0, zm0, ms0, mach_rem0))
        pushed += 1
        counter += 1
        max_queue = 1
    elif queue_mode == "lifo":
        stack = [(g0, idx0, za0, zm0, ms0, mach_rem0)]
        pushed += 1
        max_queue = 1
    else:
        raise ValueError(f"Unknown queue_mode: {queue_mode}")

    finished = False

    while True:
        now = time.time()
        if now - start > time_limit_s:
            break

        # pop next
        if queue_mode == "heap":
            if not heap:
                finished = True
                break
            g, _, idx, za, zm, ms, mach_rem = heapq.heappop(heap)
        else:
            if not stack:
                finished = True
                break
            g, idx, za, zm, ms, mach_rem = stack.pop()

        nodes += 1

        # prune by UB
        if g >= best:
            pruned_ub += 1
            continue

        # memo
        sk = state_key(idx, za, zm, memo_mode)
        prev = visited.get(sk)
        if prev is not None and prev <= g:
            pruned_memo += 1
            continue
        visited[sk] = g

        # goal check
        done = True
        for j in range(n):
            if idx[j] < len(inst[j]):
                done = False
                break
        if done:
            # exact complete schedule; ms is makespan
            if ms < best:
                best = ms
            continue

        # candidates
        if candidate_mode == "all":
            cand = expand_candidates_all(inst, idx, za, zm)
        elif candidate_mode == "gt_conflict":
            cand = expand_candidates_gt_conflict(inst, idx, za, zm)
        else:
            raise ValueError(f"Unknown candidate_mode: {candidate_mode}")

        cand = order_candidates(cand, ordering_mode)

        # For LIFO: push in reversed order so first candidate is expanded first
        it = reversed(cand) if queue_mode == "lifo" else cand

        for (j, m, d, s) in it:
            # child state copy
            idx2 = idx[:]     # next-op pointers
            za2 = za[:]
            zm2 = zm[:]
            mach_rem2 = mach_rem[:]  # incremental machine remaining

            # schedule op (j, idx[j]) on machine m
            start_t = max(za2[j], zm2[m])
            finish = start_t + d
            za2[j] = finish
            zm2[m] = finish
            idx2[j] += 1
            ms2 = ms if ms >= finish else finish

            # update machine remaining: we just scheduled one op on machine m of duration d
            mach_rem2[m] -= d

            g2 = lower_bound_simple(inst, idx2, za2, zm2, ms2, rest_job, mach_rem2)

            if g2 >= best:
                pruned_ub += 1
                continue

            if queue_mode == "heap":
                heapq.heappush(heap, (g2, counter, idx2, za2, zm2, ms2, mach_rem2))
                counter += 1
                pushed += 1
                if len(heap) > max_queue:
                    max_queue = len(heap)
            else:
                stack.append((g2, idx2, za2, zm2, ms2, mach_rem2))
                pushed += 1
                if len(stack) > max_queue:
                    max_queue = len(stack)

    runtime = time.time() - start
    proven_opt = finished  # only if queue emptied
    best_ms = int(best) if best < 10**17 else -1

    return RunStats(
        variant=variant_name,
        instance="",
        time_limit_s=time_limit_s,
        runtime_s=runtime,
        nodes_expanded=nodes,
        pushed=pushed,
        pruned_ub=pruned_ub,
        pruned_memo=pruned_memo,
        max_queue=max_queue,
        visited_size=len(visited),
        best_makespan=best_ms,
        proven_optimal=proven_opt,
        finished=finished,
        note=note,
    )


# ---------------------------
# Varianten-Definitionen
# ---------------------------

def variants_catalog() -> Dict[str, Dict[str, str]]:
    """
    Liefert Varianten als Konfigurationen.
    Diese Varianten sind so gewählt, dass sie exakt den "Weg" abbilden, den du im Bericht beschreibst.
    """
    return {
        # Baseline: DFS/LIFO, alle Kandidaten, keine Ordering, UB=inf
        "lifo_baseline": dict(queue_mode="lifo", memo_mode="idx_za_zm", candidate_mode="all",
                             ordering_mode="none", initial_ub_mode="inf"),

        # Best-First: Heap statt LIFO, sonst gleich
        "heap_baseline": dict(queue_mode="heap", memo_mode="idx_za_zm", candidate_mode="all",
                              ordering_mode="none", initial_ub_mode="inf"),

        # DFS + Ordering: LIFO, Kandidaten nach earliest start (wie Code 6/8), UB=inf
        "lifo_order_earliest": dict(queue_mode="lifo", memo_mode="idx_za_zm", candidate_mode="all",
                                    ordering_mode="earliest", initial_ub_mode="inf"),

        # Heap + Ordering: Best-First plus earliest ordering (oft hilfreich)
        "heap_order_earliest": dict(queue_mode="heap", memo_mode="idx_za_zm", candidate_mode="all",
                                    ordering_mode="earliest", initial_ub_mode="inf"),

        # Heap + UB: Greedy SPT als Start-UB (zeigt Pruning-Effekt)
        "heap_ub_spt": dict(queue_mode="heap", memo_mode="idx_za_zm", candidate_mode="all",
                            ordering_mode="none", initial_ub_mode="greedy_spt"),

        # Heap + UB + Ordering
        "heap_ub_spt_order": dict(queue_mode="heap", memo_mode="idx_za_zm", candidate_mode="all",
                                  ordering_mode="earliest", initial_ub_mode="greedy_spt"),

        # GT-Konflikt-Branching (wie dein Code 4): reduziert Branchingbreite
        "heap_gt_branch": dict(queue_mode="heap", memo_mode="idx_za_zm", candidate_mode="gt_conflict",
                               ordering_mode="none", initial_ub_mode="greedy_spt"),

        # GT-Konflikt + Ordering
        "heap_gt_branch_order": dict(queue_mode="heap", memo_mode="idx_za_zm", candidate_mode="gt_conflict",
                                     ordering_mode="earliest", initial_ub_mode="greedy_spt"),
    }


# ---------------------------
# CLI / Ausgabe
# ---------------------------

def print_run(stats: RunStats):
    ok = "YES" if stats.proven_optimal else ("NO" if stats.best_makespan != -1 else "NO (no UB)")
    print(
        f"[{stats.variant}]  "
        f"time={stats.runtime_s:.3f}s  "
        f"nodes={stats.nodes_expanded}  pushed={stats.pushed}  "
        f"pruneUB={stats.pruned_ub}  pruneMemo={stats.pruned_memo}  "
        f"maxQ={stats.max_queue}  visited={stats.visited_size}  "
        f"best={stats.best_makespan}  proven={ok}  "
        f"({stats.note})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", choices=INSTANCES.keys(), default="ft06")
    ap.add_argument("--time_limit", type=float, default=60.0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Liste von Varianten (z.B. heap_baseline heap_ub_spt heap_gt_branch). Default: alle.")
    args = ap.parse_args()

    inst = INSTANCES[args.instance]
    cat = variants_catalog()

    if args.variants is None or len(args.variants) == 0:
        chosen = list(cat.keys())
    else:
        for v in args.variants:
            if v not in cat:
                raise SystemExit(f"Unknown variant: {v}. Available: {', '.join(cat.keys())}")
        chosen = args.variants

    print(f"=== JSSP BnB Benchmark ===")
    print(f"instance={args.instance}  jobs={len(inst)}  machines={1 + max(m for job in inst for m,_ in job)}  time_limit={args.time_limit}s  repeats={args.repeats}")
    print(f"variants={', '.join(chosen)}")
    print("")

    # runs
    all_stats: List[RunStats] = []

    for vname in chosen:
        conf = cat[vname]
        for r in range(args.repeats):
            stats = bnb_solve(
                inst=inst,
                variant_name=vname,
                time_limit_s=args.time_limit,
                queue_mode=conf["queue_mode"],
                memo_mode=conf["memo_mode"],
                candidate_mode=conf["candidate_mode"],
                ordering_mode=conf["ordering_mode"],
                initial_ub_mode=conf["initial_ub_mode"],
            )
            stats.instance = args.instance
            all_stats.append(stats)
            print_run(stats)

        print("")

    # simple aggregation (min runtime, min nodes) per variant
    print("=== Summary (best-of repeats) ===")
    by_var: Dict[str, List[RunStats]] = {}
    for s in all_stats:
        by_var.setdefault(s.variant, []).append(s)

    for vname in chosen:
        runs = by_var[vname]
        best_time = min(r.runtime_s for r in runs)
        best_nodes = min(r.nodes_expanded for r in runs)
        best_ms = min((r.best_makespan for r in runs if r.best_makespan != -1), default=-1)
        proven_any = any(r.proven_optimal for r in runs)
        print(f"{vname:22s}  best_ms={best_ms:5d}  best_time={best_time:8.3f}s  best_nodes={best_nodes:9d}  proven_any={proven_any}")

if __name__ == "__main__":
    main()