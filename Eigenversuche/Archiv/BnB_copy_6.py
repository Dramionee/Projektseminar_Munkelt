import time
import heapq
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
    print("matplotlib not available — plotting disabled. Install with 'pip install matplotlib' to enable Gantt chart.")

# --- JSSP Instanz (10x10) ---
auftraege = [
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

anzahl_auftraege = len(auftraege)
anzahl_maschinen = 1 + max(m for job in auftraege for m,_ in job)
maschinenfarben = ['tab:blue','tab:orange','tab:green','tab:red','tab:olive',
                   'tab:purple','tab:grey','tab:cyan','tab:pink','tab:brown']

def giffler_thompson(auftraege):
    n = len(auftraege)
    k = len(auftraege[0])
    za = [0]*n
    zm = [0]*anzahl_maschinen
    idx = [0]*n

    plan = []
    finished_ops = 0

    while finished_ops < n*k:
        candidates = []
        for a in range(n):
            if idx[a] < k:
                m, d = auftraege[a][idx[a]]
                s = max(za[a], zm[m])
                candidates.append((a, m, d, s))

        a, m, d, s = min(candidates, key=lambda x: x[3])

        plan.append((a, m, s, d))
        za[a] = s + d
        zm[m] = s + d
        idx[a] += 1
        finished_ops += 1

    makespan = max(za)
    return plan, makespan


def greedy_initial_ub(auftraege):
    """Simple greedy list-scheduling: always schedule the available operation with smallest duration.
    Returns (plan, makespan)."""
    n = len(auftraege)
    k = len(auftraege[0])
    za = [0]*n
    zm = [0]*anzahl_maschinen
    idx = [0]*n
    plan = []
    finished_ops = 0

    while finished_ops < n*k:
        candidates = []
        for a in range(n):
            if idx[a] < k:
                m, d = auftraege[a][idx[a]]
                s = max(za[a], zm[m])
                candidates.append((a, m, d, s))

        # choose operation with smallest duration (SPT), tie-breaker earliest start
        a, m, d, s = min(candidates, key=lambda x: (x[2], x[3]))

        plan.append((a, m, s, d))
        za[a] = s + d
        zm[m] = s + d
        idx[a] += 1
        finished_ops += 1

    makespan = max(za)
    return plan, makespan

def branch_and_bound(initial_ub=float("inf")):
    za = [0]*anzahl_auftraege
    zm = [0]*anzahl_maschinen
    idx = [0]*anzahl_auftraege

    # Start without a Giffler–Thompson initial solution so B&B explores from scratch.
    # Optionally an initial upper bound can be provided via `initial_ub`.
    best = initial_ub
    plan = []

    if best < float("inf"):
        print(f">>> Startlösung (vorgegeben): Makespan={best}")
    else:
        # compute a cheap greedy initial upper bound (better than inf) so pruning is effective
        plan_g, ms_g = greedy_initial_ub(auftraege)
        best = ms_g
        plan = plan_g
        print(f">>> Greedy Startlösung: Makespan={best} (used for initial UB)")

    visited = {}
    q = []
    # Use a min-heap (best-first by lower bound). Include a tie-breaker counter
    # because lists are not directly comparable when bounds tie.
    counter = 0
    # compute initial lower bound for root
    ra0 = max(za[j] + sum(dd for _, dd in auftraege[j][idx[j]:]) for j in range(anzahl_auftraege))
    rm0 = max(zm[i] + sum(dd for j in range(anzahl_auftraege) for mi, dd in auftraege[j][idx[j]:] if mi == i)
              for i in range(anzahl_maschinen))
    g0 = max(0, ra0, rm0)
    #q=[]
    heapq.heappush(q, (g0, counter, za, zm, idx, [], 0))
    #q.append((g0, counter, za, zm, idx, [], 0))
    counter += 1
    start_time = time.time()
    last_status = start_time
    nodes = 0
    cutted = 0

    def state_key(idx, za, zm):
        return tuple(idx), tuple(zm), tuple(za)

    def pfad(p): return " > ".join(f"A{a+1}-M{m+1}@{s}" for a,m,s,d in p)
    while q:
        g, _, za, zm, idx, p, ms = heapq.heappop(q)
        #g, _, za, zm, idx, p, ms = q.pop(0)
        nodes += 1

        now = time.time()
        if now - last_status > 10:
            print(f"[{now-start_time:.1f}s] Knoten: {nodes}, Bestes: {best}, Queue: {len(q)}, Cutted: {cutted}")
            last_status = now

        # cut: schlechter als aktuelle UB
        if g > best:
            cutted += 1
            #print(f"[CUT] Node mit Schranke {g2} wurde weggeschnitten (aktuell bestes: {best})")
            continue

        # memo
        sk = state_key(idx, za, zm)
        if sk in visited and visited[sk] <= g:
            #print(f"[MEMO] Bereits besucht, kein besserer Wert. Weiter.")
            cutted += 1
            continue
        visited[sk] = g

        # fertig?
        if all(idx[i] == len(auftraege[i]) for i in range(anzahl_auftraege)):
            print(f"[PATH DONE] Vollständiger Pfad durchlaufen! Makespan: {ms}, Pfad: {pfad(p)}")
            if ms < best:
                best, plan = ms, p
                print(f"[UPDATE] Verbesserte Lösung: Makespan={ms} | Neuer bestes: {best} | Cutted: {cutted}")
            continue
        
        # expandieren: Wende Giffler–Thompson-Branching an
        # Erzeuge Kandidaten (verfügbare Operationen) und wähle das mit minimalem Startzeitpunkt
        candidates = []
        for a in range(anzahl_auftraege):
            if idx[a] < len(auftraege[a]):
                m, d = auftraege[a][idx[a]]
                s = max(za[a], zm[m])
                candidates.append((a, m, d, s))

        if not candidates:
            continue

        # wähle Kandidat mit kleinstem Start s
        a0, m0, d0, s0 = min(candidates, key=lambda x: x[3])

        # set C: alle Kandidaten, die dieselbe Maschine m0 nutzen
        C = [(a, m, d, s) for (a, m, d, s) in candidates if m == m0]

        for a, m, d, s in C:
            za2, zm2, idx2 = za[:], zm[:], idx[:]
            za2[a] = zm2[m] = s + d
            idx2[a] += 1
            ms2 = max(ms, s + d)

            # untere Schranke
            ra = max(za2[j] + sum(dd for _, dd in auftraege[j][idx2[j]:]) for j in range(anzahl_auftraege))
            rm = max(zm2[i] + sum(dd for j in range(anzahl_auftraege) for mi, dd in auftraege[j][idx2[j]:] if mi == i)
                     for i in range(anzahl_maschinen))
            g2 = max(ms2, ra, rm)

            if g2 < best:
                heapq.heappush(q, (g2, counter, za2, zm2, idx2, p + [(a, m, s, d)], ms2))
                #q.append((g2, counter, za2, zm2, idx2, p + [(a, m, s, d)], ms2))
                counter += 1
            else:
                cutted += 1

    return plan, best, nodes, time.time() - start_time

# --- MAIN ---
if __name__ == "__main__":
    # 1. GT als Startlösung


    # 2. B&B mit GT-UB
    plan, makespan, nodes, runtime = branch_and_bound()

    print("\n=== Ergebnis Branch and Bound ===")
    print(f"Makespan: {makespan}")
    print(f"Knoten:   {nodes}")
    print(f"Laufzeit: {runtime:.1f}s\n")

    # 3. Gantt-Chart
    if _HAS_MPL:
        fig, ax = plt.subplots()
        for a, m, s, d in plan:
            ax.broken_barh([(s, d)], (a*10, 9),
                           facecolors=maschinenfarben[m % len(maschinenfarben)])
            ax.text(s + d/2, a*10 + 4.5, f"M{m+1}",
                    ha='center', va='center', color='white')
        ax.set_yticks([a*10 + 4.5 for a in range(anzahl_auftraege)])
        ax.set_yticklabels([f"A{a+1}" for a in range(anzahl_auftraege)])
        ax.set_xlabel('Zeit')
        ax.set_title(f"Makespan = {makespan}")
        plt.tight_layout()
        plt.show()
    else:
        print("Plot skipped because matplotlib is not installed. To enable plotting, run: pip install matplotlib")
