import time
import matplotlib.pyplot as plt
import heapq

# --- JSSP Instanz (10x10) ---
auftraege = [
[(2,1), (0,3), (1,6), (3,7), (2,3), (4,6)],
[(1,8), (2,5), (4,10), (5,10), (0,10), (3,4)],
[(2,5), (3,4), (5,8), (0,9), (1,1), (4,7)],
[(1,5), (0,5), (2,5), (3,3), (4,8), (5,9)],
[(0,9), (3,3), (4,5), (5,4), (2,3), (1,1)],
[(2,3), (3,3), (4,9), (1,10), (0,4), (5,1)]
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

def branch_and_bound(initial_ub=float("inf")):
    za = [0]*anzahl_auftraege
    zm = [0]*anzahl_maschinen
    idx = [0]*anzahl_auftraege

    initial_plan, initial_ms = giffler_thompson(auftraege)
    best = initial_ms
    plan = initial_plan

    print(f">>> Startlösung (Giffler–Thompson): Makespan={best}")

    visited = {}
    q = []
    q.append((0, za, zm, idx, [], 0))
    #heapq.heappush(q, (0, za, zm, idx, [], 0))  # (level=0, g=0, ...)
    start_time = time.time()
    last_status = start_time
    nodes = 0
    cutted = 0

    def state_key(idx, zm):
        return tuple(idx), tuple(zm)

    def pfad(p): return " > ".join(f"A{a+1}-M{m+1}@{s}" for a,m,s,d in p)
    while q:
        g, za, zm, idx, p, ms = q.pop()
        #g, za, zm, idx, p, ms = heapq.heappop(q)
        nodes += 1

        now = time.time()
        if now - last_status > 10:
            print(f"[{now-start_time:.1f}s] Knoten: {nodes}, Bestes: {best}, Queue: {len(q)}, Cutted: {cutted}")
            last_status = now

        # cut: schlechter als aktuelle UB
        if g >= best:
            cutted += 1
            #print(f"[CUT] Node mit Schranke {g2} wurde weggeschnitten (aktuell bestes: {best})")
            continue

        # memo
        sk = state_key(idx, zm)
        if sk in visited and visited[sk] <= g:
            #print(f"[MEMO] Bereits besucht, kein besserer Wert. Weiter.")
            continue
        visited[sk] = g

        # fertig?
        if all(idx[i] == len(auftraege[i]) for i in range(anzahl_auftraege)):
            #print(f"[PATH DONE] Vollständiger Pfad durchlaufen! Makespan: {ms}, Pfad: {pfad(p)}")
            if ms < best:
                best, plan = ms, p
                print(f"[UPDATE] Verbesserte Lösung: Makespan={ms} | Neuer bestes: {best} | Cutted: {cutted}")
            continue

        # expandieren
        for a in range(anzahl_auftraege):
            if idx[a] < len(auftraege[a]):
                m, d = auftraege[a][idx[a]]
                s = max(za[a], zm[m])
                za2, zm2, idx2 = za[:], zm[:], idx[:]
                za2[a] = zm2[m] = s+d
                idx2[a] += 1
                ms2 = max(ms, s+d)

                # untere Schranke
                ra = max(za2[j] + sum(dd for _,dd in auftraege[j][idx2[j]:]) for j in range(anzahl_auftraege))
                rm = max(zm2[i] + sum(dd for j in range(anzahl_auftraege)
                                      for mi,dd in auftraege[j][idx2[j]:] if mi==i)
                         for i in range(anzahl_maschinen))
                g2 = max(ms2, ra, rm)

                #print(f"{g2}")
                #print(f"[Expand] {pfad(p)} + A{a+1}-M{m+1}@{s} | Grenze={g2}")
                #print(f"-----------------------------------------------------------------------------")
                
                if g2 < best:
                    q.append((g2, za2, zm2, idx2, p+[(a,m,s,d)], ms2))
                    #heapq.heappush(q, (g2, za2, zm2, idx2, p + [(a, m, s, d)], ms2))
                else:
                    cutted += 1
                    #print(f"[CUT] Neuer Knoten mit Schranke {g2} wurde weggeschnitten (aktuell bestes: {best})")
                
    return plan, best, nodes, time.time() - start_time

# --- MAIN ---
if __name__ == "__main__":

    plan, makespan, nodes, runtime = branch_and_bound()

    print("\n=== Ergebnis Branch and Bound ===")
    print(f"Makespan: {makespan}")
    print(f"Knoten:   {nodes}")
    print(f"Laufzeit: {runtime:.1f}s\n")

    # Gantt-Diagramm
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
