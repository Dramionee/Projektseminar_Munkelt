# tiefensuche

import time
import matplotlib.pyplot as plt

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
anzahl_maschinen = 1 + max(m for job in auftraege for m, _ in job)
maschinenfarben = ['tab:blue','tab:orange','tab:green','tab:red','tab:olive','tab:purple','tab:grey','tab:cyan']

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

def branch_and_bound_with_memo_and_progress():
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

    start_time = time.time()
    last_status = start_time
    nodes = 0

    def state_key(idx, zm):
        return tuple(idx), tuple(zm)

    while q:
        g, za, zm, idx, p, ms = q.pop()  # LIFO Entnahme
        nodes += 1

        now = time.time()
        if now - last_status > 10:
            print(f"[{now-start_time:.1f}s] Knoten: {nodes}, Aktuell bestes Makespan: {best}, Queue: {len(q)}")
            last_status = now

        if g >= best:
            continue

        sk = state_key(idx, zm)
        if sk in visited and visited[sk] <= g:
            continue
        visited[sk] = g

        if all(idx[i] == len(auftraege[i]) for i in range(anzahl_auftraege)):
            runtime = now - start_time
            print(f">>> [Lösung] Makespan={ms} (bisher Optimum={best}) nach {runtime:.1f}s, {nodes} Knoten")
            if ms < best:
                best, plan = ms, p
            continue
        
        # alle möglichen nächsten Operationen sammeln
        candidates = []

        for a in range(anzahl_auftraege):
            if idx[a] < len(auftraege[a]):
                m, d = auftraege[a][idx[a]]
                s = max(za[a], zm[m])
                candidates.append((a, m, d, s))

        # nur Suchordnung: kleinste Startzeit zuerst
        candidates.sort(key=lambda x: (x[3], x[1], -x[2]))

        for (a, m, d, s) in reversed(candidates):

                m, d = auftraege[a][idx[a]]
                s = max(za[a], zm[m])
                za2, zm2, idx2 = za[:], zm[:], idx[:]
                za2[a] = zm2[m] = s + d
                idx2[a] += 1
                ms2 = max(ms, s + d)

                ra = max(za2[j] + sum(dd for _,dd in auftraege[j][idx2[j]:]) for j in range(anzahl_auftraege))
                rm = max(zm2[i] + sum(dd for j in range(anzahl_auftraege)
                                      for mi, dd in auftraege[j][idx2[j]:] if mi == i)
                         for i in range(anzahl_maschinen))

                g2 = max(ms2, ra, rm)
                q.append((g2, za2, zm2, idx2, p + [(a, m, s, d)], ms2))

    print(f"\nFERTIG. Optimum: {best}. Gesamtknoten: {nodes}. Laufzeit: {time.time()-start_time:.1f} Sekunden.")

    return plan, best

if __name__ == "__main__":
    plan, makespan = branch_and_bound_with_memo_and_progress()

    print("\n[Optimum]:")
    for a, m, s, d in plan:
        print(f"  A{a+1} M{m+1} [{s}-{s+d}]")

    fig, ax = plt.subplots()
    for a, m, s, d in plan:
        ax.broken_barh([(s, d)], (a*10, 9), facecolors=maschinenfarben[m % len(maschinenfarben)])
        ax.text(s + d/2, a*10 + 4.5, f"M{m+1}", ha='center', va='center', color='white', fontsize=9)
    ax.set_yticks([a*10 + 4.5 for a in range(anzahl_auftraege)])
    ax.set_yticklabels([f"A{a+1}" for a in range(anzahl_auftraege)])
    ax.set_xlabel('Zeit')
    ax.set_title(f"Makespan = {makespan}")
    plt.tight_layout()
    plt.show()
