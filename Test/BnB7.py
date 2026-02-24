
import time
import matplotlib.pyplot as plt


auftraege = [[(0,2),(1,5),(2,4)], [(1,2),(2,3),(0,5)], [(2,4),(0,2),(1,3)]]

anzahl_auftraege = len(auftraege)
anzahl_maschinen = 1 + max(m for job in auftraege for m, _ in job)
maschinenfarben = ['tab:blue','tab:orange','tab:green','tab:red','tab:olive','tab:purple','tab:grey','tab:cyan']

# restliche Bearbeitungszeit je Job und Index vorrechnen
restzeit_job = [
    [sum(d for _, d in job[i:]) for i in range(len(job)+1)]
    for job in auftraege
]

def branch_and_bound_with_memo_and_progress():
    za = [0]*anzahl_auftraege   
    zm = [0]*anzahl_maschinen
    idx = [0]*anzahl_auftraege

    best = 9999
    plan = []
    cutted = 0

    print(f">>> Startlösung (Giffler–Thompson): Makespan={best}")
    visited = {}

    q = []
    q.append((0, za, zm, idx, [], 0))

    start_time = time.time()
    last_status = start_time
    nodes = 0

    def state_key(idx):
        return (tuple(idx))

    while q:
        g, za, zm, idx, p, ms = q.pop()
        nodes += 1

        now = time.time()
        if now - last_status > 10:
            print(f"[{now-start_time:.1f}s] Knoten: {nodes}, Aktuell bestes Makespan: {best}, Queue: {len(q)}")
            last_status = now

        if g >= best:
            cutted+=1
            continue

        sk = state_key(idx)
        if sk in visited and visited[sk] <= g:
            cutted+=1
            continue
        visited[sk] = g

        if all(idx[i] == len(auftraege[i]) for i in range(anzahl_auftraege)):
            runtime = now - start_time
            print(f">>> [Lösung] Makespan={ms} (bisher Optimum={best}) nach {runtime:.1f}s, {nodes} Knoten, Cutted={cutted}")
            if ms < best:
                best, plan = ms, p
                cutted+=1
            continue
        
        # alle möglichen nächsten Operationen sammeln
        candidates = []

        for a in range(anzahl_auftraege):
            if idx[a] < len(auftraege[a]):
                m, d = auftraege[a][idx[a]]
                s = max(za[a], zm[m])
                candidates.append((a, m, d, s))

        # nur Suchordnung: kleinste Startzeit zuerst
        candidates.sort(key=lambda x: (x[3], -x[2]))

        for (a, m, d, s) in reversed(candidates):

                m, d = auftraege[a][idx[a]]
                s = max(za[a], zm[m])
                za2, zm2, idx2 = za[:], zm[:], idx[:]
                za2[a] = zm2[m] = s + d
                idx2[a] += 1
                ms2 = max(ms, s + d)

                ra = max(
                    za2[j] + restzeit_job[j][idx2[j]]
                    for j in range(anzahl_auftraege)
                )
                rm = max(zm2[i] + sum(dd for j in range(anzahl_auftraege)
                                      for mi, dd in auftraege[j][idx2[j]:] if mi == i)
                         for i in range(anzahl_maschinen))

                g2 = max(ms2, ra, rm)
        
                if g2 < best:
                    q.append((g2, za2, zm2, idx2, p + [(a, m, s, d)], ms2))
                else:
                    cutted+=1

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
