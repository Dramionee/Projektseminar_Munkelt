# tiefensuche, aber nur kleinste startzeit

import time
import matplotlib.pyplot as plt

auftraege = [
[(2,1), (0,3), (1,6), (3,7), (2,3), (4,6)],
[(1,8), (2,5), (4,10), (5,10), (0,10), (3,4)],
[(2,5), (3,4), (5,8), (0,9), (1,1), (4,7)],
[(1,5), (0,5), (2,5), (3,3), (4,8), (5,9)],
[(0,9), (3,3), (4,5), (5,4), (2,3), (1,1)],
[(2,3), (3,3), (4,9), (1,10), (0,4), (5,1)]
]

anzahl_auftraege = len(auftraege)
anzahl_maschinen = 1 + max(m for job in auftraege for m, _ in job)
maschinenfarben = ['tab:blue','tab:orange','tab:green','tab:red','tab:olive','tab:purple','tab:grey','tab:cyan']

def branch_and_bound_with_memo_and_progress():
    za = [0]*anzahl_auftraege
    zm = [0]*anzahl_maschinen
    idx = [0]*anzahl_auftraege

    best = float('inf')
    plan = []
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
        
        # sortierung nächste operationen nach kleinster startzeit und nimmt dann nur die zum weiter branchen
        candidates = []
        min_s = float('9999')

        for a in range(anzahl_auftraege):
            if idx[a] < len(auftraege[a]):
                m, d = auftraege[a][idx[a]]
                s = max(za[a], zm[m])
                if s < min_s:
                    candidates = [(a, m, d, s)]
                    min_s = s
                elif s == min_s:
                    candidates.append((a, m, d, s))

        for (a, m, d, s) in candidates:

                m, d = auftraege[a][idx[a]]
                s = max(za[a], zm[m])
                za2, zm2, idx2 = za[:], zm[:], idx[:]
                za2[a] = zm2[m] = s + d
                idx2[a] += 1
                ms2 = max(ms, s + d)

                ra = max(za2[j] + sum(dd for _, dd in auftraege[j][idx2[j]:]) for j in range(anzahl_auftraege))
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
