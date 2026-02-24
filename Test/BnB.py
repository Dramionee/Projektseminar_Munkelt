#schafft nichtmal 3x3 Problem
import heapq
import matplotlib.pyplot as plt
import time, tracemalloc

# 12
auftraege = [[(0,2),(1,5),(2,4)], [(1,2),(2,3),(0,5)], [(2,4),(0,2),(1,3)]]

# ft06 - 55

'''auftraege = [
    [(2,1), (0,3), (1,6), (3,7), (2,3), (4,6)],
    [(1,8), (2,5), (4,10), (5,10), (0,10), (3,4)],
    [(2,5), (3,4), (5,8), (0,9), (1,1), (4,7)],
    [(1,5), (0,5), (2,5), (3,3), (4,8), (5,9)],
    [(0,9), (3,3), (4,5), (5,4), (2,3), (1,1)],
    [(2,3), (3,3), (4,9), (1,10), (0,4), (5,1)]
]'''

# abz6 - 943
"""
auftraege = [
    [(4,62), (5,24), (2,25), (1,84), (0,47), (7,38), (3,82), (8,93), (9,24), (6,66)],
    [(5,47), (2,97), (8,92), (9,22), (1,93), (4,29), (7,56), (3,80), (0,78), (6,67)],
    [(1,28), (0,12), (7,94), (5,49), (3,81), (6,52), (4,59), (9,21), (2,57), (8,39)],
    [(2,87), (6,77), (4,39), (1,96), (7,90), (5,95), (0,70), (8,99), (3,88), (9,21)],
    [(6,15), (4,77), (7,28), (2,45), (9,23), (1,78), (5,35), (8,75), (0,12), (3,52)],
    [(4,74), (6,88), (3,52), (9,27), (2,9), (8,52), (1,56), (5,48), (7,82), (0,62)],
    [(3,93), (4,92), (7,91), (8,30), (1,56), (0,27), (6,66), (9,48), (5,63), (2,98)],
    [(9,87), (6,29), (5,96), (2,26), (4,15), (3,71), (0,52), (8,78), (7,67), (1,87)],
    [(1,63), (4,7), (5,44), (9,79), (0,24), (6,78), (3,84), (8,88), (2,71), (7,25)],
    [(8,6), (0,48), (4,9), (6,54), (3,63), (7,69), (9,72), (5,61), (2,96), (1,35)]
]
"""

anzahl_auftraege = len(auftraege)
anzahl_maschinen = 1 + max(m for job in auftraege for m,_ in job)
maschinenfarben = ['tab:blue','tab:orange','tab:green']

def branch_and_bound():
    za = [0]*anzahl_auftraege
    zm = [0]*anzahl_maschinen
    idx = [0]*anzahl_auftraege
    best = float('inf')
    plan = []
    
    start = time.time()  # Start-Zeit merken
    timeout = 600
    
    q = [(0, za, zm, idx, [], 0)]
    def pfad(p): return " > ".join(f"A{a+1}-M{m+1}@{s}" for a,m,s,d in p)
    
    while q:
        g, za, zm, idx, p, ms = q.pop()  # LIFO nimmt einfach das letzte
        #g, za, zm, idx, p, ms = heapq.heappop(q) # breitensuche
        #print(f"===========================")
        
        if g >= best:   # bessere Lösung schon gefunden
            #print(f"[Cut] {pfad(p)} | Grenze={g} >= Optimum={best}")
            continue
            
        if all(idx[i]==len(auftraege[i]) for i in range(anzahl_auftraege)): # vollständiger Plan fertig
            runtime = time.time() - start  # ← Laufzeit berechnen
            print(f"---------------------------------------------------------------------------------------------------")
            print(f"[Lösung] {pfad(p)} | Makespan={ms} | Laufzeit: {runtime:.3f}s")  # ← Laufzeit ausgeben
            if ms < best: best, plan = ms, p
            continue
            
        for a in range(anzahl_auftraege):
            if idx[a]<len(auftraege[a]):
                m,d = auftraege[a][idx[a]]
                s = max(za[a], zm[m])
                za2, zm2, idx2 = za[:], zm[:], idx[:]
                za2[a] = zm2[m] = s+d
                idx2[a] += 1
                ms2 = max(ms, s+d)
                ra = max(za2[j]+sum(dd for _,dd in auftraege[j][idx2[j]:]) for j in range(anzahl_auftraege))
                rm = max(zm2[i]+sum(dd for j in range(anzahl_auftraege) for mi,dd in auftraege[j][idx2[j]:] if mi==i) for i in range(anzahl_maschinen))
                g2 = max(ms2, ra, rm)
                #print(f"[Zweig] {pfad(p)} + A{a+1}-M{m+1}@{s} | Grenze={g2}")
                #heapq.heappush(q, (g2, za2, zm2, idx2, p+[(a,m,s,d)], ms2))
                q.append((g2, za2, zm2, idx2, p+[(a,m,s,d)], ms2))
    
    runtime = time.time() - start
    print(f"\nGesamtlaufzeit: {runtime:.3f}s")
            
    return plan, best


if __name__=="__main__":
    
    plan, makespan = branch_and_bound()
    
    
    
    print("\n[Optimum]:")
    
    for a, m, s, d in plan:
        print(f"  A{a+1} M{m+1} [{s}-{s+d}]")
    
    fig, ax = plt.subplots()
    for a, m, s, d in plan:
        ax.broken_barh([(s, d)], (a*10, 9), facecolors=maschinenfarben[m])
        ax.text(s+d/2, a*10+4.5, f"M{m+1}", ha='center', va='center', color='white')
    ax.set_yticks([a*10+4.5 for a in range(anzahl_auftraege)])
    ax.set_yticklabels([f"A{a+1}" for a in range(anzahl_auftraege)])
    ax.set_xlabel('Zeit')
    ax.set_title(f"Makespan = {makespan}")
    plt.tight_layout()
    plt.show()

