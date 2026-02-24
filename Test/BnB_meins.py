import heapq
import matplotlib.pyplot as plt

auftraege = [[(0,2),(1,5),(2,4)], [(1,2),(2,3),(0,5)], [(2,4),(0,2),(1,3)]]

"""
auftraege = [
[(2,1), (0,3), (1,6), (3,7), (2,3), (4,6)],
[(1,8), (2,5), (4,10), (5,10), (0,10), (3,4)],
[(2,5), (3,4), (5,8), (0,9), (1,1), (4,7)],
[(1,5), (0,5), (2,5), (3,3), (4,8), (5,9)],
[(0,9), (3,3), (4,5), (5,4), (2,3), (1,1)],
[(2,3), (3,3), (4,9), (1,10), (0,4), (5,1)]
]
"""

"""
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
"""

anzahl_auftraege = len(auftraege)
anzahl_maschinen = 1 + max(m for job in auftraege for m,_ in job)
maschinenfarben = ['tab:blue','tab:orange','tab:green']

def branch_and_bound():
    za = [0]*anzahl_auftraege # liste, wann jeder auftrag fertig ist (fließend aktualisiert)
    zm = [0]*anzahl_maschinen # liste, wann jede maschiene fertig ist (fließend aktualisiert)
    idx = [0]*anzahl_auftraege # liste, welche operation pro auftrag als nächstes kommt
    best = float('inf') # beste makespan, anfangs unendlich
    plan = [] # bester fertiger plan wird hier gespeichert
    q = [(0, za, zm, idx, [], 0)]
    # Warteschlange mit allen offenen Suchalternativen (Teilpläne).
    # Jeder Eintrag enthält:
    # (grenze, Auftragzeiten, Maschinenzeiten, aktuelle Operationsindizes, bisheriger Ablauf/Plan, aktuelles Makespan)
    # Hier werden bei jedem Suchschritt alle Zweige abgelegt und nacheinander abgearbeitet.
    def pfad(p): return " > ".join(f"A{a+1}-M{m+1}@{s}" for a,m,s,d in p) # formatierung der ausgabe
    while q:
        g, za, zm, idx, p, ms = heapq.heappop(q) # nimmt das mit der geringsten grenze, also makespan, bei gleicher grenze das letzte eingetragene
        #g, za, zm, idx, p, ms = q.pop()
        if g >= best: # g = untere schranke
            print(f"[Prune] {pfad(p)} | Grenze={g} >= Optimum={best}")
            continue
        if all(idx[i]==len(auftraege[i]) for i in range(anzahl_auftraege)): # schaut aufträge, ob alle operationen eingeplant
            print(f"[Lösung] {pfad(p)} | Makespan={ms}") # wenn ja gib die lösung aus
            print(f"---------------------------------------------------------------------------------------------------")
            if ms < best: best, plan = ms, p # wenn plan besser als aktuell bestes, speicher makespan und plan
            continue
        for a in range(anzahl_auftraege):   # für welchen auftrag
            if idx[a]<len(auftraege[a]):    # gibt es noch eine operation zu bearbeiten
                m,d = auftraege[a][idx[a]]  # holt maschiene und dauer der nächsten operation
                s = max(za[a], zm[m])       # start = wenn beides frei ist
                za2, zm2, idx2 = za[:], zm[:], idx[:]   # i know china, very well
                za2[a] = zm2[m] = s+d       # ende der operation bestimmen
                idx2[a] += 1                # nächste operation
                ms2 = max(ms, s+d)          # neue makespan nach der operation
                # max dauer aufträge (optimistische Fertigstellung aller Jobs)
                ra = max(za2[j]+sum(dd for _,dd in auftraege[j][idx2[j]:]) for j in range(anzahl_auftraege))
                # max dauer maschinen (optimistische Fertigstellung aller Maschinen)    
                rm = max(zm2[i]+sum(dd for j in range(anzahl_auftraege) for mi,dd in auftraege[j][idx2[j]:] if mi==i) for i in range(anzahl_maschinen))
                g2 = max(ms2, ra, rm)   # maximale dauer nach der neuen operation (Bound für diesen Ast)
                print(f"[Expand] {pfad(p)} + A{a+1}-M{m+1}@{s} | Grenze={g2}")
                heapq.heappush(q, (g2, za2, zm2, idx2, p+[(a,m,s,d)], ms2)) # legt den neuen ast in die warteschleife
                #g, za, zm, idx, p, ms = q.pop()
    return plan, best

if __name__=="__main__":
    plan, makespan = branch_and_bound()
    print("\n[Optimum]:")
    for a, m, s, d in plan:
        print(f"  A{a+1} M{m+1} [{s}-{s+d}]")
    fig, ax = plt.subplots() # erstellt figur
    for a, m, s, d in plan:
        ax.broken_barh([(s, d)], (a*10, 9), facecolors=maschinenfarben[m]) # zeichnet horizontale balken
        ax.text(s+d/2, a*10+4.5, f"M{m+1}", ha='center', va='center', color='white') # schreibt text da rein
    ax.set_yticks([a*10+4.5 for a in range(anzahl_auftraege)]) # höhe y achse
    ax.set_yticklabels([f"A{a+1}" for a in range(anzahl_auftraege)]) # text auf y achse
    ax.set_xlabel('Zeit') # x achse
    ax.set_title(f"Makespan = {makespan}")
    plt.tight_layout()
    plt.show()
