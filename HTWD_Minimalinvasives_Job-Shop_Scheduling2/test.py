# tiefensuche

import time
import matplotlib.pyplot as plt

auftraege = [
    [(2, 31), (0, 86), (1, 46), (5, 74), (4, 32), (6, 88), (8, 19), (9, 48), (7, 36), (3, 79)],
    [(1, 46), (0, 37), (3, 61), (2, 13), (6, 32), (5, 21), (9, 32), (8, 89), (7, 30), (4, 55)],
    [(0, 43), (2, 90), (4, 75), (9, 11), (3, 69), (1, 28), (6, 46), (5, 46), (7, 72), (8, 30)],
    [(2, 84), (1, 2), (5, 52), (3, 95), (8, 48), (9, 72), (0, 47), (6, 65), (4, 6), (7, 25)],
    [(2, 14), (0, 6), (1, 22), (5, 61), (3, 26), (4, 69), (8, 21), (7, 49), (9, 72), (6, 53)],
    [(1, 91), (0, 85), (3, 39), (2, 74), (8, 90), (5, 10), (7, 12), (6, 89), (9, 45), (4, 33)],
    [(0, 29), (1, 78), (2, 9), (3, 36), (4, 49), (5, 11), (6, 62), (7, 56), (8, 44), (9, 21)],
    [(1, 81), (2, 95), (0, 71), (4, 99), (6, 9), (8, 52), (7, 85), (3, 98), (9, 22), (5, 43)],
    [(1, 85), (0, 13), (2, 61), (6, 7), (8, 64), (9, 76), (5, 47), (3, 52), (4, 90), (7, 45)],
    [(0, 76), (1, 69), (3, 76), (5, 51), (2, 85), (9, 11), (6, 40), (7, 89), (4, 26), (8, 74)]
]

anzahl_auftraege = len(auftraege)
anzahl_maschinen = 1 + max(m for job in auftraege for m, _ in job)
maschinenfarben = ['tab:blue','tab:orange','tab:green','tab:red','tab:olive','tab:purple','tab:grey','tab:cyan']

def carlier_pinson_bound(za, zm, idx):
    """
    Praktische CP-ähnliche Bound-Implementierung.
    - berechnet Head (h) und Tail (t) für alle verbleibenden Operationen
    - für jede Maschine simuliert zwei optimistische Best-Case-Planer:
        * Greedy nach Head
        * Verfügbarkeits-SPT (immer kürzeste verfügbare o. wählen)
    - wählt pro Maschine das optimistischste Ergebnis (kleinster Abschluss)
    - return: LB (float)
    """
    # 1) prepare heads and tails for remaining operations
    # heads[j][k] for k >= idx[j] = earliest start of op (j,k) within its job given current za[j]
    # tails[j][k] = sum of durations after k in job j
    heads = {}
    tails = {}
    for j in range(anzahl_auftraege):
        # accumulate earliest starts for remaining ops in job j
        cur = za[j]
        for k in range(idx[j], len(auftraege[j])):
            heads[(j, k)] = cur
            cur += auftraege[j][k][1]
        # tails: from end backwards (0 for last op)
        tot_after = 0
        for k in range(len(auftraege[j]) - 1, -1, -1):
            if k >= idx[j]:
                tails[(j, k)] = tot_after
            else:
                # operations already done have no tail relevance here; set 0
                tails[(j, k)] = 0
            tot_after += auftraege[j][k][1]

    # 2) aggregate remaining ops per machine
    mas_ops = {m: [] for m in range(anzahl_maschinen)}
    for j in range(anzahl_auftraege):
        for k in range(idx[j], len(auftraege[j])):
            m, p = auftraege[j][k]
            h = heads[(j, k)]
            t = tails[(j, k)]
            mas_ops[m].append({'j': j, 'k': k, 'p': p, 'h': h, 't': t})

    # helper: simulate schedule by given ordering (list of ops)
    def simulate_ordered_schedule(ops_list, start_time):
        """Simulate linear scheduling of ops_list (already ordered).
        Each op is dict with keys p,h,t. Returns dict of finishes { (j,k): finish } and max_finish_plus_tail.
        """
        time_ptr = start_time
        finishes = {}
        max_ft_plus_tail = -10**15
        for op in ops_list:
            start = max(time_ptr, op['h'])
            finish = start + op['p']
            finishes[(op['j'], op['k'])] = finish
            time_ptr = finish
            max_ft_plus_tail = max(max_ft_plus_tail, finish + op['t'])
        return finishes, max_ft_plus_tail

    # helper: simulate SPT among available ops (respect heads)
    def simulate_spt_by_availability(ops_list, start_time):
        """Simulate schedule choosing shortest processing time among currently available ops.
        Returns finishes and max_finish_plus_tail (optimistic).
        """
        # copy and sort by head for fast access
        remaining = sorted(ops_list, key=lambda o: o['h'])
        time_ptr = start_time
        finishes = {}
        max_ft_plus_tail = -10**15
        import heapq
        avail = []  # heap by processing time
        i = 0
        L = len(remaining)
        while i < L or avail:
            while i < L and remaining[i]['h'] <= time_ptr:
                heapq.heappush(avail, (remaining[i]['p'], i, remaining[i]))
                i += 1
            if not avail:
                # no available op, advance to next head
                time_ptr = remaining[i]['h']
                continue
            p, _, op = heapq.heappop(avail)
            start = max(time_ptr, op['h'])
            finish = start + op['p']
            finishes[(op['j'], op['k'])] = finish
            time_ptr = finish
            max_ft_plus_tail = max(max_ft_plus_tail, finish + op['t'])
        return finishes, max_ft_plus_tail

    # 3) for each machine compute optimistic completion estimate
    lb_per_machine = []
    for m in range(anzahl_maschinen):
        ops = mas_ops[m]
        if not ops:
            # no remaining ops on this machine -> its contribution is its current zm[m]
            lb_per_machine.append(zm[m])
            continue

        # Option A: order by head (non-decreasing)
        by_head = sorted(ops, key=lambda o: (o['h'], -o['p']))
        _, candidate_head = simulate_ordered_schedule(by_head, zm[m])

        # Option B: SPT among available ops (greedy)
        _, candidate_spt = simulate_spt_by_availability(ops, zm[m])

        # We take the most optimistic (smallest) candidate as the machine's optimistic completion bound
        # Reason: both schedules respect earliest job heads (so are feasible wrt job precedence), and they
        # give different optimistic orderings; the minimum of both is a valid lower bound.
        machine_lb = min(candidate_head, candidate_spt)
        lb_per_machine.append(machine_lb)

    # 4) also compute a job-based simple lower bound (heads + remaining durations per job)
    lb_per_job = []
    for j in range(anzahl_auftraege):
        # earliest end of remainder of job j = head of first remaining op + sum of durations remaining
        if idx[j] < len(auftraege[j]):
            rem = sum(dd for _, dd in auftraege[j][idx[j]:])
            lb_per_job.append(za[j] + rem)
        else:
            lb_per_job.append(za[j])

    # final LB is maximum of current partial makespan (implicit), per-machine optimistic completions and per-job
    lb = max(max(lb_per_machine) if lb_per_machine else 0,
             max(lb_per_job) if lb_per_job else 0)

    return lb

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

        # sortierung nächste operationen nach kleinster startzeit
        candidates = []                      # Liste der Operationen mit minimaler frühester Startzeit
        min_s = float('9999')                # aktuell kleinste gefundene früheste Startzeit

        for a in range(anzahl_auftraege):    # gehe alle Aufträge durch
            if idx[a] < len(auftraege[a]):   # prüfe, ob Auftrag a noch nicht abgeschlossen ist
                m, d = auftraege[a][idx[a]]  # nächste Operation von Auftrag a: Maschine m, Dauer d
                s = max(za[a], zm[m])        # frühester Start: Auftrag und Maschine müssen frei sein
                if s < min_s:                # falls diese Operation früher starten kann als alle bisherigen
                    candidates = [(a, m, d, s)]  # neue Kandidatenliste nur mit dieser Operation
                    min_s = s                # aktualisiere die kleinste Startzeit
                elif s == min_s:             # falls diese Operation gleich früh starten kann
                    candidates.append((a, m, d, s))  # füge sie zum Konfliktset hinzu

        for (a, m, d, s) in candidates:      # verzweige über alle Operationen mit minimaler Startzeit


                m, d = auftraege[a][idx[a]]
                s = max(za[a], zm[m])
                za2, zm2, idx2 = za[:], zm[:], idx[:]
                za2[a] = zm2[m] = s + d
                idx2[a] += 1
                ms2 = max(ms, s + d)

                # existing simple bounds
                ra = max(za2[j] + sum(dd for _, dd in auftraege[j][idx2[j]:]) for j in range(anzahl_auftraege))
                rm = max(zm2[i] + sum(dd for j in range(anzahl_auftraege)
                                      for mi, dd in auftraege[j][idx2[j]:] if mi == i)
                         for i in range(anzahl_maschinen))

                # new Carlier & Pinson like bound
                lb_cp = carlier_pinson_bound(za2, zm2, idx2)

                g2 = max(ms2, ra, rm, lb_cp)
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
