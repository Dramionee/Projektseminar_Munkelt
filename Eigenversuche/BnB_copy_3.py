# tiefensuche, rockt 6x6 bricht sich das genick bei 10x10

import heapq
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

def carlier_pinson_bound(za, zm, idx):
    
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
    heapq.heappush(q, (0, za, zm, idx, [], 0))
    #q.append((0, za, zm, idx, [], 0))

    start_time = time.time()
    last_status = start_time
    nodes = 0

    def state_key(idx, zm):
        return tuple(idx), tuple(zm)

    while q:
        g, za, zm, idx, p, ms = heapq.heappop(q)
        #g, za, zm, idx, p, ms = q.pop()
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
        
        for a in range(anzahl_auftraege):
            if idx[a] < len(auftraege[a]):
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
                heapq.heappush(q, (g2, za2, zm2, idx2, p+[(a,m,s,d)], ms2))
                #q.append((g2, za2, zm2, idx2, p + [(a, m, s, d)], ms2))

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
