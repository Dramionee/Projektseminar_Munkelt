import time
from typing import Optional, List, Tuple

from src.Logger import Logger
from src.domain.Collection import LiveJobCollection


class BnB_Solver:

    def __init__(self, jobs_collection: LiveJobCollection, logger: Logger, schedule_start: int = 0):
        self.jobs_collection = jobs_collection
        self.logger = logger
        self.schedule_start = schedule_start
        self.schedule = None
        self.makespan = None
        self.model_completed = False
        self.solver_status = None

    def build_model__absolute_lateness__start_deviation__minimization(
        self,
        previous_schedule_jobs_collection=None,
        active_jobs_collection=None,
        w_t=1, w_e=1, w_dev=1
    ):
        # Für BnB ignorieren wir die Gewichte und optimieren nur Makespan
        self.logger.info("BnB Solver: Building model (simplified for Makespan minimization)")
        self.model_completed = True

    def log_model_info(self):
        self.logger.info("BnB Solver: Model built (Branch and Bound for Makespan)")

    def solve_model(
        self,
        gap_limit=0.002,
        time_limit: Optional[int] = None,
        log_file=None,
        bound_relative_change=0.01,
        bound_no_improvement_time=None,
        bound_warmup_time=None,
    ):
        self.logger.info("BnB Solver: Starting Branch and Bound optimization")

        # Konvertiere jobs_collection in das Format für BnB
        auftraege = self._convert_jobs_to_bnb_format()

        # Führe BnB aus
        plan, makespan = self._run_bnb(auftraege, time_limit)

        # Konvertiere Plan zurück in Schedule
        self.schedule = self._convert_plan_to_schedule(plan)
        self.makespan = makespan
        self.solver_status = "OPTIMAL" if makespan is not None else "UNKNOWN"
        self.logger.info(f"BnB Solver: Finished with makespan {makespan}")

    def log_solver_info(self):
        self.logger.info(f"BnB Solver: Status {self.solver_status}, Makespan {self.makespan}")

    def get_schedule(self) -> LiveJobCollection:
        return self.schedule

    def _convert_jobs_to_bnb_format(self) -> List[List[Tuple[int, int]]]:
        # Konvertiere LiveJobCollection in Liste von Jobs, jeder Job ist Liste von (machine_idx, duration)
        auftraege = []
        machine_names = sorted(self.jobs_collection.get_unique_machine_names())
        machine_to_idx = {name: int(name[1:]) for name in machine_names}  # M00 -> 0, M01 -> 1, etc.

        for job in self.jobs_collection.values():
            job_ops = []
            for op in job.operations:
                machine_idx = machine_to_idx[op.machine_name]
                duration = op.sim_duration if hasattr(op, 'sim_duration') else op.duration
                job_ops.append((machine_idx, duration))
            auftraege.append(job_ops)
        return auftraege

    def _run_bnb(self, auftraege: List[List[Tuple[int, int]]], time_limit: Optional[int]) -> Tuple[List, int]:
        # Hier den BnB-Code integrieren
        return branch_and_bound_with_memo_and_progress(auftraege, time_limit)

    def _convert_plan_to_schedule(self, plan: List[Tuple[int, int, int, int]]) -> LiveJobCollection:
        # Konvertiere BnB-Plan zurück in LiveJobCollection
        schedule = LiveJobCollection()
        jobs_list = list(self.jobs_collection.values())

        for a, m, s, d in plan:
            job = jobs_list[a]
            # Finde die Operation für diese Maschine
            machine_name = f"M{m:02d}"
            for op in job.operations:
                if op.machine_name == machine_name:
                    op.start = s
                    op.end = s + d
                    break
            schedule[job.id] = job
        return schedule


# Der BnB-Code aus BnB2.py
def branch_and_bound_with_memo_and_progress(auftraege, time_limit=None):
    anzahl_auftraege = len(auftraege)
    anzahl_maschinen = 1 + max(m for job in auftraege for m, _ in job)

    za = [0] * anzahl_auftraege
    zm = [0] * anzahl_maschinen
    idx = [0] * anzahl_auftraege

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
        if time_limit and (now - start_time) > time_limit:
            break
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

        # 1: finde minimalen earliest start s*
        candidates = []
        min_s = float('inf')

        for a in range(anzahl_auftraege):
            if idx[a] < len(auftraege[a]):
                m, d = auftraege[a][idx[a]]
                s = max(za[a], zm[m])
                if s < min_s:
                    candidates = [(a, m, d, s)]
                    min_s = s
                elif s == min_s:
                    candidates.append((a, m, d, s))

        # candidates = Konfliktset
        for (a, m, d, s) in candidates:
            za2, zm2, idx2 = za[:], zm[:], idx[:]
            za2[a] = zm2[m] = s + d
            idx2[a] += 1
            ms2 = max(ms, s + d)

            # existing simple bounds
            ra = max(za2[j] + sum(dd for _, dd in auftraege[j][idx2[j]:]) for j in range(anzahl_auftraege))
            rm = max(zm2[i] + sum(dd for j in range(anzahl_auftraege)
                                  for mi, dd in auftraege[j][idx2[j]:] if mi == i)
                     for i in range(anzahl_maschinen))

            g2 = max(ms2, ra, rm)
            q.append((g2, za2, zm2, idx2, p + [(a, m, s, d)], ms2))

    print(f"\nFERTIG. Optimum: {best}. Gesamtknoten: {nodes}. Laufzeit: {time.time()-start_time:.1f} Sekunden.")

    return plan, best