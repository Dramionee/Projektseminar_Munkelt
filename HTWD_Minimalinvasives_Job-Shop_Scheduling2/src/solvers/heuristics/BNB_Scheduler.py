from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List

from src.domain.Collection import LiveJobCollection

# Achtung: pyscipopt muss bei euch im Environment vorhanden sein
from pyscipopt import Model


class Scheduler:
    def __init__(
        self,
        jobs_collection: LiveJobCollection,
        schedule_start: int,
        previous_schedule_jobs_collection: Optional[LiveJobCollection] = None,
        active_jobs_collection: Optional[LiveJobCollection] = None,
        bnb_params: Optional[Dict[str, Any]] = None,
    ):
        self.jobs_collection = jobs_collection
        self.schedule_start = schedule_start
        self.previous_schedule_jobs_collection = previous_schedule_jobs_collection or LiveJobCollection()
        self.active_jobs_collection = active_jobs_collection or LiveJobCollection()
        self.bnb_params = bnb_params or {}

    def get_schedule(self) -> LiveJobCollection:
        jobs = list(self.jobs_collection.values())
        self.jobs_collection.sort_jobs_by_id()
        self.jobs_collection.sort_operations()

        # ops: (job_obj, op_index, machine_name, proc_time, op_obj)
        ops: List[Tuple[Any, int, str, int, Any]] = []
        for job in self.jobs_collection.values():
            for k, op in enumerate(job.operations):
                p = int(getattr(op, "duration", 0))
                m = str(op.machine_name)
                ops.append((job, k, m, p, op))

        machines = sorted({m for _, _, m, _, _ in ops})
        M = sum(p for _, _, _, p, _ in ops)

        model = Model("JobShop_BNB")

        # --- FIX: eindeutige int-Indizes statt job.id casten ---
        jobs_ordered = list(self.jobs_collection.values())
        jobs_ordered.sort(key=lambda j: str(getattr(j, "id", "")))

        # LiveJob ist unhashable -> wir nutzen die string-id als Key
        job_idx_by_id = {str(job.id): idx for idx, job in enumerate(jobs_ordered)}

        # ------------------------------------------------------

        # Startvariablen S[j,k]
        S: Dict[Tuple[int, int], Any] = {}
        for job, k, m, p, op in ops:
            j = job_idx_by_id[str(job.id)]

            S[(j, k)] = model.addVar(lb=float(self.schedule_start), vtype="C", name=f"S_{j}_{k}")

        Cmax = model.addVar(lb=0.0, vtype="C", name="Cmax")

        # Job-Precedence
        for job in self.jobs_collection.values():
            j = job_idx_by_id[str(job.id)]

            for k in range(len(job.operations) - 1):
                p = int(getattr(job.operations[k], "duration", 0))
                model.addCons(S[(j, k + 1)] >= S[(j, k)] + p)

        # Maschinenkonflikte
        for m in machines:
            ops_m = [(job_idx_by_id[str(job.id)], k, p) for (job, k, mm, p, _) in ops if mm == m]
            for i in range(len(ops_m)):
                for j in range(i + 1, len(ops_m)):
                    j1, k1, p1 = ops_m[i]
                    j2, k2, p2 = ops_m[j]
                    y = model.addVar(vtype="B", name=f"y_{j1}_{k1}_{j2}_{k2}")

                    model.addCons(S[(j1, k1)] + p1 <= S[(j2, k2)] + M * (1 - y))
                    model.addCons(S[(j2, k2)] + p2 <= S[(j1, k1)] + M * y)

        # Makespan
        for job, k, m, p, op in ops:
            j = job_idx_by_id[str(job.id)]
            model.addCons(Cmax >= S[(j, k)] + p)

        model.setObjective(Cmax, "minimize")

        time_limit = self.bnb_params.get("time_limit", None)
        if time_limit is not None:
            model.setParam("limits/time", float(time_limit))

        model.optimize()

        # Lösung zurück
        schedule_job_collection = LiveJobCollection()

        op_by_key: Dict[Tuple[int, int], Any] = {}
        dur_by_key: Dict[Tuple[int, int], int] = {}

        for job, k, m, p, op in ops:
            j = job_idx_by_id[str(job.id)]
            op_by_key[(j, k)] = op
            dur_by_key[(j, k)] = p

        for (j, k), var in S.items():
            start = float(model.getVal(var))
            p = dur_by_key[(j, k)]
            end = start + p
            schedule_job_collection.add_operation_instance(
                op=op_by_key[(j, k)],
                new_start=int(round(start)),
                new_end=int(round(end)),
            )

        return schedule_job_collection

