from __future__ import annotations

from decimal import Decimal
from typing import Dict, Any

from src.domain.Collection import LiveJobCollection
from src.domain.Query import JobQuery, MachineInstanceQuery, ExperimentQuery
from src.simulation.LognormalFactorGenerator import LognormalFactorGenerator
from src.simulation.ProductionSimulation import ProductionSimulation

from src.solvers.heuristics.BNB_Scheduler import Scheduler as BNBScheduler


def run_experiment(
    experiment_id: int,
    shift_length: int,
    total_shift_number: int,
    source_name: str,
    max_bottleneck_utilization: Decimal,
    sim_sigma: float,
    bnb_params: Dict[str, Any],
) -> None:
    """
    Führt EIN Experiment (experiment_id) über total_shift_number Shifts aus.
    """
    simulation = ProductionSimulation(verbose=False)

    jobs = JobQuery.get_by_source_name_max_util_and_lt_arrival(
        source_name=source_name,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}"),
        arrival_limit=60 * 24 * total_shift_number,
    )
    jobs_collection = LiveJobCollection(jobs)

    machines_instances = MachineInstanceQuery.get_by_source_name_and_max_bottleneck_utilization(
        source_name=source_name,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}"),
    )

    # Transition times setzen
    for machine_instance in machines_instances:
        for job in jobs_collection.values():
            for operation in job.operations:
                if operation.machine_name == machine_instance.name:
                    operation.transition_time = machine_instance.transition_time

    # Simulations-Dauern samplen
    factor_gen = LognormalFactorGenerator(sigma=sim_sigma, seed=42)
    jobs_collection.sort_jobs_by_id()
    jobs_collection.sort_operations()
    for job in jobs_collection.values():
        for operation in job.operations:
            operation.sim_duration = int(operation.duration * factor_gen.sample())

    schedule_jobs_collection = LiveJobCollection()
    active_job_ops_collection = LiveJobCollection()
    waiting_job_ops_collection = LiveJobCollection()

    for shift_number in range(1, total_shift_number + 1):
        shift_start = shift_number * shift_length
        shift_end = (shift_number + 1) * shift_length
        print(f"Experiment {experiment_id} shift {shift_number}: {shift_start} to {shift_end}")

        new_jobs_collection = jobs_collection.get_subset_by_earliest_start(earliest_start=shift_start)
        current_jobs_collection = new_jobs_collection + waiting_job_ops_collection
        print("active jobs this shift:", len(current_jobs_collection.values()))
        print("job ids:", [str(j.id) for j in current_jobs_collection.values()])

        scheduler = BNBScheduler(
            jobs_collection=current_jobs_collection,
            schedule_start=shift_start,
            previous_schedule_jobs_collection=schedule_jobs_collection,
            active_jobs_collection=active_job_ops_collection,
            bnb_params=bnb_params,
        )

        schedule_jobs_collection = scheduler.get_schedule()

        ExperimentQuery.save_schedule_jobs(
            experiment_id=experiment_id,
            shift_number=shift_number,
            live_jobs=schedule_jobs_collection.values(),
        )

        simulation.run(
            schedule_collection=schedule_jobs_collection,
            start_time=shift_start,
            end_time=shift_end,
        )

        active_job_ops_collection = simulation.get_active_operation_collection()
        waiting_job_ops_collection = simulation.get_waiting_operation_collection()

    entire_simulation_jobs = simulation.get_entire_finished_operation_collection()
    ExperimentQuery.save_simulation_jobs(
        experiment_id=experiment_id,
        live_jobs=entire_simulation_jobs.values(),
    )
