from decimal import Decimal

from src.domain.Collection import LiveJobCollection
from src.domain.Initializer import ExperimentInitializer
from src.domain.Query import JobQuery, MachineInstanceQuery, ExperimentQuery
from src.simulation.LognormalFactorGenerator import LognormalFactorGenerator
from src.simulation.ProductionSimulation import ProductionSimulation
from src.BnB_Experiment_Runner import run_experiment


def run_experiment_internal(
    experiment_id, shift_length: int, total_shift_number: int,
    source_name: str, max_bottleneck_utilization: Decimal, sim_sigma: float):

    # Preparation  ----------------------------------------------------------------------------------
    simulation = ProductionSimulation(verbose=False)

    # Jobs Collection
    jobs = JobQuery.get_by_source_name_max_util_and_lt_arrival(
        source_name=source_name,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}"),
        arrival_limit=60 * 24 * total_shift_number
    )
    jobs_collection = LiveJobCollection(jobs)

    # Machines with transition times
    machines_instances = MachineInstanceQuery.get_by_source_name_and_max_bottleneck_utilization(
        source_name=source_name,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization}"),
    )

    # Add transition times to operations
    for machine_instance in machines_instances:
        for job in jobs_collection.values():
            for operation in job.operations:
                if operation.machine_name == machine_instance.name:
                    operation.transition_time = machine_instance.transition_time

    # Add simulation durations to operations
    factor_gen = LognormalFactorGenerator(
        sigma=sim_sigma,
        seed=42
    )
    jobs_collection.sort_jobs_by_id()
    jobs_collection.sort_operations()
    for job in jobs_collection.values():
        for operation in job.operations:
            sim_duration_float = operation.duration * factor_gen.sample()
            operation.sim_duration = int(sim_duration_float)

    # Collections(empty)
    schedule_jobs_collection = LiveJobCollection()  # pseudo previous schedule
    active_job_ops_collection = LiveJobCollection()

    waiting_job_ops_collection = LiveJobCollection()

    # Shifts ----------------------------------------------------------------------------------------
    for shift_number in range(1, total_shift_number + 1):
        shift_start = shift_number * shift_length
        shift_end = (shift_number + 1) * shift_length
        print(f"Experiment {experiment_id} shift {shift_number}: {shift_start} to {shift_end}")

        new_jobs_collection = jobs_collection.get_subset_by_earliest_start(earliest_start=shift_start)
        current_jobs_collection = new_jobs_collection + waiting_job_ops_collection

        # Scheduling --------------------------------------------------------------
        from src.solvers.BnB_Solver import BnB_Solver
        solver = BnB_Solver(
            jobs_collection=current_jobs_collection,
            logger=None,  # No logger for simplicity
            schedule_start=shift_start
        )

        solver.build_model__absolute_lateness__start_deviation__minimization(
            previous_schedule_jobs_collection=schedule_jobs_collection,
            active_jobs_collection=active_job_ops_collection,
            w_t=1, w_e=1, w_dev=1
        )

        solver.solve_model(time_limit=60*60)  # 1 hour limit

        schedule_jobs_collection = solver.get_schedule()

        ExperimentQuery.save_schedule_jobs(
            experiment_id=experiment_id,
            shift_number=shift_number,
            live_jobs=schedule_jobs_collection.values(),
        )

        # Simulation --------------------------------------------------------------
        simulation.run(
            schedule_collection=schedule_jobs_collection,
            start_time=shift_start,
            end_time=shift_end
        )

        active_job_ops_collection = simulation.get_active_operation_collection()
        waiting_job_ops_collection = simulation.get_waiting_operation_collection()

    # Save entire Simulation -------------------------------------------------------
    entire_simulation_jobs = simulation.get_entire_finished_operation_collection()
    ExperimentQuery.save_simulation_jobs(
        experiment_id=experiment_id,
        live_jobs=entire_simulation_jobs.values(),
    )


def init_experiment(shift_length: int, total_shift_number: int,
    source_name: str, max_bottleneck_utilization: Decimal, sim_sigma: float):

    experiment_id = ExperimentInitializer.insert_experiment(
        source_name=source_name,
        absolute_lateness_ratio=0,
        inner_tardiness_ratio=0,
        max_bottleneck_utilization=Decimal(f"{max_bottleneck_utilization:.2f}"),
        sim_sigma=sim_sigma,
        experiment_type="BnB",
    )
    run_experiment(
        experiment_id=experiment_id,
        shift_length=shift_length,
        total_shift_number=total_shift_number,
        logger=None,  # Simplified
        time_limit=60*60,
    )