
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.project_config import get_data_path, PROJECT_ROOT
from src.domain.Query import ExperimentAnalysisQuery


# max_bottleneck_utilization_list = [0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

max_bottleneck_utilization_list = [0.75, 0.85]

for max_utilization in max_bottleneck_utilization_list:

    #sub_directory = "2025_09_08_limit"
    sub_directory = None  # Use root DB
    experiments_file_path = get_data_path(
        sub_directory,
        f"experiments_{max_utilization:.2f}".replace(".", "_")
    )

    schedules_file_path = get_data_path(
        sub_directory,
        f"schedules_{max_utilization:.2f}".replace(".", "_")
    )

    # Use root DB
    db_path = str(PROJECT_ROOT / "experiments.db")

    df_experiments = ExperimentAnalysisQuery.get_experiments_dataframe(
        max_bottleneck_utilization= max_utilization,
        db_path= db_path
    )
    df_experiments.to_csv(f"{experiments_file_path}.csv", index=False)


    df_schedules = ExperimentAnalysisQuery.get_schedule_jobs_operations_dataframe(
        max_bottleneck_utilization= max_utilization,
        db_path=db_path
    )
    df_schedules.to_csv(f"{schedules_file_path}.csv", index=False)
    