from __future__ import annotations

import argparse
from decimal import Decimal
from typing import Any, Dict

from src.domain.Initializer import ExperimentInitializer
from src.BNB_Experiment_Runner1 import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONE single BNB experiment (no grid).")

    # Experiment-Parameter (wie du sie bisher genutzt hast)
    parser.add_argument("--util", type=float, required=True, help="max_bottleneck_utilization, z.B. 0.75")
    parser.add_argument("--sigma", type=float, required=True, help="sim_sigma, z.B. 0.1")
    parser.add_argument("--mip_gap", type=float, default=None)

    # Lauf-Parameter
    parser.add_argument("--shift_length", type=int, default=1440)
    parser.add_argument("--total_shift_number", type=int, default=20)
    parser.add_argument("--source_name", type=str, default="Fisher and Thompson 10x10")

    # BnB/SCIP-Parameter (anlehnend an dein framework copy 2.py) :contentReference[oaicite:1]{index=1}
    parser.add_argument("--time_limit", type=int, default=None, help="SCIP time limit per shift (seconds)")
    parser.add_argument("--scip_threads", type=int, default=1)
    parser.add_argument("--perm_seed", type=int, default=12345)
    parser.add_argument("--rand_seed_shift", type=int, default=12345)

    args = parser.parse_args()

    # 1) Experiment in DB anlegen (genau eins)
    experiment_id = ExperimentInitializer.insert_experiment(
        source_name=args.source_name,
        absolute_lateness_ratio=0,
        inner_tardiness_ratio=0,
        max_bottleneck_utilization=Decimal(f"{args.util:.2f}"),
        sim_sigma=float(args.sigma),
        experiment_type="BNB_SINGLE",
    )

    bnb_params = {
        "time_limit": args.time_limit,
        "scip_threads": args.scip_threads,
        "mip_gap": args.mip_gap,
        "perm_seed": args.perm_seed,
        "rand_seed_shift": args.rand_seed_shift,
    }

    # 2) Experiment laufen lassen (genau eins)
    run_experiment(
        experiment_id=experiment_id,
        shift_length=int(args.shift_length),
        total_shift_number=int(args.total_shift_number),
        source_name=str(args.source_name),
        max_bottleneck_utilization=Decimal(f"{args.util:.2f}"),
        sim_sigma=float(args.sigma),
        bnb_params=bnb_params,
    )


if __name__ == "__main__":
    """
    Beispiel:
    python run_single_bnb_experiment.py --util 0.75 --sigma 0.1 --time_limit 1800
    """
    main()
