from __future__ import annotations

import argparse
import math


Z_SCORES = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}


def safety_stock(error_sigma: float, service_level: float, lead_time_days: int) -> float:
    if service_level not in Z_SCORES:
        raise ValueError(f"Unsupported service level {service_level}. Use one of {sorted(Z_SCORES)}.")
    if error_sigma < 0:
        raise ValueError("error_sigma must be non-negative.")
    if lead_time_days <= 0:
        raise ValueError("lead_time_days must be positive.")
    return Z_SCORES[service_level] * error_sigma * math.sqrt(lead_time_days)


def reorder_point(average_daily_demand: float, error_sigma: float, service_level: float, lead_time_days: int) -> float:
    if average_daily_demand < 0:
        raise ValueError("average_daily_demand must be non-negative.")
    return average_daily_demand * lead_time_days + safety_stock(error_sigma, service_level, lead_time_days)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--average-daily-demand", type=float, required=True)
    parser.add_argument("--error-sigma", type=float, required=True)
    parser.add_argument("--service-level", type=float, choices=sorted(Z_SCORES), default=0.95)
    parser.add_argument("--lead-time-days", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ss = safety_stock(args.error_sigma, args.service_level, args.lead_time_days)
    rop = reorder_point(args.average_daily_demand, args.error_sigma, args.service_level, args.lead_time_days)
    print(f"safety_stock_units={ss:.2f}")
    print(f"reorder_point_units={rop:.2f}")
