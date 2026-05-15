"""Time aggregation helpers for backtesting engines (array-first)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def compute_time_metrics(
    dt_list: Sequence,
    day_list: Sequence,
    num_observations: int,
    annual_days: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute time-based annualization metrics from datetime/date indices.

    This matches the legacy dictionary keys and (when possible) value types.
    """
    num_dates = len(day_list)

    if dt_list:
        datetime_min = dt_list[0]
        datetime_max = dt_list[-1]
        time_diff = datetime_max - datetime_min
        num_years = max(
            time_diff.days / 365.25 + time_diff.seconds / (365.25 * 24 * 3600),
            1 / 365.25,
        )
    else:
        num_years = 1

    dates_to_years = annual_days if annual_days else num_dates / num_years
    return {
        "num_dates": num_dates,
        "num_years": num_years,
        "dates_to_years": dates_to_years,
        "num_observations": int(num_observations),
        "observations_to_years": num_observations / num_years if num_years else 0.0,
    }

