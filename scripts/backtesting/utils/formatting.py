"""Metric formatting utilities for backtesting reports."""

import pandas as pd


# Unified display name mapping for metrics
METRIC_DISPLAY_NAMES = {
    # L/S metrics
    'long_short_ret_annual': 'L/S Annual Return',
    'long_short_ret_sharpe': 'L/S Sharpe',
    'long_short_ret_max_dd': 'L/S Max Drawdown',
    'long_short_ret_calmar': 'L/S Calmar',
    'long_short_ret_sortino': 'L/S Sortino',
    'long_short_ret_sharpe_per_turnover': 'L/S Sharpe/Turnover',
    'long_short_turnover_ratio': 'L/S Turnover',
    'long_short_turnover': 'L/S Turnover',
    'long_short_win_rate': 'L/S Win Rate',
    # Long metrics
    'long_ret_annual': 'Long Annual Return',
    'long_ret_sharpe': 'Long Sharpe',
    'long_ret_max_dd': 'Long Max Drawdown',
    'long_ret_calmar': 'Long Calmar',
    'long_ret_sortino': 'Long Sortino',
    'long_ret_sharpe_per_turnover': 'Long Sharpe/Turnover',
    'long_turnover_ratio': 'Long Turnover',
    'long_turnover': 'Long Turnover',
    'long_win_rate': 'Long Win Rate',
    # Short metrics
    'short_ret_annual': 'Short Annual Return',
    'short_ret_sharpe': 'Short Sharpe',
    'short_ret_max_dd': 'Short Max Drawdown',
    'short_ret_calmar': 'Short Calmar',
    'short_ret_sortino': 'Short Sortino',
    'short_ret_sharpe_per_turnover': 'Short Sharpe/Turnover',
    'short_turnover_ratio': 'Short Turnover',
    'short_turnover': 'Short Turnover',
    'short_win_rate': 'Short Win Rate',
    # Passive metrics
    'passive_ret_annual': 'Passive Annual Return',
    'passive_ret_sharpe': 'Passive Sharpe',
    'passive_ret_max_dd': 'Passive Max Drawdown',
    'passive_ret_calmar': 'Passive Calmar',
    'passive_ret_sortino': 'Passive Sortino',
    'passive_ret_sharpe_per_turnover': 'Passive Sharpe/Turnover',
    'passive_turnover_ratio': 'Passive Turnover',
    'passive_turnover': 'Passive Turnover',
    'passive_win_rate': 'Passive Win Rate',
    # Rank IC metrics
    'rank_ic': 'Rank IC',
    'rank_icir': 'Rank ICIR',
    'rank_icir_annual': 'Rank ICIR Annualized',
    'rank_ic_p_value': 'Rank IC p-value',
    'rank_ic_winratio': 'Rank IC Win Ratio',
    # ACF metrics
    'acf_1': 'ACF 1',
    'acf_halflife': 'ACF Half-life',
}

# Metrics to display as percentages
PERCENTAGE_METRICS = {
    'long_short_ret_annual', 'long_short_ret_max_dd', 'long_short_win_rate',
    'long_short_turnover_ratio', 'long_short_turnover',
    'long_ret_annual', 'long_ret_max_dd', 'long_win_rate',
    'long_turnover_ratio', 'long_turnover',
    'short_ret_annual', 'short_ret_max_dd', 'short_win_rate',
    'short_turnover_ratio', 'short_turnover',
    'passive_ret_annual', 'passive_ret_max_dd', 'passive_win_rate',
    'passive_turnover_ratio', 'passive_turnover',
    'rank_ic_winratio',
}

# Metrics to display as ratios (4 decimal places)
RATIO_METRICS = {
    'long_short_ret_sharpe', 'long_ret_sharpe', 'short_ret_sharpe', 'passive_ret_sharpe',
    'long_short_ret_calmar', 'long_ret_calmar', 'short_ret_calmar', 'passive_ret_calmar',
    'long_short_ret_sortino', 'long_ret_sortino', 'short_ret_sortino', 'passive_ret_sortino',
    'long_short_ret_sharpe_per_turnover', 'long_ret_sharpe_per_turnover',
    'short_ret_sharpe_per_turnover', 'passive_ret_sharpe_per_turnover',
    'rank_ic', 'rank_icir', 'rank_icir_annual', 'acf_1',
}


def get_metric_display_name(name: str) -> str:
    """
    Map internal metric names to professional display names.

    Args:
        name: Internal metric name

    Returns:
        Display name for the metric
    """
    return METRIC_DISPLAY_NAMES.get(name, name.replace('_', ' ').title())


def format_metric(name: str, value: float) -> str:
    """
    Format metric values appropriately for display.

    Args:
        name: Metric name
        value: Metric value

    Returns:
        Formatted string representation
    """
    try:
        is_nan = pd.isna(value)
    except Exception:
        is_nan = False

    if is_nan:
        return "NaN"

    if name in PERCENTAGE_METRICS:
        return f"{value * 100:.2f}%"
    elif name == 'acf_halflife':
        return f"{int(value)}"
    elif name in RATIO_METRICS:
        return f"{value:.4f}"
    elif name == 'rank_ic_p_value':
        return f"{value:.4f}"
    else:
        return f"{value:.4f}"
