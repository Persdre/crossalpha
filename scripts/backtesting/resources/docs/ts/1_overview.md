# Time Series Backtest Overview

Single-asset backtesting framework for evaluating factor-based trading signals over time.

## Workflow

```
Factor Data → Factor-to-Position → PnL Calculation → Performance Metrics → Report
```

1. **Data Preparation**: Align factor and label series, apply lag
2. **Factor-to-Position**: Convert factor values to position weights
3. **PnL Calculation**: Compute transaction-adjusted returns per fee scenario
4. **Metrics**: Calculate fee-independent and fee-dependent performance metrics
5. **Visualization**: Generate interactive plots

## Backtest Modes

| Mode | Description | Key Parameters |
|------|-------------|----------------|
| `weights` | Direct factor values as positions | None |
| `long/short_threshold` | Binary signals at thresholds | `long_threshold`, `short_threshold` |
| `gradual_long/short_threshold` | Smooth transitions between zones | `end_short`, `start_short`, `start_long`, `end_long` |
| `long/short_normalization` | Rolling winsorize + normalize | `winsorize_method`, `normalization_method` |

## Position Types

Four position types are evaluated for each backtest:

| Type | Description | Formula |
|------|-------------|---------|
| `ls` (Long/Short) | Full position from mode | pos |
| `long` | Long-only positions | max(pos, 0) |
| `short` | Short-only positions | min(pos, 0) |
| `passive` | Buy-and-hold benchmark | 1.0 (constant) |

## Fee Scenarios

Multiple transaction cost scenarios are tested (default: 0, 2, 5 basis points) to evaluate strategy robustness to trading costs.

## Output Structure

**Fee-Independent Metrics** (calculated once):
- Rank IC / IC, ICIR, p-value, win ratio
- Factor autocorrelation (ACF)
- Turnover ratios

**Fee-Dependent Metrics** (per fee scenario, per position type):
- Annualized Return, Annualized Excess Return (ls/long/short only)
- Sharpe, Sortino, Max Drawdown, Calmar
- Win Rate, Sharpe per Turnover
