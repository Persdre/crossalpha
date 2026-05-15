# Cross Sectional Backtest Overview

Cross-sectional backtesting framework for evaluating factor-based strategies across multiple assets.

## Workflow

```
Factor Data → Factor-to-Position → PnL Calculation → Performance Metrics → Report
```

1. **Data Preparation**: Align factor and label data, apply lag
2. **Factor-to-Position**: Convert factor values to portfolio weights
3. **PnL Calculation**: Compute transaction-adjusted returns per fee scenario
4. **Metrics**: Calculate fee-independent and fee-dependent performance metrics
5. **Visualization**: Generate interactive plots

## Backtest Modes

| Mode | Description | Key Parameters |
|------|-------------|----------------|
| `long/short_layers` | Quantile ranking → layer assignment | `long_layer_index`, `short_layer_index`, `layers` |
| `long/short_normalization` | Winsorize → normalize → constrain | `winsorize_method`, `normalization_method` |

## Position Types

Four position types are evaluated for each backtest:

| Type | Description | Constraint |
|------|-------------|------------|
| `ls` (Long/Short) | Dollar-neutral portfolio | sum ≈ 0, gross = 1 |
| `long` | Long-only positions | weights ≥ 0, sum = 1 |
| `short` | Short-only positions | weights ≤ 0, sum = -1 |
| `passive` | Equal-weight benchmark | 1/n per symbol |

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
