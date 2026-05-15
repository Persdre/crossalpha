# Performance Metrics

All metrics computed for each position type (ls, long, short, long passive, short passive).

---

## Fee-Independent Metrics

Calculated once; intrinsic to factor-label relationship.

### Rank IC (Information Coefficient)

Cross-sectional Spearman correlation between factor and label:

$$
\text{Rank IC}(t) = \text{Spearman}(\text{factor}(t), \text{label}(t))
$$

$$
\text{Rank IC} = \text{mean}(\text{Rank IC}(t))
$$

### IC (Information Coefficient)

Cross-sectional Pearson correlation between factor and label:

$$
\text{IC}(t) = \text{Pearson}(\text{factor}(t), \text{label}(t))
$$

$$
\text{IC} = \text{mean}(\text{IC}(t))
$$

### Rank ICIR

Consistency of Rank IC over time (daily aggregation):

$$
\text{Rank ICIR} = \frac{|\text{mean(daily Rank IC)}|}{\text{std(daily Rank IC)}}
$$

$$
\text{Rank ICIR Annual} = \text{Rank ICIR} \times \sqrt{\text{dates\_to\_years}}
$$

### Rank IC Statistics

| Metric | Formula |
|--------|---------|
| Rank IC p-value | $2 \times (1 - \Phi(\|z\|))$ where $z = \frac{\mu}{\sigma / \sqrt{n}}$ |
| Rank IC Win Ratio | $\frac{\text{count}(\text{Rank IC} > 0)}{n}$ |

### ICIR

Consistency of IC over time (daily aggregation):

$$
\text{ICIR} = \frac{|\text{mean(daily IC)}|}{\text{std(daily IC)}}
$$

$$
\text{ICIR Annual} = \text{ICIR} \times \sqrt{\text{dates\_to\_years}}
$$

### IC Statistics

| Metric | Formula |
|--------|---------|
| IC p-value | $2 \times (1 - \Phi(\|z\|))$ where $z = \frac{\mu}{\sigma / \sqrt{n}}$ |
| IC Win Ratio | $\frac{\text{count}(\text{IC} > 0)}{n}$ |

### Factor Autocorrelation (ACF)

Cross-sectional Pearson correlation of factor values across time lags:

$$
\text{ACF}(\text{lag}) = \text{mean}(\text{Pearson}(f_t, f_{t+\text{lag}}))
$$

| Metric | Description |
|--------|-------------|
| `acf_1` | ACF at lag 1 (factor persistence) |
| `acf_halflife` | First lag where ACF < 0.5 |

### Turnover

Position changes per datetime:

$$
\text{Turnover}(t) = \sum_i |w_{i,t} - w_{i,t-1}|
$$

$$
\text{Turnover Ratio} = \text{mean(daily turnover)}
$$

---

## Fee-Dependent Metrics

Calculated for each fee scenario (default: 0, 2, 5 bps).

### PnL Calculation

Transaction-adjusted profit/loss:

$$
\text{tpnl}(t) = \text{old\_pos\_adjusted}(t) - \text{old\_pos}(t) - \text{trade\_cost}(t)
$$

Where:
- $\text{old\_pos\_adjusted}$ = previous position adjusted for price change
- $\text{trade\_cost} = |\text{position\_change}| \times \text{fee}$

### Return Metrics

| Metric | Formula |
|--------|---------|
| Annualized Return | $\frac{\sum \text{daily PnL}}{\text{years}}$ |
| Annualized Excess Return | Long: $\frac{\sum (\text{daily PnL}_{\text{long}} - \text{daily PnL}_{\text{long passive}})}{\text{years}}$; Short: $\frac{\sum (\text{daily PnL}_{\text{short}} - \text{daily PnL}_{\text{short passive}})}{\text{years}}$ |
| Sharpe Ratio | $\frac{\mu_{\text{daily}}}{\sigma_{\text{daily}}} \times \sqrt{\text{dates\_to\_years}}$ |
| Sortino Ratio | $\frac{\mu_{\text{daily}}}{\sigma_{\text{downside}}} \times \sqrt{\text{dates\_to\_years}}$ |
| Max Drawdown | $\min\left(\frac{\text{cumPnL} - \text{runningMax}}{\text{runningMax} + 1}\right)$ (capped at -100%) |
| Calmar Ratio | $\frac{\text{Annualized Return}}{|\text{Max Drawdown}|}$ |
| Win Rate | $\frac{\text{count}(\text{tpnl} > 0 \land \text{pos} \neq 0)}{\text{count}(\text{pos} \neq 0)}$ |
| Sharpe per Turnover | $\frac{\text{Sharpe}}{\text{Turnover Ratio}}$ |
---
