# Performance Plots

15 interactive plots visualize strategy performance, factor quality, and risk characteristics.

---

## Strategy Returns (5 plots)

Cumulative returns over time for each position type, with separate curves per fee scenario.

| Plot | Description |
|------|-------------|
| L/S Cumulative Returns | Long/short dollar-neutral strategy |
| Long Cumulative Returns | Long-only positions |
| Short Cumulative Returns | Short-only positions |
| Long Passive Investment Cumulative Returns | Equal-weight long-only benchmark |
| Short Passive Investment Cumulative Returns | Equal-weight short-only benchmark |

---

## Layer Analysis (3 plots)

Layer-specific performance.

| Plot | Description |
|------|-------------|
| Layer Cumulative Returns | Return curves per layer over time |
| Layer Annual Returns | Bar chart of annualized returns per layer |
| Layer Rank IC | Bar chart of mean Rank IC per layer |

**Interpretation**: Monotonic relationship across layers (e.g., layer 1 highest, layer N lowest) indicates strong factor predictive power.

---

## Factor Quality (2 plots)

Signal strength and consistency over time.

| Plot | Description |
|------|-------------|
| Rank IC Cumulative | Cumulative sum of daily Rank IC over time |
| Rank IC Interval Analysis | Rank IC averaged over 15 non-overlapping intervals with mean reference line |

**Interpretation**:
- Upward-sloping cumulative IC indicates persistent predictive signal
- Consistent interval IC indicates stable signal across market regimes

---

## Factor Characteristics (2 plots)

Factor distribution and persistence.

| Plot | Description |
|------|-------------|
| Factor Distribution | Histogram with mean, std, min, max annotations |
| Factor Autocorrelation | ACF by lag with 0.5 reference line (half-life marker) |

**Interpretation**:
- High autocorrelation = slow-changing factor = lower turnover
- Low autocorrelation = fast-changing factor = higher turnover

---

## Turnover Analysis (3 plots)

Position changes at each datetime with mean/min/max reference lines.

| Plot | Description |
|------|-------------|
| L/S Turnover | Long/short strategy turnover scatter |
| Long Turnover | Long-only turnover |
| Short Turnover | Short-only turnover |

**Interpretation**: High turnover increases transaction costs; compare gross returns (0 bp) vs net returns (2/5 bp) to assess impact.
