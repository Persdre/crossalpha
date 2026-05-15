# Factor to Position Conversion

Convert factor values to position weights using one of four modes.

---

## Mode 1: Weights

Direct use of factor values as positions:

$$
\text{pos} = \text{factor}
$$

**Use case**: When factor values directly represent desired position sizes.

---

## Mode 2: Long/Short Threshold

Binary positions based on threshold levels:

$$
\text{pos} = \begin{cases}
+1 & \text{if factor} \geq \text{long\_threshold} \\
-1 & \text{if factor} \leq \text{short\_threshold} \\
0 & \text{otherwise}
\end{cases}
$$

| Parameter | Description |
|-----------|-------------|
| `long_threshold` | Factor value above which to go long |
| `short_threshold` | Factor value below which to go short |

---

## Mode 3: Gradual Long/Short Threshold

Smooth transitions between position zones to avoid cliff effects:

| Zone | Condition | Position |
|------|-----------|----------|
| Fully Short | factor ≤ end_short | -1 |
| Gradual Short | end_short < factor < start_short | Linear: -1 → 0 |
| Neutral | start_short ≤ factor ≤ start_long | 0 |
| Gradual Long | start_long < factor < end_long | Linear: 0 → +1 |
| Fully Long | factor ≥ end_long | +1 |

**Linear interpolation formulas**:
- Gradual Short: $\text{pos} = -1 + \frac{\text{factor} - \text{end\_short}}{\text{start\_short} - \text{end\_short}}$
- Gradual Long: $\text{pos} = \frac{\text{factor} - \text{start\_long}}{\text{end\_long} - \text{start\_long}}$

---

## Mode 4: Long/Short Normalization

Rolling window preprocessing pipeline with three stages:

### Stage 1: Winsorization (Optional)

Clip extreme values using rolling window statistics:

| Method | Formula |
|--------|---------|
| `std` | Clip to $[\mu - n\sigma, \mu + n\sigma]$ |
| `mad` | Clip to $[\text{median} - n \cdot \text{MAD}, \text{median} + n \cdot \text{MAD}]$ |
| `quantile` | Clip to $[Q_n, Q_{1-n}]$ |

### Stage 2: Normalization

Apply rolling window normalization:

| Method | Formula | Output Range |
|--------|---------|--------------|
| `zscore` | $(x - \mu) / \sigma$ | Unbounded |
| `minmax` | $(x - \min) / (\max - \min) \times 2 - 1$ | $[-1, 1]$ |
| `rank` | $\text{rank} / n \times 2 - 1$ | $[-1, 1]$ |
| `demean` | $x - \mu$ | Unbounded |

### Stage 3: Squashing

Transform to bounded range:

| Method | Formula | Output |
|--------|---------|--------|
| `clip` | $\text{clip}(x, -1, 1)$ | $[-1, 1]$ |
| `tanh` | $\tanh(x)$ | $(-1, 1)$ |

---

## Position Types (All Modes)

From the base position, four position types are derived:

| Type | Formula | Description |
|------|---------|-------------|
| `pos_ls` | pos | Full long/short position |
| `pos_long` | $\max(\text{pos}, 0)$ | Long-only component |
| `pos_short` | $\min(\text{pos}, 0)$ | Short-only component |
| `pos_passive` | 1.0 | Buy-and-hold benchmark |
