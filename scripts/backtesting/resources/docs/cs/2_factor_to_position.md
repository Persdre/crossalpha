# Factor to Position Conversion

Convert factor values to portfolio weights using one of two modes.

---

## Mode 1: Long/Short Layers

Discrete position assignment based on quantile ranking.

### Layer Assignment

At each datetime $t$, assets are ranked by factor value (descending) and assigned to $L$ quantile layers:

$$
\text{layer}(i, t) = \lceil \text{rank}(i, t) \times L / N \rceil
$$

Where $N$ = total assets, $L$ = number of layers. Layer 1 contains top-ranked assets (highest factor values).

### Position Construction

| Position Type | Formula |
|---------------|---------|
| Long/Short | $+1$ if layer $\in$ `long_layer_index`, $-1$ if layer $\in$ `short_layer_index`, else $0$ |
| Long-only | $+1$ if layer $\in$ `long_layer_index`, else $0$ |
| Short-only | $-1$ if layer $\in$ `short_layer_index`, else $0$ |

Default: `long_layer_index=[1]`, `short_layer_index=[L]` (top vs bottom layer)

---

## Mode 2: Long/Short Normalization

Continuous position weights derived from normalized factor values.

### Winsorization (Optional)

Clips extreme values cross-sectionally before normalization:

| Method | Formula | Default $n$ |
|--------|---------|-------------|
| `std` | Clip to $[\mu - n\sigma, \mu + n\sigma]$ | 3 |
| `mad` | Clip to $[\text{median} - n \cdot \text{MAD}, \text{median} + n \cdot \text{MAD}]$ | 3 |
| `quantile` | Clip to $[Q_n, Q_{1-n}]$ | 0.01 |

### Normalization Methods

| Method | Formula | Output Range |
|--------|---------|--------------|
| `rank` | $2 \times \frac{\text{rank}(x_i) - 1}{N - 1} - 1$ | $[-1, 1]$ |
| `minmax` | $2 \times \frac{x_i - x_{\min}}{x_{\max} - x_{\min}} - 1$ | $[-1, 1]$ |
| `zscore` | $\frac{x_i - \mu}{\sigma}$ | Unbounded |
| `demean` | $x_i - \mu$ | Unbounded |

### Position Construction

| Position Type | Formula |
|---------------|---------|
| Long/Short | $\text{pos}_{\text{ls}} = \text{normalized factor}$ |
| Long-only | $\text{pos}_{\text{long}} = \max(0, \text{pos}_{\text{ls}})$ |
| Short-only | $\text{pos}_{\text{short}} = \min(0, \text{pos}_{\text{ls}})$ |

---

## Position Normalization (Both Modes)

All positions are normalized to achieve dollar-neutral long/short portfolios:

### Step 1: Demean
$$
\text{pos}_{\text{demeaned}} = \text{pos} - \text{mean}(\text{pos})
$$

### Step 2: Normalize to Gross = 1
$$
\text{weight}_i = \frac{\text{pos}_{\text{demeaned}, i}}{\sum_j |\text{pos}_{\text{demeaned}, j}|}
$$

**Result**: Long weights sum to ~0.5, short weights sum to ~-0.5, gross exposure = 1.

---

## Passive Benchmark

Equal-weight portfolio for comparison:

$$
\text{weight}_{\text{passive}, i} = \frac{1}{N}
$$

Where $N$ = number of assets at datetime $t$.
