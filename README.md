# driftmind-benchmark

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](CHANGELOG.md)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#current-limitations)

An open-source, evolving benchmarking framework for real-time signal monitoring, forecasting, and drift-aware analytics.
This repository focuses on systematic, reproducible evaluation of DriftMind against classical and modern baselines under non-stationary conditions.

The objective is not merely accuracy comparison, but a holistic assessment of adaptability, latency, and robustness in environments subject to concept drift.

---

## What is DriftMind?

DriftMind is a real-time forecasting and change-detection engine designed for high-frequency, non-stationary signals.
It explicitly addresses Concept Drift, where the underlying statistical properties of a signal evolve over time, often abruptly and without prior notice.

Unlike static or batch-trained models, DriftMind continuously adapts to shifting regimes while maintaining predictable execution characteristics.

---

## Core Capabilities

- Real-Time Adaptation
  Dynamically updates internal structures and parameters to align with the current signal regime, without manual re-tuning or retraining cycles.

- Multi-Domain Applicability
  Validated across heterogeneous domains, including:
  - Network telemetry (TCP/UDP metrics)
  - IoT and sensor streams
  - Financial and market time series

- Change-Point Detection
  Identifies precise points of structural change in the signal, forming a robust foundation for anomaly detection, root-cause analysis, and adaptive forecasting.

---

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
# Core dependencies only (ARIMA models)
uv sync

# Include lstm dependencies (required for lstm.py)
uv sync --extra lstm

# Include demo dependencies (JupyterLab, pandas, matplotlib, seaborn, tqdm — required to run the notebook)
uv sync --extra demo

# Everything
uv sync --extra lstm --extra demo

# Development tools (pytest, ruff)
uv sync --extra dev
```

## Model implementations

Four baseline forecasting strategies, each exposing a `predict_point(window)` interface for step-by-step online prediction:

| Model               | File                   | Description                                                                                                        |
|---------------------|------------------------|--------------------------------------------------------------------------------------------------------------------|
| **Static ARIMA**    | `src/arima/s_arima.py` | Fits ARIMA once on a training block, then reuses the fixed coefficients for all future predictions.                |
| **Frozen ARIMA**    | `src/arima/f_arima.py` | Fits once, then updates the Kalman Filter state with new observations — no parameter refit ever.                   |
| **Triggered ARIMA** | `src/arima/t_arima.py` | Separates state updates from parameter fitting; re-fits only when explicitly triggered (e.g., on drift detection). |
| **LSTM**            | `src/lstm/lstm.py`     | PyTorch LSTM neural network. Trains on an initial block of data, then does single-step inference point-by-point.   |

All models measure per-prediction latency with `time.perf_counter()`.

## Usage

Every baseline follows the same two-phase lifecycle, making them interchangeable in the benchmark loop:

```python
import numpy as np
from arima.s_arima import StaticARIMABaseline  # or any other baseline

signal = np.load("my_signal.npy")

# Phase 1 — warm-up: fit the model on an initial block of data
warmup_fraction = 0.01
warmup_end = int(len(signal) * warmup_fraction)

model = StaticARIMABaseline(order=(5, 1, 0))
model.train(signal[:warmup_end])

# Phase 2 — streaming: slide a window and predict one step at a time
window_size = 200
predictions, latencies = [], []

for i in range(warmup_end, len(signal)):
    window = signal[max(0, i - window_size):i]
    yhat, latency = model.predict_point(window)
    predictions.append(yhat)
    latencies.append(latency)
```

`train()` must be called before `predict_point()`; all models raise `RuntimeError` otherwise. `TriggeredARIMABaseline` additionally exposes `train()` as a public method so an external drift detector can call it again at any point during streaming to re-fit the model on the latest window.

## Analysis notebook

`benchmark/t_arima.ipynb` compares DriftMind API results against a baseline adaptive ARIMA. It has two phases:

### Phase 1 — Aggregate analysis

Loads pre-computed DriftMind results from `data/driftmind_results/analysis_results.csv` and produces:

- A summary table grouped by **Category**: file count, average sMAE, average throughput (predictions/s), total points.
- **Average sMAE by category** — normalised error (lower is better).
- **Average throughput by category** — inference speed (higher is better).
- **Volatility vs error scatter** — StdDev (log scale) vs sMAE by category; checks whether error degrades as signal variance grows.

### Phase 2 — Per-experiment comparison: DriftMind vs adaptive ARIMA

`visualize_experiment_full(id)` plots a single DriftMind experiment: actual signal, predicted trace, and absolute error over time.

`benchmark_with_target(id)` runs an adaptive online ARIMA(5, 1, 0) on the same experiment using `TriggeredARIMABaseline` (`t_arima.py`) and produces a head-to-head comparison. Two drift-detection triggers control re-fitting:

| Trigger          | Condition                                        | Action                               |
|------------------|--------------------------------------------------|--------------------------------------|
| MASE drift       | MASE > 3.0 for 20 consecutive steps              | Full re-fit on last 200 observations |
| Structural drift | Pearson correlation < 0.20 (over last 100 steps) | Immediate re-fit                     |

Results are printed (MAE, sMAE, throughput) and plotted as dual prediction traces with absolute error ribbons.

### Data layout required to run the notebook

All result files are committed under `benchmark/data/` and are ready to use:

```
benchmark/data/
  driftmind_results/
    analysis_results.csv                                    # One row per experiment (metadata + metrics)
    experiments/
      Experiment_<id>_<name>_Report.csv                     # Per-experiment predictions: Actual, Expected, AE columns
```

---

## Scientific Fairness and Benchmarking Integrity

This benchmark suite is designed with methodological rigor as a first-class concern.

### On-Premises Latency Simulation

The DriftMind core engine is proprietary and consumed via a REST API.
To ensure fair and meaningful comparison with local baselines (for example ARIMA, LSTM, and PyTorch-based models), all reported DriftMind benchmarks reflect on-premises execution characteristics, equivalent to the binary/paid deployment.

This approach removes network latency from the equation and avoids artificially penalizing DriftMind in throughput and latency measurements.

### Verifiable and Reproducible Results

- All DriftMind outputs included in this repository are genuine and reproducible.
- Benchmarks are executed offline, but rely on the exact same core logic used in production deployments.
- Intermediate logs and consolidated result files are provided to support independent verification.

---

## Repository structure

```text
benchmark/
|-- *.ipynb                     # Exploration and head-to-head benchmark notebooks
`-- data/
    |-- driftmind_results/      # DriftMind predictions, logs, and analysis_results.csv
    `-- source_data/            # Raw input signal files

src/
|-- arima/
|   |-- f_arima.py              # Frozen ARIMA (state-only updates)
|   |-- t_arima.py              # Triggered ARIMA (reactive re-fitting)
|   `-- s_arima.py              # Static/standard ARIMA baseline
`-- lstm/
    `-- lstm.py                 # LSTM baseline (deep learning approach)

images/benchmark/               # Result plots
pyproject.toml                  # Package config
```

---

## Hyperparameters

### Static ARIMA

| Param   | Default   | Description                                      |
|---------|-----------|--------------------------------------------------|
| `order` | (5, 1, 0) | ARIMA (p, d, q) — AR lags, differencing, MA lags |

### Frozen ARIMA

| Param   | Default   | Description                                          |
|---------|-----------|------------------------------------------------------|
| `order` | (5, 1, 0) | ARIMA (p, d, q); parameters frozen after initial fit |

### Triggered ARIMA

| Param   | Default   | Description                                                    |
|---------|-----------|----------------------------------------------------------------|
| `order` | (5, 1, 0) | ARIMA (p, d, q); only re-fit when drift is detected externally |

### LSTM

| Param               | Default | Description                             |
|---------------------|---------|-----------------------------------------|
| `hidden_layer_size` | 64      | Number of LSTM hidden units             |
| `seq_length`        | 50      | Input sequence length (lookback window) |
| `epochs`            | 20      | Training epochs                         |
| `learning_rate`     | 0.001   | Adam optimizer learning rate            |
| `feature_range`     | (0, 1)  | MinMaxScaler output range               |

### Benchmark settings (notebook)

| Param                    | Value | Description                                               |
|--------------------------|-------|-----------------------------------------------------------|
| `training_fraction`      | 0.01  | Fraction of data used for initial model training          |
| `arima_fit_window`       | 200   | Lookback window size fed to ARIMA models                  |
| `mase_limit`             | 3.0   | MASE threshold to trigger re-fit                          |
| `mase_consecutive_steps` | 20    | Consecutive steps above threshold before triggering       |
| `corr_floor`             | 0.20  | Correlation floor; re-fit if correlation drops below this |

---

## Participation and Contributions

This benchmark suite is intentionally open and extensible.
We welcome contributions from researchers and practitioners interested in advancing the state of real-time signal monitoring.

### Submit a New Dataset

We accept univariate and multivariate time series. Datasets should include:

- Ground-truth annotations (for example drift points, anomalies)
- Clear domain or category labeling
- Sufficient metadata to ensure interpretability

### Propose a New Baseline

New baseline models are welcome (for example Prophet, Transformers, custom DSP, or statistical filters), provided they meet the following criteria:

- Cold-Start Ready
  Must operate with minimal historical context.

- Adaptable
  Explicitly designed for non-stationary data.

- Transparent and Verifiable
  Full source code and reproducible results are required.

---

## Current limitations

- **No runnable pipeline.** The model implementations are standalone; there is no harness or runner script that executes them against a dataset and collects results.

## Documentation

A pre-rendered version of the analysis notebook — including all output plots — is available at [`docs/t_arima.md`](docs/t_arima.md).

To regenerate it after a new run:

```bash
uv sync --extra demo
uv run jupyter nbconvert --to markdown benchmark/t_arima.ipynb --output-dir docs/
mv docs/t_arima_files/* docs/images/
rmdir docs/t_arima_files
```

## License

[MIT](LICENSE)
