
# DriftMind Benchmark Suite

An open-source, evolving benchmarking framework for **real-time signal monitoring, forecasting, and drift-aware analytics**.  
This repository focuses on **systematic, reproducible evaluation** of DriftMind against classical and modern baselines under **non-stationary conditions**.

The objective is not merely accuracy comparison, but a **holistic assessment** of adaptability, latency, and robustness in environments subject to **concept drift**.

---

## What is DriftMind?

DriftMind is a **real-time forecasting and change-detection engine** designed for high-frequency, non-stationary signals.  
It explicitly addresses **Concept Drift**, where the underlying statistical properties of a signal evolve over time—often abruptly and without prior notice.

Unlike static or batch-trained models, DriftMind continuously adapts to shifting regimes while maintaining predictable execution characteristics.

---

## Core Capabilities

- **Real-Time Adaptation**  
  Dynamically updates internal structures and parameters to align with the current signal regime, without manual re-tuning or retraining cycles.

- **Multi-Domain Applicability**  
  Validated across heterogeneous domains, including:
  - Network telemetry (TCP/UDP metrics)
  - IoT and sensor streams
  - Financial and market time series

- **Change-Point Detection**  
  Identifies precise points of structural change in the signal, forming a robust foundation for anomaly detection, root-cause analysis, and adaptive forecasting.

---

## Scientific Fairness & Benchmarking Integrity

This benchmark suite is designed with **methodological rigor** as a first-class concern.

### On-Premises Latency Simulation

The DriftMind core engine is proprietary and consumed via a REST API.  
To ensure **fair and meaningful comparison** with local baselines (e.g., ARIMA, LSTM, PyTorch-based models), all reported DriftMind benchmarks reflect **on-premises execution characteristics**, equivalent to the binary/paid deployment.

This approach removes network latency from the equation and avoids artificially penalizing DriftMind in throughput and latency measurements.

### Verifiable and Reproducible Results

- All DriftMind outputs included in this repository are **genuine and reproducible**.
- Benchmarks are executed offline, but rely on the **exact same core logic** used in production deployments.
- Intermediate logs and consolidated result files are provided to support independent verification.

---

## Repository Structure

The project cleanly separates **benchmark orchestration**, **data**, and **baseline implementations**:

```
benchmark/
├── *.ipynb                     # Exploration and head-to-head benchmark notebooks
└── data/
├── driftmind_results/       # DriftMind predictions, logs, and analysis_results.csv
└── source_data/             # Raw input signal files

src/
├── t_arima.py                   # Triggered ARIMA (reactive re-fitting)
├── f_arima.py                   # Frozen ARIMA (state-only updates)
└── lstm_baseline.py             # LSTM baseline (deep learning approach)

```

Notebook names explicitly indicate the **benchmark scenario** and corresponding baselines under evaluation.

---

## Participation & Contributions

This benchmark suite is intentionally **open and extensible**.  
We welcome contributions from researchers and practitioners interested in advancing the state of real-time signal monitoring.

### Submit a New Dataset

We accept univariate and multivariate time series. Datasets must include:

- Ground-truth annotations (e.g., drift points, anomalies)
- Clear domain or category labeling
- Sufficient metadata to ensure interpretability

### Propose a New Baseline

New baseline models are welcome (e.g., Prophet, Transformers, custom DSP or statistical filters), provided they meet the following criteria:

- **Cold-Start Ready**  
  Must operate with minimal historical context.

- **Adaptable**  
  Explicitly designed for non-stationary data.

- **Transparent & Verifiable**  
  Full source code and reproducible results are required.

---

## Getting Started

1. Clone the repository.
2. Install dependencies:
```

pip install pandas numpy matplotlib seaborn torch statsmodels tqdm

```
3. Open and run the notebooks in:
```
benchmark/*.ipynb
```

to reproduce baseline comparisons and analysis.

