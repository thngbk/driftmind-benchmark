# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-01

### Fixed

**Uninitialised-model guards** — all three cases produced silent or confusing failures when `predict_point()` was called before `train()`:
- `StaticARIMABaseline.predict_point()`: missing `model_fit is None` check caused an `AttributeError`; now raises `RuntimeError`.
- `TriggeredARIMABaseline.predict_point()`: silently called `self.train(window)`, masking the misuse entirely; now raises `RuntimeError`.
- `LSTMBaseline.predict_point()`: no guard at all; now raises `RuntimeError` via `hasattr(self.scaler, "scale_")`.

**Warning suppression** — statsmodels warnings were either too broadly suppressed or not suppressed at all during inference:
- All three ARIMA baselines: blanket `warnings.simplefilter("ignore")` silenced all warnings indiscriminately; narrowed to targeted suppression of `ConvergenceWarning` and `UserWarning` only.
- `FrozenARIMABaseline.predict_point()`: suppression was missing from the inference path entirely; forecast and state-update are now wrapped in `warnings.catch_warnings()`.

**Logic errors**:
- `TriggeredARIMABaseline.train()`: warm-start condition `if self.model_fit` could evaluate a fitted model object as falsy; corrected to `if self.model_fit is not None`.
- `LSTMBaseline.predict_point()`: latency measurement started after the input-scaling step, understating true end-to-end inference time; `time.perf_counter()` now called before all inference operations.

### Changed

- All three ARIMA baselines: added a `_SUPPRESSED = (ConvergenceWarning, UserWarning)` class-level constant and a `_check_window()` helper that validates the input window length against the minimum observations required by the configured `ARIMA(p, d, q)` order, raising a descriptive `ValueError` on violation.
- All source files: removed unused `numpy` and `pandas` imports from the ARIMA modules; imports reordered to comply with isort conventions (stdlib → third-party, alphabetically within each group).
- `LSTMBaseline`: unused training-loop variable renamed from `i` to `_i`.
- `benchmark/t_arima.ipynb`: imports reordered (stdlib → third-party, alphabetical) and string literals standardised to double quotes; outputs and execution counts stripped.

### Build

- Added `pyproject.toml` defining project metadata, core dependencies (`numpy`, `statsmodels`), optional-dependency groups (`lstm`, `demo`, `dev`), `ruff` linting/formatting configuration, and `pytest` settings; requires Python ≥ 3.9.
- Added `uv.lock` for reproducible dependency resolution via `uv`.
- Added `.gitattributes`.
- `nbstripout` added to `dev` dependencies and installed as a git filter; notebook outputs are stripped automatically on every commit.

### Documentation

- Added `CHANGELOG.md` (this file).
- `README.md`: major rewrite — added version/license/status badges; Installation section with `uv sync` commands for each optional group; Model implementations table; Usage section with two-phase lifecycle example; Analysis notebook walkthrough with drift-trigger condition table and corrected data layout path (`benchmark/data/`) and filename pattern; Hyperparameters tables for all models and the benchmark notebook; Current limitations section; Documentation section with pre-rendered notebook link and regeneration instructions; and a License section.
- Added `docs/t_arima.md`: pre-rendered export of `benchmark/t_arima.ipynb` including all output plots, generated via `nbconvert`.
- Added `docs/images/`: plots extracted from the notebook export (aggregate analysis charts and head-to-head comparison).
- Removed `images/` — static artefacts with no traceable source in the repository.

## [0.1.0] - 2026-02-25

- Initial release.
