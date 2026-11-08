# SGCC electricity theft detection

This is the consumer-classification implementation for Energy Loss Agent. The original project overview is preserved in [docs/agent-overview.md](docs/agent-overview.md); the producers-module plan remains in [docs/producers-plan.md](docs/producers-plan.md).

This project prepares consumer-level supervised binary classification datasets, then trains, evaluates and compares LightGBM, XGBoost, a PyTorch 1D CNN, and a hybrid CNN/statistical-feature model. Preprocessing and training have separate entry points: preparing data never trains a classifier.

The supplied `data.csv` comes from the [SGCC electricity theft dataset repository](https://github.com/henryRDlab/ElectricityTheftDetection). `CONS_NO` identifies a consumer; `FLAG=0` means normal and `FLAG=1` means labeled as electricity theft. Labels make this supervised classification, rather than unsupervised clustering. A label applies to the whole consumer history and does not establish the day theft occurred.

The expected source has 42,372 consumers (38,757 normal, 3,615 theft), 1,034 daily measurement columns, and two identity/target columns. Date columns are originally out of order. Alignment to January 1, 2014–October 31, 2016 produces 1,035 days, inserting the absent September 18, 2016 as missing. The loader measures actual dimensions and anomalies instead of assuming these reference counts are correct.

## Setup and execution

Run from this folder. The verified environment uses Python 3.14.4; use Python 3.14 with the supplied lockfile. Preprocessing needs no GPU. Training uses LightGBM, XGBoost and PyTorch and supports CPU execution.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.prepare_data --config config.yaml --all
.\.venv\Scripts\python.exe -m src.validation --output-dir data/processed
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.prepare_data --config config.yaml --all
.venv/bin/python -m src.validation --output-dir data/processed
```

`requirements.txt` defines supported dependency ranges; `requirements-lock.txt` records the exact versions used for this run. Install the former when intentionally updating dependencies, then rerun tests and regenerate datasets. Binary wheels must be available for your platform/Python version.

Pandas is pinned to 2.3.3: this host's Windows Application Control rejected the pandas 3.1.1 binary wheel. The pinned version passed the complete test suite and real-data run without changing Windows security settings.

With the environment activated, the equivalent entry points are:

```bash
python -m src.prepare_data --config config.yaml --all
python -m src.prepare_data --config config.yaml --model lightgbm
python -m src.prepare_data --config config.yaml --model xgboost
python -m src.prepare_data --config config.yaml --model cnn1d
python -m src.prepare_data --config config.yaml --model hybrid
```

All paths in YAML resolve relative to that YAML file. The default input is the existing root-level `data.csv`; no copy or move is necessary. Unknown settings and unsupported policies fail explicitly. When source content, configuration, code, or dependency versions change, use `--all` or a new output directory. Individual-variant runs reject mixing different generations. Only consume outputs after successful completion and `shared/run_manifest.json` reports `validation_passed: true`.

## Layout and artifacts

```text
config.yaml                 centralized policies and paths
src/config.py               defaults and configuration validation
src/data_loader.py          chunked loading, audit, calendar alignment
src/splitting.py            shared stratified consumer assignment
src/feature_engineering.py  common statistical feature extraction
src/preprocessing.py        saved fit/transform components
src/dataset_builders.py     serialization and reports
src/prepare_data.py         command-line entry point
src/validation.py           full saved-artifact validation
tests/                      synthetic unit and integration tests
data/processed/
  shared/                   splits, standardized data, audit, reports, manifest
  lightgbm/                 train/val/test.parquet, feature_names.json, metadata.json
  xgboost/                  same features/targets as LightGBM
  cnn1d/                    X_*.npy, y_*.npy, *_ids.csv, preprocessor.joblib, metadata.json
  hybrid/                   X_sequence_*.npy, X_features_*.npy, y_*.npy, *_ids.csv,
                            feature_names.json, feature_preprocessor.joblib,
                            sequence_preprocessor.joblib, metadata.json
```

Shared outputs include `consumption.npy` (sanitized, otherwise unchanged float64 observed values), `observation_mask.npy` (uint8), `labels.npy`, `consumers.csv`, `calendar.csv`, `engineered_features.parquet`, `feature_names.json`, `splits.csv`, `monthly_coverage.parquet`, `insufficient_history.csv`, `measurement_issues.csv`, both shared preprocessors, and JSON/Markdown dataset reports. Standardized rows retain source order for traceability; model datasets use the shared randomized split order. Never feed a row index or identifier to a model.

The JSON report separates measurements from the original CSV from counts after sanitization/calendar insertion. It reports descriptive missingness by class, raw extremes, zero counts, insufficient histories, fitted parameters, undefined features, split distributions, output dimensions and validation. `measurement_issues.csv` contains every detected invalid, infinite, negative, or suspiciously large measurement with consumer ID, date and original text. An unusual observation does not establish theft.

## Splits, leakage and reproducibility

All variants use one stratified 70%/15%/15% train/validation/test assignment with `random_state=42`. IDs are sorted before randomized splitting, making assignments invariant to input row order, including the source's grouping by label. Integer rounding follows two scikit-learn splits: train vs held-out, then equally divided validation/test. Every consumer and their complete history belongs to exactly one split. `shared/splits.csv` contains `CONS_NO,FLAG,split` in the exact row order used by each output split.

Clipping, global sequence normalization, feature imputation, and feature scaling fit **only training consumers**. Training ID digests and fit counts are saved in the preprocessors and metadata. Labels and IDs never enter a transform or feature matrix. Per-consumer calendar statistics need no population fit. There is no resampling, window-level splitting, or class balancing. The dataset is imbalanced: evaluate class weighting or other imbalance strategies later during training while retaining untouched held-out sets.

The manifest records resolved configuration, input SHA256, source module hashes, dependency/Python versions, and a generation fingerprint. Keep the same source CSV, configuration, code and lockfile to reproduce the run. No random sampling is used to estimate clipping quantiles. Changing batch size affects memory/runtime rather than feature definitions. Full regeneration overwrites the named generated artifacts; use a separate `output_dir` to retain older experiments.

## Missing values, invalid values and outliers

Recorded zero is an observation; missing is not. The original usable observation mask is captured after explicit input sanitation and **before** clipping, interpolation, log transforms or scaling. Empty/NA cells and inserted calendar days have mask 0. Finite nonnegative measurements, including zero, have mask 1.

Default `invalid_policy: nan` converts nonnumeric and infinite values to missing and reports them. `error` aborts instead. Default `negative_policy: nan` also converts negatives to missing; alternatives are `error` or explicitly `clip_zero` (which treats the corrected reading as observed). Corrections are audited; the input CSV is unchanged. Raw finite min/max and anomaly counts are computed before these policies. Standardized arrays preserve finite observed values without outlier clipping; the original text is retained in the CSV/audit.

Tree datasets use available observations and retain undefined feature values as NaN. A fully missing consumer has undefined consumption summaries, observed count 0, missing ratio 1, and zero/missing flags set appropriately. A zero-only consumer has mean/std 0 but undefined ratios with zero denominators, skewness and correlations. No consumer is removed by default.

The global upper clipping threshold is the exact 99.9th percentile of finite observed **training** measurements (configurable). The same threshold defines per-consumer outlier count/ratio and clips neural sequences only. Neural consumption is clipped, transformed with `log1p`, and globally standardized using the population mean/std of observed transformed training measurements. Robust scaling (median/IQR) or no scaling can be selected. A degenerate scale becomes 1; no observed training values use center 0/scale 1 and no learned clipping threshold.

After normalization, missing consumption uses the numerical placeholder 0 and mask 0. A real zero follows the consumption transform and has mask 1. Even if normalized consumption happens to equal zero, the mask preserves this distinction. The absent September day is never presented as an observed zero.

Optional `cnn.imputation: short_linear` interpolates only bounded internal missing runs of at most `max_interpolation_gap` calendar days, in raw consumption space before clipping/log/scaling. It never fills long runs or either endpoint. Fitting normalization still uses only original training observations. The original mask remains unchanged, so interpolated values have mask 0. The default `placeholder` performs no interpolation.

Hybrid features use train-median imputation then train-fitted RobustScaler (configurable). Entirely undefined training columns stay in the schema and use zero as their imputation fallback. Constant columns use a deterministic nonzero scale. Binary missing indicators for **all** statistical columns are appended after scaling, so validation-only missingness is represented. Thus the default 80 statistical features become 160 hybrid inputs. Infinities are converted to NaN at this transform boundary; the feature extractor itself rejects infinite output.

## Engineered features

The default schema contains 80 numeric features. All consumption statistics, changes, trends and calendar summaries use **sanitized raw, unclipped measurements**. Only `outlier_count` and `outlier_ratio` use a training-fitted threshold; no tree features are globally scaled. Full ordered names are saved in `feature_names.json`.

| Group | Features / definitions |
|---|---|
| General (19) | observed_days; total, mean, median, population std, min, max; p10, p25, p75, p90, p95; IQR; std/mean; mean/median; max/median; population standardized third moment (skewness, at least three observations); outlier_count; outlier_count/observed_days |
| Temporal (19) | mean/median absolute daily change; population std of signed daily changes; maximum increase and decrease magnitudes (0 if valid pairs exist but no such movement); mean fractional daily change; sudden_drop_count/ratio; sudden_increase_count; max/mean zero-run lengths; recent_30d_mean; recent_90d_mean; recent_vs_historical_ratio; first_half_vs_second_half_ratio; monthly_trend_slope; Pearson autocorrelations at lags 1, 7, 30 |
| Calendar (22) | weekday_mean; weekend_mean; their ratio; winter/spring/summer/autumn means; monthly_mean_01 through monthly_mean_12; population std/range of those 12 monthly means; weekly_pattern_strength |
| Quality (14) | missing_days/ratio; zero_days/ratio; nonzero_days; longest_missing_streak; number_of_missing_streaks; first/last valid calendar index; valid_observation_span; observed_days_ratio; has_insufficient_history; all_observations_zero; all_observations_missing |
| Coverage (6) | recent_30d_observed_ratio; recent_90d_observed_ratio; winter/spring/summer/autumn observed ratios |

Definitions and edge cases:

- Daily changes require valid measurements at **both adjacent calendar days**. Missing gaps do not become daily differences. Fractional change is `(current-previous)/previous`, with strictly positive previous consumption. Sudden drops are changes <= -0.50, increases >= +0.50 by default; both thresholds are configurable. Drop ratio uses the number of eligible positive-previous adjacent pairs.
- Lag correlations use valid endpoint pairs at their exact calendar lag; intervening missing values are allowed. Fewer than two pairs or either endpoint series having zero variance yields NaN. All divisions with zero/undefined denominator yield NaN, never infinity.
- All std calculations use population `ddof=0`. Skewness is the unadjusted standardized third central moment. Percentiles use NumPy's default linear interpolation.
- Recent windows are the final 30/90 days of the full calendar, not the last observed measurements. `recent_vs_historical_ratio` divides recent-90 mean by the mean **before** that window. First/second half uses floor(calendar_length/2) for the first half. Empty windows yield NaN; coverage features show availability. For a configured calendar shorter than a window, that window covers the available calendar.
- Monthly trend fits an unweighted least-squares line through available year-month means, using elapsed calendar month positions. Missing months are not compressed in time. Fewer than two available months yields NaN.
- Month-of-year and seasonal means pool all available daily values across years, weighting by observed days. Monthly std/range use the available 12 month-of-year means with equal month weight. Weekends are Saturday/Sunday; seasons use northern meteorological months: DJF/MAM/JJA/SON. `weekly_pattern_strength` is variance of available weekday means / variance of all observed daily values, bounded to [0,1]; zero total variance yields NaN. It is descriptive, not a fitted seasonality model.
- Coverage differences can influence calendar means. Inspect the four seasonal coverage ratios and per-consumer **year-month observed counts** in `shared/monthly_coverage.parquet` before interpreting seasonal differences; missing months remain undefined, never zero-filled.
- Zero ratio divides by **observed** days; missing and observed ratios divide by the complete calendar. Missing measurements break zero runs. No zero/missing runs have length/count 0. First/last valid indices are zero-based **day positions**, not consumer row indices. Span includes internal gaps and is NaN for a fully missing history. Fully missing is not classified as all-zero.

## Optional experiments

Set `features.create_quality_ablation: true` to additionally write tree datasets under `ablations/without_quality/{lightgbm,xgboost}` with explicit quality/coverage fields (including `observed_days`) removed. Default splits and remaining values are identical. This tests reliance on explicit availability indicators; other statistics can still indirectly reflect coverage. Set `include_quality_features: false` to use that reduced feature schema throughout a separate run.

Set `filtering.filter_insufficient_history: true` for an **additional** population with at least `features.insufficient_history_threshold` observations (default 180). Outputs and newly stratified splits go under `sensitivity/min_180_observed_days/`; all fitted parameters are relearned from that population's training split. The complete population is still generated first. The filtered experiment is separately labeled and never replaces the main population or its evaluation coverage. Both options are disabled by default. Too few retained consumers for stratification fails explicitly.

## Loading prepared data

LightGBM and XGBoost use identical, unscaled tables and feature order:

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path("data/processed")
for variant in ("lightgbm", "xgboost"):
    names = json.loads((root / variant / "feature_names.json").read_text())
    train = pd.read_parquet(root / variant / "train.parquet")
    X = train[names]               # excludes CONS_NO and FLAG explicitly
    y = train["FLAG"].to_numpy()
    ids = train["CONS_NO"].to_numpy()
```

Standalone CNN arrays already contain consumption and mask channels:

```python
X = np.load(root / "cnn1d/X_train.npy", mmap_mode="r", allow_pickle=False)
y = np.load(root / "cnn1d/y_train.npy", allow_pickle=False)
ids = pd.read_csv(root / "cnn1d/train_ids.csv", dtype={"CONS_NO": str})
assert X.shape == (len(y), 1035, 2)
# Future PyTorch training code (torch is optional and not installed here):
# import torch
# batch = torch.from_numpy(np.array(X[:32], copy=True)).permute(0, 2, 1)
# batch shape: (32, 2, 1035), ready for Conv1d; no model is defined here.
```

Hybrid arrays share one ID file and target per split. Verify their common order against the tree table:

```python
sequence = np.load(root / "hybrid/X_sequence_train.npy", mmap_mode="r", allow_pickle=False)
stats = np.load(root / "hybrid/X_features_train.npy", mmap_mode="r", allow_pickle=False)
y = np.load(root / "hybrid/y_train.npy", allow_pickle=False)
ids = pd.read_csv(root / "hybrid/train_ids.csv", dtype={"CONS_NO": str}).CONS_NO
tree = pd.read_parquet(root / "lightgbm/train.parquet")
assert len(sequence) == len(stats) == len(y) == len(ids)
np.testing.assert_array_equal(ids.to_numpy(), tree.CONS_NO.to_numpy())
np.testing.assert_array_equal(y, tree.FLAG.to_numpy())
np.testing.assert_array_equal(sequence, np.load(root / "cnn1d/X_train.npy", mmap_mode="r"))

# Confirm each statistical row was transformed from that same consumer:
import joblib
pre = joblib.load(root / "hybrid/feature_preprocessor.joblib")
np.testing.assert_array_equal(stats, pre.transform(tree[pre.input_names_]))
```

Run Python from the project root so saved `src.*` preprocessing classes remain importable. Use joblib artifacts only from trusted sources. Full artifact validation reloads and recomputes every neural row, proves alignment to source IDs/labels, verifies tree schemas/values and masks, and checks training-fit provenance. Synthetic tests additionally prove held-out extreme values cannot modify fitted clipping, imputation or scaling parameters and detect deliberately shuffled hybrid rows.

## Verified local run

The preprocessing pipeline was executed on the supplied `data.csv` on November 7, 2026. Its original 35 tests passed. Both the integrated artifact validator and the standalone validation command passed for every one of the 42,372 consumers. No classifiers were trained by the preprocessing command. The expanded suite, including training, evaluation and plotting, now contains 50 tests; all passed in the final run, recorded in `test-results.xml`.

## Training, evaluation and comparison

Install the expanded requirements in the existing environment; processed data does not need regeneration:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.training.train --all --smoke --evaluate --visualize
.\.venv\Scripts\python.exe -m src.training.train --all --evaluate --visualize
```

With the environment activated, Windows and Linux use the same commands:

```bash
python -m src.training.train --model lightgbm
python -m src.training.train --model xgboost
python -m src.training.train --model cnn1d
python -m src.training.train --model hybrid
python -m src.training.train --all --evaluate --visualize
python -m src.evaluation.evaluator --all
python -m src.visualization.comparison --all
python -m src.visualization.dashboard
python -m src.evaluation.validation
```

Use `--config configs/training.yaml` explicitly when desired. Configuration paths resolve relative to the YAML file, with the default YAML under `configs/`; commands otherwise locate it relative to the package, rather than assuming the process working directory. Existing preprocessed data is loaded and audited without altering it. Missing files produce instructions to run the preprocessing command.

The actual input schemas are discovered from saved files: 80 tree features, `(N,1035,2)` sequences, and 160 hybrid statistical inputs. IDs and labels are verified against shared assignments. Different orders are explicitly aligned by `CONS_NO`; hybrid branches are independently recomputed from their shared source consumers before training. Arrays are memory mapped, and the PyTorch Dataset converts sequences to `(2,1035)` without including identifiers in predictive tensors.

### Models and selection

- **LightGBM:** 1,500 maximum rounds, learning rate 0.03, 31 leaves, configured regularization and validation early stopping with 100-round patience. Uses the official sklearn wrapper; detects the new `eval_X`/`eval_y` API while supporting older `eval_set` versions.
- **XGBoost:** 1,500 maximum rounds, histogram trees, depth 5, learning rate 0.03, configured regularization and validation early stopping. Its native `aucpr` monitoring metric is not mislabeled as sklearn Average Precision. Candidate selection and optional tuning use actual sklearn AP.
- **CNN:** three Conv1d/BatchNorm/ReLU blocks, channels 32/64/128 and kernels 7/5/3, max pooling after the first two, adaptive average pooling and a 64-unit classification head. Training uses logits and BCEWithLogitsLoss; inference applies sigmoid.
- **Hybrid:** an independently initialized CNN encoder plus a statistical branch `F→128→64`, concatenated into a `192→64→1` fusion head with default encoder width 128. The feature count is discovered. Statistical BatchNorm uses stored running statistics for a singleton training batch, retaining the last consumer rather than dropping it.

Neural defaults: AdamW, learning rate 0.001, weight decay 0.0001, batch size 128, at most 100 epochs, AP-based early stopping patience 12 and minimum improvement 0.0001, ReduceLROnPlateau, dropout 0.3 and gradient clipping 1.0. The best checkpoint, not the final epoch, is evaluated. Training loaders shuffle; validation/test do not. CPU is supported, CUDA/MPS are auto-detected, and CUDA mixed precision is enabled only on compatible CUDA runs. This Windows installation uses a CPU PyTorch wheel and falls back to CPU on its Intel Arc hardware.

Tree runs compare unweighted and weighted candidates by default. The positive weight is `training_normal/training_theft`; validation/test class counts never determine it. Neural `weighted_loss: true` uses the same train-only ratio. Set it to false for an independently tracked unweighted baseline. `compare_class_weights`, architecture widths, epochs, patience, thresholds and other settings are centralized in `configs/training.yaml`.

Enable optional Optuna tuning with `tuning_enabled: true`. Trial count and each search range are configurable under the model's `tuning_space`; each range is `[int|float, lower, upper, logarithmic]`. TPE uses the experiment seed. Tuning sees train/validation only. Default tuning is disabled. To run multiple independently archived seeds:

```bash
python -m src.training.train --all --evaluate --visualize --seeds 42,7,123
```

All requested models finish training and validation-only threshold selection **before any test inference**. The manifest records `all_selections_frozen`; evaluation refuses an incomplete selection. Static dataset-integrity inspection may read test identities/labels, but never optimizes or scores a classifier during selection. The comparison preselects the best model by validation AP, then reports its test results alongside the other frozen models.

Threshold policies use a predefined 0.01–0.99 grid. Default maximizes validation F1; ties favor precision, then higher threshold. `recall_at_precision` maximizes recall subject to the configured validation precision floor. Infeasibility explicitly leaves the optimized threshold unavailable instead of switching objectives. Fixed 0.5 results are always separate. Class weighting can alter calibration; reported risk scores are not automatically calibrated probabilities or proof of theft.

### Saved training outputs

Every run has a UTC timestamp/configuration hash/unique suffix. Previous trained experiments are retained:

```text
experiments/<model>/<run_id>/
  model.joblib + model.txt/model.ubj   # trees; selected checkpoint
  best.pt                            # neural best checkpoint
  history.csv, feature_names.json, configuration.json
  training_result.json, validation_metrics.json, test_metrics.json
  threshold.json, threshold_grid.csv
  weighted/, unweighted/              # tree candidate configurations/histories
outputs/runs/<run_id>/
  run.json, alignment_audit.json, visualization_manifest.json
  predictions/, metrics/, figures/, dashboard/, reports/
```

The latest full run is published to `outputs/predictions`, `outputs/metrics`, `outputs/figures`, `outputs/dashboard` and `outputs/reports`. `outputs/latest_run.json` points to its archive; the append/update registry is `outputs/metrics/experiments.csv`. Re-evaluation updates metrics for the same run without retraining or duplicating registry rows. Canonical latest-result files are replaceable views; trained experiment directories and unique run archives preserve prior runs. Use `--run-dir outputs/runs/<run_id>` with evaluation/plot/dashboard commands to select an older run.

Smoke mode uses at most 512 training consumers, 128 validation consumers for selection, and another 128 validation consumers for diagnostic inference, with eight tree rounds and two neural epochs. When possible the latter validation subset excludes consumers used in diagnostic selection; tiny synthetic fixtures can reuse validation records. It saves to `experiments/smoke` and `outputs/smoke/<run_id>`, is marked **diagnostic**, and never updates the full-run pointer or main comparison. Its diagnostic output filenames retain the `_test` suffix for the shared output contract, but metadata labels their source `diagnostic_validation`. Smoke scores do not measure full-population performance. The first implementation smoke run used 128 final-test consumers before this safeguard was added; no full-training settings were changed using its scores. This development exposure is disclosed in the comparison report, and the original diagnostic archive is retained.

Prediction CSVs include original IDs, true labels, continuous risk scores, fixed and optimized binary predictions, and frozen thresholds. Combined predictions explicitly join identities and verify labels. Training histories, per-candidate weighting outcomes, feature gain importance, checkpoints, environment versions, seed, input fingerprint, source module hashes, hardware, runtimes and peak CUDA memory (if available) are retained.

### Metrics, figures and dashboard

The comparison CSV/JSON contains sklearn Average Precision, ROC-AUC, accuracy, balanced accuracy, precision, recall, F1, F2, MCC, specificity, FPR, FNR, Brier score and TP/TN/FP/FN. Fixed-threshold values use the `fixed_05_` prefix. Undefined metrics are explicit nulls, not fabricated zero performance. Precision@K and Recall@K use `ceil(K% × N)` on the full test population with ID-only tie breaking.

Static figures are saved at 300 DPI PNG and as SVG by default. They include ROC/PR curves, raw and row-normalized confusion matrices, main metrics, error counts/rates, validation threshold sweeps, raw score distributions, learning histories, top-20 gain importance, runtime, top-K detection, agreement/disagreement, baseline differences in percentage points, and representative raw-consumption histories. LightGBM uses total split gain; XGBoost uses mean split gain. These definitions are kept distinct. No SHAP or gain result is presented as causal evidence. `python -m src.evaluation.validation` independently reloads checkpoints, recomputes metrics from prediction CSVs, checks threshold selection from validation scores, verifies consumer alignment and confirms required figures and the embedded dashboard exist.

`outputs/dashboard/model_comparison.html` embeds Plotly and works without a backend or CDN. It includes interactive comparisons, histories, thresholds, gain importance and a searchable, filterable table limited to the first configured number of IDs in lexical order. Full predictions remain in CSV. `outputs/reports/model_comparison.md` contains actual results, selection rules, computation costs and limitations.

Error analysis compares false positives with correct normal consumers, and false negatives with correctly detected theft consumers, using missingness, zeros, variability, extreme consumption, history length and temporal-change features. It saves group medians and descriptive median differences divided by within-class IQR. TP/TN/FP/FN examples are nearest the within-group median score for the validation-preselected reference model; disagreement cases have the largest cross-model score range, with ID tie breaking. Missing measurements remain gaps in original-consumption plots. No theft dates are inferred.

Optional `bootstrap_enabled: true` estimates 95% percentile intervals for AP, ROC-AUC, F1 and recall, including specified paired model differences. Stratified consumer indices are resampled identically across models; no model is retrained. Intervals describe frozen test predictions and must not drive post-test selection or automatic significance claims.

Tests cover synthetic metrics, confusion ordering, threshold feasibility, class weighting, input shapes, tensor order, ID realignment, shuffled-branch rejection, checkpoints, repeatable predictions, paired bootstrap, no test fitting, actual-data alignment when available, and synthetic end-to-end graph/dashboard generation.

Primary implementation references: [LightGBM Python introduction](https://lightgbm.readthedocs.io/en/stable/Python-Intro.html), [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html), [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

| Split | Consumers | Normal (0) | Theft (1) | Theft percentage |
|---|---:|---:|---:|---:|
| Train | 29,660 | 27,130 | 2,530 | 8.5300% |
| Validation | 6,356 | 5,813 | 543 | 8.5431% |
| Test | 6,356 | 5,814 | 542 | 8.5274% |

| Predictive input | Train shape | Validation shape | Test shape | Dtype |
|---|---|---|---|---|
| LightGBM / XGBoost | (29660, 80) | (6356, 80) | (6356, 80) | float64, NaN retained |
| CNN / hybrid sequences | (29660, 1035, 2) | (6356, 1035, 2) | (6356, 1035, 2) | float32 |
| Hybrid statistical branch | (29660, 160) | (6356, 160) | (6356, 160) | float32 |

Targets have shapes `(29660,)`, `(6356,)`, `(6356,)`, with dtype int8. Tree Parquet tables additionally contain `CONS_NO` and `FLAG` (82 total columns); their predictive inputs have 80 columns. The standardized source matrix is `(42372, 1035)` float64. No engineered columns were discarded or entirely undefined in training.

Measured quality findings:

- The original 1,034 daily columns contain 11,233,528 missing cells. Inserting September 18, 2016 adds 42,372 missing cells: 11,275,900 total, or 25.7118% of the complete calendar matrix.
- There are no invalid numeric, negative, or infinite measurements. Six values exceed the reporting cutoff of 100,000; the maximum is 800,003.32. They remain unchanged in the standardized data and raw statistical features.
- The train-only 99.9th-percentile clipping threshold is **370.8**, based on 22,808,105 observed training measurements. Neural sequence normalization has center 1.5832523055587897 and scale 1.1207233035092221 after clipping/log1p. These same parameters apply to validation and test.
- All 157 consumers with fewer than 180 observations, including five fully missing histories, were retained. There are 2,109 observed all-zero consumers and 5,788,603 observed zero measurements. Their edge cases were handled explicitly.
- Measurement availability differs by class: 25.1686% missing for normal consumers and 31.5349% for theft-labeled consumers after calendar insertion. These are descriptive figures, not preprocessing inputs or evidence that missingness establishes theft.

See [the JSON report](data/processed/shared/dataset_report.json), [the Markdown report](data/processed/shared/dataset_report.md), and [the manifest](data/processed/shared/run_manifest.json) for the exact measured counts, feature names, parameters and reproducibility record. Optional ablation and filtered outputs were tested on synthetic data and remain disabled for this complete-population run.

## Verified full training results

Run `20261107T200402624046Z_0a32ed6a_8d3773` trained all four models on 29,660 consumers and evaluated the frozen checkpoints on the same 6,356 test consumers. XGBoost was preselected by validation AP (0.5254), before full test inference. The tree weighting comparison selected unweighted candidates. Neural losses used train-derived positive weighting. No Optuna search or bootstrap was enabled for this initial run.

| Model | Test AP | ROC-AUC | Precision | Recall | F1 | FP | FN | Frozen threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.5289 | 0.8746 | 0.4928 | 0.5074 | 0.5000 | 283 | 267 | 0.19 |
| XGBoost | 0.5328 | 0.8747 | 0.4871 | 0.5221 | 0.5040 | 298 | 259 | 0.18 |
| CNN 1D | 0.4272 | 0.8381 | 0.4592 | 0.4262 | 0.4421 | 272 | 311 | 0.77 |
| Hybrid | 0.4079 | 0.8263 | 0.4436 | 0.4207 | 0.4318 | 286 | 314 | 0.77 |

CNN stopped after 91 epochs and selected epoch 79; hybrid stopped after 38 and selected epoch 26. Tree selected rounds were 506 for LightGBM and 1,123 for XGBoost. In this initial configuration and seed, hybrid AP was lower than the standalone CNN and both tree models. This is an observed comparison, not a claim that all possible hybrid configurations are inferior.

All 50 tests passed. Completed-run verification independently reproduced saved metrics, thresholds, labels, consumer ordering and binary predictions, and loaded every checkpoint. There are 46 PNG and 46 SVG figures, including 25 raw-consumption example plots. The standalone dashboard is approximately 6.5 MB and its 20 embedded JavaScript blocks passed syntax checks. Automated native-browser rendering could not be checked because the browser access host was unavailable.

Development limitation: the first diagnostic implementation evaluated 128 final-test consumers before full training. No predefined model settings or selection rules were changed using those results. Future smoke inference uses validation consumers instead. The initial exposure remains disclosed, so this test population should not be described as completely blind during development.

Full results: [comparison report](outputs/reports/model_comparison.md), [metrics CSV](outputs/metrics/model_comparison.csv), [interactive dashboard](outputs/dashboard/model_comparison.html). Each trained checkpoint is under `experiments/<model>/20261107T200402624046Z_0a32ed6a_8d3773/`.
