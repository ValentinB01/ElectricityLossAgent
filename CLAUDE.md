# Implementation guide: maintenance planning agent (wind + PV)

> **For humans:** give the coding agent **one stage at a time** (section 8). After each stage, check its acceptance criteria before moving on.
>
> **For the agent:** read this whole document once. Then work strictly on the requested stage. Follow the rules in section 2 without exception.

Project for **Smarthack XI 2026, PPC theme "24h to Power PPC"**. Guide version: 2026-11-07.

**Where things live (each topic has one owner):**

| Document | Owns |
|---|---|
| `CLAUDE.md` (this file) | Build rules, stages, data schema, module specs, download links |
| `README.md` | Public overview, how to run, results, dataset licences and citations, limitations, team |
| `docs/design.md` | Why the system is designed this way; the consumers module design |
| `docs/producers-plan.md` | Current step-by-step plan for the producers module |

**Scope:** the project has two loss modules under one agent. This guide specifies the **producers module** (PPC Renewables) and the shared agent layer. The consumers module (Rețele Electrice) has a different owner and is designed in `docs/design.md`.

---

## 1. Context and objective

### 1.1 What we are building

An agent that **plans maintenance interventions** for a portfolio of wind turbines and PV parks. For each job, the agent picks the day with the lowest loss. What matters:

- **weather:** for turbines, calm days; for panels, hours without sun or cloudy days;
- **periodic revisions** (scheduled service), with their tolerance window;
- **detected underperformance**, meaning equipment that runs but produces below normal;
- **optionally, failure risk.**

The agent produces a **work order calendar**, **alerts** and **explanations**. Every decision must be traceable to the model prediction that produced it.

### 1.2 PPC context (for the demo)

- PPC Renewables Romania operates the **Fântânele-Cogealac** wind farm: 600 MW, 240 GE 2.5xl turbines, approx. coordinates **44.59028 N, 28.56528 E** (Constanța).
- **Lumina PV park, Călugăreni** (Giurgiu): 63 MW, approx. 120,000 bifacial modules, approximate coordinates **44.0951 N, 25.9761 E**.
- PPC's real data is not public. We train on public data (UK, Portugal, India, USA) and **demonstrate** on real weather at the coordinates of PPC's parks. We say so openly in the README and the presentation.

### 1.3 Scoring (from the official brief)

| Criterion | Points | What the jury looks for |
|---|---|---|
| Idea and relevance to PPC | 25 | The problem is real and credible for PPC colleagues |
| ML quality | 25 | Data preparation, separate test set, comparison with a baseline, error analysis |
| Robustness | 10 | Missing data, rare cases, bad inputs, uncertainty |
| The agent | 25 | Complete automatic flow data → model → decision → action, justified decision layer |
| Explainability | 10 | The agent shows why it made the decision |
| Demo and repository | 5 | Reproducible README, clean code, commits along the way |

### 1.4 Mandatory requirements of the brief

1. At least one ML model **trained by the team**, evaluated on a **separate test set** and compared with a **simple baseline**. The training code lives in the repository. The model **runs locally**, not through an external API.
2. An LLM is **not enough**. It may coordinate the agent or the interface, but the agent must rely on our models.
3. The agent **takes actions**: it prioritises, plans, alerts.
4. Only public, synthetic or team-annotated data, **with sources stated**. Public repository, **commits throughout the 24 hours**.

---

## 2. Rules for the agent (mandatory)

1. **Do not invent columns, units or file formats.** After each download, inspect the files (`head`, `df.columns`, `df.describe()`, the README in the archive) and write what you found to `docs/data_notes.md`. Column names in this guide are **hints to be verified**.
2. **Data does not go into git.** `data/` and `models/` are in `.gitignore`. Download scripts recreate them.
3. **Temporal split, never random,** for time series. The test set is chosen at the start and not touched until final evaluation.
4. **Fixed seed** (`SEED = 42`) everywhere: numpy, lightgbm, torch, fault injection.
5. **Physical units, not abstract scores,** at model outputs: kW, MWh/day, calibrated probabilities. The optimizer needs quantities that can be added.
6. **The LLM does not decide.** It does not compute costs or choose days. It only calls tools (`optimize_maintenance` etc.) and explains what they returned.
7. **Commit after every functional sub-step**, with clear messages (`feat(m1): quantile LightGBM + binned baseline`).
8. **Every script runs end to end** through a `Makefile` command and writes results to `results/` (JSON + PNG).
9. **Every numeric assumption** (failure cost, wind limit, job duration) lives in `config/params.yaml`, with a comment saying it is an assumption and where the real value would come from.
10. When unsure about a data detail, **stop and report** instead of guessing.

---

## 3. Architecture

Three layers, each with its own job:

```
LAYER 1 — ML MODELS (predict physical quantities, with uncertainty)
  M1 Expected production -> expected kW (quantiles 10/50/90%)
  M2 Underperformance    -> probable cause + MWh lost/day + confidence
  M3 Failure risk (opt.) -> failure probability over a 1..14 day horizon
  M4 Downtime cost       -> MWh (or RON) lost for each hour of the next 7-10 days
                            (= M1 applied to the weather forecast)

LAYER 2 — DECISION (deterministic, reproducible)
  Optimizer: jobs (revisions, repairs, checks) + constraints (crews, wind, windows)
             -> calendar that minimises total cost

LAYER 3 — LLM AGENT (coordination + interface)
  Calls tools, runs what-if scenarios, turns field reports into structured data,
  explains decisions. Does not decide on its own.
```

The agent's daily flow:

```
SCADA/inverter data ─► M1 ─► M2 ─┐
weather forecast ────► M1 ─► M4 ─┼─► Optimizer ─► calendar + alerts + explanations
SCADA temperatures ──► M3 ───────┤
revisions + crews ───────────────┘
```

---

## 4. Stack and setup

- **Python 3.11.** Compatible with all libraries below; OpenOA requires ≤ 3.11, EnergyFaultDetector requires 3.10–3.12.
- Minimal `requirements.txt`:

```
pandas>=2.1
pyarrow
numpy
scikit-learn
lightgbm
shap
matplotlib
pvlib
requests
pyyaml
tqdm
ortools          # optional: exact optimisation (CP-SAT)
fastapi
uvicorn
streamlit        # demo interface
kaggle           # PV dataset download (needs a Kaggle token)
gdown            # PVF-10 download from Google Drive (optional)
boto3            # PVDAQ (optional)
```

- Optional, in a separate environment (heavy dependencies):
  - `energy-fault-detector`: baseline for M3; requires TensorFlow ≥ 2.15;
  - `openoa`: power curves, losses;
  - `torch` + `torchvision`: PV classifier on thermal images.

---

## 5. Repository structure

```
.
├── CLAUDE.md                  # this guide
├── README.md                  # how to run, results, sources, limitations
├── LICENSE                    # MIT
├── Makefile                   # setup, data, train, eval, agent, demo, test
├── requirements.txt
├── config/
│   ├── params.yaml            # all numeric assumptions
│   ├── sites.yaml             # park coordinates (Kelmarsh, Fântânele, Călugăreni, ...)
│   └── agent.yaml             # LLM provider, confidence thresholds
├── data/                      # (gitignored) raw/ interim/ processed/
├── models/                    # (gitignored) saved models
├── results/                   # JSON metrics + PNG charts + work orders (committed)
├── docs/
│   ├── design.md              # design rationale
│   ├── producers-plan.md      # current plan
│   ├── data_notes.md          # what data inspection found
│   ├── decisions.md           # why we chose thresholds/parameters; error analysis
│   └── img/
├── src/
│   ├── common.py              # paths, seed, shared thresholds, chart style
│   ├── data/
│   │   └── prepare.py         # downloads + preparation: Kelmarsh, Kaggle solar, Open-Meteo history
│   ├── producers/
│   │   ├── expected_power.py  # M1: expected production + baseline
│   │   ├── underperformance.py # M2: performance index, CUSUM, cause classifier
│   │   ├── failure_risk.py    # M3 (optional): NBM on temperatures (CARE) or alarm prediction (HoT)
│   │   ├── downtime_cost.py   # M4: weather-only M1-W + cost map from forecast
│   │   └── solar.py           # PV branch: M1, M2, M4 on the Kaggle data
│   ├── consumers/             # C1, C2, inspection queue (see docs/design.md)
│   ├── simulate/
│   │   └── inject_faults.py   # labelled synthetic faults (wind + PV)
│   ├── decision/
│   │   ├── optimizer.py       # job dataclasses + greedy (optional CP-SAT)
│   │   └── backtest.py        # replay of one year, 3 policies
│   ├── agent/
│   │   ├── tools.py           # functions exposed to the LLM
│   │   ├── llm_agent.py       # tool-calling orchestration
│   │   └── api.py             # FastAPI
│   └── app/
│       └── dashboard.py       # Streamlit
└── tests/                     # small tests for data schema and optimizer
```

---

## 6. Data sources: what to download and from where

> **MVP rule:** download only rows marked **MVP** first. The rest only if time allows. Wind datasets are gigabytes, so start the download in the first hour.

Licences and formal citations are in `README.md` sections 5 and 12.

### 6.1 Wind

| Dataset | Role | What to download | Size |
|---|---|---|---|
| **Kelmarsh** (Cubico), Zenodo `16807551` — **MVP** | M1, M2, backtest | `Kelmarsh_WT_static.csv`, `Kelmarsh_WT_dataSignalMapping.csv`, **one SCADA year**, e.g. `Kelmarsh_SCADA_2019_3085.zip` (+ `Kelmarsh_SCADA_2020_3086.zip` for the backtest) | 311 MB / 474 MB |
| **CARE to Compare** (Fraunhofer IEE), Zenodo `15846963` (v6) | M3 | `CARE_To_Compare.zip` | 5.5 GB |
| **Hill of Towie** (RES), Zenodo concept `14870023` (use the latest version) | M3 alternative (alarms), M2 validation (upgrades with known dates) | `Hill_of_Towie_turbine_metadata.csv`, `*_description.csv`, `Hill_of_Towie_AeroUp_install_dates.csv`, `Hill_of_Towie_ShutdownDuration.zip`, **one year**, e.g. `2023.zip` | approx. 1.4–1.6 GB/year |
| Penmanshiel (Cubico), Zenodo `16807304` | M1 reserve | one SCADA year | hundreds of MB |

**Exact links:**

```
# Kelmarsh (v4)
https://zenodo.org/records/16807551/files/Kelmarsh_WT_static.csv?download=1
https://zenodo.org/records/16807551/files/Kelmarsh_WT_dataSignalMapping.csv?download=1
https://zenodo.org/records/16807551/files/Kelmarsh_SCADA_2019_3085.zip?download=1
https://zenodo.org/records/16807551/files/Kelmarsh_SCADA_2020_3086.zip?download=1

# CARE to Compare (v6), MD5 2547b58c21ac8c242d13232860cf500c
https://zenodo.org/records/15846963/files/CARE_To_Compare.zip?download=1

# Hill of Towie: resolve the latest version ID first
https://zenodo.org/api/records/14870023/versions/latest    # JSON: id + file links
# then: https://zenodo.org/records/<ID_LATEST>/files/2023.zip?download=1
```

**Data notes (to verify on inspection):**

- **Kelmarsh:** 6 Senvion MM92 turbines of 2,050 kW, 10-minute SCADA + events. The annual archives contain separate CSVs per turbine for data and for status/events. Use `Kelmarsh_WT_dataSignalMapping.csv` to map signal names. Not all signals exist for the whole period.
- **CARE to Compare:**
  - 95 datasets, 36 turbines, 3 farms. Farm A: 5 onshore turbines in Portugal, 86 signals.
  - Each dataset has approx. one year of `train` rows and a `prediction` period of 4–98 days containing the event.
  - Descriptive columns: id, timestamp, asset id, `train_test`, status (`status_type_id`). Signals are anonymised (`sensor_*`); only power, reactive power and wind have recognisable names.
  - Sensor descriptions and `event_info.csv` (with `asset_id`) are in the archive; **read the README in the zip.**
  - `status_type_id` values: 0 normal, 1 derated, 2 idling, 3 service, 4 downtime/fault, 5 other states. Treat only 0 and 2 as "normal".
  - For farm A, statuses come from the fault log (14 days before a fault marked 4, 3 days after marked 3). Use them **only to filter training data, not for evaluation.**
  - For farm B, Min/Max/Std statistics are unreliable: use only the means.
- **Hill of Towie:**
  - 21 Siemens SWT-2.3-VS-82 turbines, 10-minute SCADA + alarm log, UTC timestamps (marking the end of the interval).
  - `ShutdownDuration`: seconds of downtime per 10-minute interval (0–600).
  - Upgrades with known dates: AeroUp on T13 (2021-09-29), then the remaining turbines (2022-07-14 to 2023-05-26); TuneUp on test turbines from 2024-03-14.

### 6.2 Photovoltaic

| Dataset | Role | What to download | Size |
|---|---|---|---|
| **Solar Power Generation Data** (Kaggle `anikannal/solar-power-generation-data`) — **MVP** | M1, M2 PV | 4 CSVs: `Plant_1_Generation_Data.csv`, `Plant_1_Weather_Sensor_Data.csv`, `Plant_2_Generation_Data.csv`, `Plant_2_Weather_Sensor_Data.csv` | approx. 2 MB |
| PVDAQ (NREL), public S3 bucket `oedi-data-lake`, prefix `pvdaq/` | M1 PV on long series (optional) | list of systems + one system | varies |
| PVF-10 | Classifier on thermal images (optional) | `PVF-10.zip` | – |
| InfraredSolarModules (Raptor Maps) | Image pre-training (optional) | `2020-02-14_InfraredSolarModules.zip` | – |

**Commands:**

```bash
# Kaggle (needs ~/.kaggle/kaggle.json, generated from Kaggle account -> Settings -> API)
kaggle datasets download -d anikannal/solar-power-generation-data -p data/raw/pv_kaggle --unzip

# PVDAQ: list of systems (public CSV)
curl -o data/raw/pvdaq/systems.csv https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/systems_20250729.csv
# anonymous listing
aws s3 ls s3://oedi-data-lake/pvdaq/csv/ --no-sign-request
# or: pip install pvdaq_access (downloads daily files for a system id)

# PVF-10 (repo: github.com/wangbobby1026/PVF-Dataset; archive on Google Drive)
gdown 1SQq0hETXi8I3Kdq9tDAEVyZgIsRCbOah -O data/raw/pvf10/PVF-10.zip

# InfraredSolarModules (MIT)
# https://github.com/RaptorMaps/InfraredSolarModules  -> 2020-02-14_InfraredSolarModules.zip
```

**Data notes:**

- **Kaggle solar:**
  - 2 plants in India, 34 days, at 15 minutes.
  - Production columns: `DATE_TIME, PLANT_ID, SOURCE_KEY, DC_POWER, AC_POWER, DAILY_YIELD, TOTAL_YIELD`, where `SOURCE_KEY` = the inverter, approx. 22 per plant.
  - Sensor columns: `AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION`.
  - **Check the irradiance units** and the DC/AC ratio for each plant before modelling. Some days have missing data: document them.
- **PVF-10:**
  - 5,579 cropped thermal images, from 8 plants, 10 classes (5 repairable, 5 non-repairable), with a fixed train/test split in the archive.
  - **Take the class → repairable/non-repairable mapping from the README or the paper; do not guess it.**

### 6.3 Weather (Open-Meteo)

- **History (for training and backtest):** `https://archive-api.open-meteo.com/v1/archive`
- **Forecast (for the live demo):** `https://api.open-meteo.com/v1/forecast`
- Useful hourly variables:
  - wind: `wind_speed_10m`, `wind_speed_100m` (in the archive), `wind_direction_100m`, `wind_gusts_10m`;
  - atmosphere: `temperature_2m`, `surface_pressure`, `relative_humidity_2m`, `precipitation`, `cloud_cover`;
  - radiation: `shortwave_radiation`, `direct_normal_irradiance`, `diffuse_radiation`.
- Use `wind_speed_unit=ms` and `timezone=UTC`. For the forecast, check the documentation for which heights wind is available at (e.g. 80 m, 120 m) and extrapolate to hub height.
- **Terms:** free for non-commercial use, under 10,000 calls per day (5,000/hour, 600/minute), data under CC BY 4.0. Put the notice "Weather data by Open-Meteo.com" in the README and the interface. **Use a local cache** (`data/raw/weather/*.parquet`) so calls are not repeated.

Example:

```
https://archive-api.open-meteo.com/v1/archive?latitude=52.4006&longitude=-0.9471
  &start_date=2019-01-01&end_date=2020-12-31
  &hourly=wind_speed_100m,wind_direction_100m,temperature_2m,surface_pressure,precipitation
  &wind_speed_unit=ms&timezone=UTC
```

### 6.4 `config/sites.yaml`

```yaml
kelmarsh:      {lat: 52.4006,  lon: -0.9471, type: wind, note: "coordinates from Kelmarsh_WT_static.csv"}
fantanele:     {lat: 44.59028, lon: 28.56528, type: wind, note: "PPC, 240 x GE 2.5xl, demo"}
calugareni:    {lat: 44.0951,  lon: 25.9761,  type: pv,   note: "PPC Lumina 63 MW, approximate coordinates, demo"}
```

---

## 7. Canonical data schema (Parquet in `data/processed/`)

**`wind_scada.parquet`** (10 minutes):

| column | type | unit | notes |
|---|---|---|---|
| `ts_utc` | datetime64[ns, UTC] | – | end of the interval |
| `site_id`, `asset_id` | str | – | |
| `power_kw` | float | kW | 10-minute mean |
| `wind_ms`, `wind_std_ms` | float | m/s | nacelle anemometer |
| `wind_dir_deg`, `nacelle_dir_deg` | float | ° | if available |
| `pitch_deg`, `rotor_rpm` | float | | if available |
| `ambient_temp_c` | float | °C | |
| `temp_*_c` | float | °C | bearings, oil, generator (for M3) |
| `status_normal` | bool | – | derived from status/events |
| `rated_kw` | float | kW | from metadata |

**`pv_inverter.parquet`** (15 minutes): `ts_utc, site_id, inverter_id, ac_kw, dc_kw, irradiance_wm2, module_temp_c, ambient_temp_c, rated_kw`.

**`weather_hourly.parquet`**: `ts_utc, site_id` + the Open-Meteo variables.

**`tasks.json`** (optimizer input): see 9.6.

---

## 8. Stages (24 hours)

Each stage has **acceptance criteria**. Do not move on until they are met. Who owns what: `README.md` section 11.

| Stage | Hours | Delivers | Acceptance criteria |
|---|---|---|---|
| **E0 Setup** | 0–1 | Repository, `Makefile`, `requirements.txt`, `.gitignore`, `config/*.yaml`, MVP downloads started | `make setup` works; first commit |
| **E1 Wind data** | 1–4 | `download.py`, `wind_ingest.py`, `weather.py`, `docs/data_notes.md` | `wind_scada.parquet` with 6 turbines × 1–2 years; missing-value percentage reported; Kelmarsh weather cached |
| **E2 M1 wind** | 4–7 | Expected production (quantiles) + binned-curve baseline | `results/m1_wind.json` with test MAE/RMSE for model and baseline; 10–90% interval coverage; real vs. expected curve chart |
| **E3 M2 wind** | 7–11 | `inject_faults.py` + performance index + CUSUM + cause classifier | `results/m2_wind.json`: detection rate, mean delay (days), false alarms per turbine-month, F1 by cause; confusion matrix |
| **E4 M4 + optimizer + backtest** | 11–15 | Cost map, greedy, one-year replay | `results/backtest.json` + chart: MWh lost by agent vs. "repair immediately" vs. "repair at revision" |
| **E5 PV** | 15–18 | M1 + M2 on Kaggle solar, integrated into the same job queue | M1/M2 PV metrics + PV jobs in the calendar |
| **E6 M3 (optional, in parallel)** | 6–18 | NBM on CARE farm A, compared with EnergyFaultDetector | `results/m3_care.json` with the CARE score for model and baseline |
| **E7 LLM agent** | 18–21 | `tools.py`, `llm_agent.py`, `api.py` | Scenarios from 10.4 work end to end; no-LLM (CLI) mode works |
| **E8 Demo + README** | 21–24 | `dashboard.py`, README, single evaluation script | `make eval` reproduces every figure in the presentation; demo rehearsal |

---

## 9. Module specifications

### 9.1 M1: expected production (`src/producers/expected_power.py`)

**What it represents:** how much the equipment would produce right now if it had no problems.

**Training data filtering (critical):**

- `status_normal == True`, no active events;
- remove curtailment: power capped at a fixed value below rated in high wind;
- remove stuck sensors (same value > N intervals) and impossible values (power < −5% of rated, wind < 0 or > 40 m/s);
- PV: only hours with irradiance above a threshold (e.g. 50 W/m²), no days marked with faults.

**Features:**

- **Wind:**
  - `wind_ms`, `wind_ms**3`;
  - air density `rho = p / (287.05 * T_K)`, with pressure from Open-Meteo if not in SCADA;
  - turbulence `wind_std_ms / wind_ms`;
  - `sin/cos` of wind direction, hour and month.
- **PV:** `irradiance_wm2`, `module_temp_c`, solar elevation and azimuth (`pvlib.solarposition`), `sin/cos` of the hour, `irradiance * (1 + γ·(T_mod − 25))` as a correction feature.

**Model:**

- LightGBM, `objective="quantile"`, one model for each α ∈ {0.1, 0.5, 0.9};
- one model per turbine type (or per turbine, if there is enough data);
- starting hyperparameters: `num_leaves=31, learning_rate=0.05, n_estimators=800`, early stopping on validation.

**Split:** e.g. year 1 → train (first 10 months) + validation (last 2 months); year 2 → test. Document the choice.

**Baseline:**

- **wind:** power curve on 0.5 m/s bins, i.e. the mean power per bin, computed on train (the method of bins, as in IEC 61400-12-1);
- **PV:** `rated_kw * irradiance/1000 * (1 + γ·(T_mod − 25))`, with γ ≈ −0.004/°C as an assumption in `params.yaml`.

**Metrics:** MAE and RMSE (kW and % of rated), plus 10–90% interval coverage (ideally approx. 80%), separately per wind or irradiance bin.

**API:**

```python
def predict_expected(df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """Returns df with columns exp_p10_kw, exp_p50_kw, exp_p90_kw."""
```

### 9.2 Synthetic faults (`src/simulate/inject_faults.py`)

All injections are **deterministic with a seed** and save labels (`asset_id, start, end, cause, severity`) to `data/processed/fault_labels.parquet`.

| Cause | Technology | Injection | Parameters (from `params.yaml`) |
|---|---|---|---|
| `yaw_misalignment` | wind | `P *= cos(θ)**2` below rated power (documented approximation) | θ ∈ [4°, 15°] |
| `pitch_fault` | wind | cap power at `k * rated` when wind > a threshold | k ∈ [0.8, 0.95] |
| `anemometer_bias` | wind | `wind_ms *= (1 + b)`, power unchanged | b ∈ [−0.10, +0.10] |
| `icing` | wind | `P *= (1 − s)` only when T ∈ [−3, +1] °C and humidity is high | s ∈ [0.2, 0.6] |
| `soiling` | PV | loss growing linearly by `r` per day, reset by rain | r ∈ [0.1%, 0.5%]/day |
| `string_outage` | PV | `P *= (1 − 1/N)` abruptly | N strings per inverter (assumption) |
| `shading` | PV | `P *= (1 − s)` only in a fixed hourly interval, growing weekly | |
| `inverter_derate` | PV | cap at `k * rated` | |

Inject **only into the test or backtest period** for evaluation. To train the cause classifier, inject separately into train. Each fault lasts between 3 and 30 days, at random.

### 9.3 M2: underperformance (`src/producers/underperformance.py`)

1. **Performance index** `PI = Σ P_real / Σ exp_p50` over a sliding 72-hour window, only on intervals where `exp_p50 > 10%` of rated.
2. **Change detection:** CUSUM on `(P_real − exp_p50)/rated`, with the threshold chosen on validation to limit false alarms (e.g. ≤ 1 per turbine per month). Alternative: alarm if `P_real < exp_p10` in > X% of the intervals of the last 24 h.
3. **Cause classifier** (multiclass LightGBM) on deviation features:
   - where it appears: relative loss per wind or irradiance bin; by hour; by temperature;
   - how it appeared: the slope (abrupt or gradual);
   - how it looks against neighbouring turbines: same period, same wind;
   - whether it resets after rain (PV).
4. **Outputs per asset:** `pi_72h`, `loss_mwh_per_day` (= the 72 h mean of `exp_p50 − P_real`, scaled to a day), `cause`, `cause_proba`, `confidence`.

**Baseline:** fixed threshold `PI < 0.95` with no CUSUM and no classifier.

**Metrics:** detection (%), mean detection delay (days), false alarms per asset-month, macro F1 by cause, error of the `loss_mwh_per_day` estimate against the injected loss.

**Real-data validation:** at Hill of Towie, the AeroUp/TuneUp upgrades are real performance changes with known dates. Check whether M2 sees the change (in the positive direction) at those dates.

### 9.4 M3: failure risk, optional (`src/producers/failure_risk.py`)

**Option A, CARE to Compare (recommended):**

- Start with **farm A** only.
- For each dataset: on rows with `train_test == "train"` and normal status (0 or 2), train a normal-behaviour model (LightGBM) that predicts a few temperatures (bearings, oil, generator) from power, wind, rotor speed and outside temperature.
- Anomaly score = standardised, smoothed (moving average) deviation.
- The threshold is chosen on the validation part of the `train` data.
- Compute the **CARE score**, exactly as in the paper:
  - **Coverage:** Fβ with β = 0.5 on points in the prediction window with normal status;
  - **Accuracy:** tn/(fp+tn) on normal datasets;
  - **Reliability:** Fβ on events, with a criticality counter that rises by 1 on each anomaly with normal status and falls by 1 (minimum 0) otherwise; alarm if the maximum reaches **72** (12 hours);
  - **Earliness:** weight 1 in the first half of the event, decreasing linearly to 0 towards the fault;
  - **Combination:** WA = (Cov + Earl + Rel + 2·Acc) / 5; CARE = 0 if no anomaly is detected; CARE = Acc if Acc < 0.5; otherwise CARE = WA.
- **Baseline:** fixed threshold on raw temperature and `energy-fault-detector` (Fraunhofer IEE autoencoder), run on the same datasets.

**Option B, Hill of Towie:** classifier for "serious alarm or downtime > X hours in the next 7 days" on daily aggregated features (number of minor alarms, temperature deviations, `pi_72h` from M2). Temporal split. Metrics: precision, recall, PR-AUC.

**Output for the optimizer:** `p_fail(d)` for d = 1..14 days, **calibrated** (isotonic on validation).

### 9.5 M4: downtime cost (`src/producers/downtime_cost.py`)

1. Take the hourly Open-Meteo forecast for the park's coordinates (in the backtest: historical weather + noise, see 9.7).
2. Extrapolate wind to hub height: `v_hub = v_ref * (h_hub/h_ref)**α`, with α = 0.143 as an assumption.
3. Build features as for M1 (without SCADA columns unavailable in the future) and apply M1 → `exp_p50_kw` for each hour.
4. `cost_mwh[asset, hour] = exp_p50_kw / 1000` (one stopped hour = that hour's production). Optional: × hourly price → RON.
5. `workable[asset, hour] = v_hub < V_SAFE and gust < G_SAFE and precip < P_MAX`. For PV, each job type has its own allowed hours. **`V_SAFE`, `G_SAFE` are assumptions in `params.yaml`; real values come from the manufacturer's manual.**

**Output:** a DataFrame `asset_id, ts_utc, cost_mwh, cost_mwh_p90, workable`.

### 9.6 Optimizer (`src/decision/optimizer.py`)

**Jobs:**

```python
@dataclass
class Task:
    task_id: str
    asset_id: str
    site_id: str
    kind: Literal["revision", "repair", "preventive", "inspection"]
    duration_h: float
    due_start: date | None          # tolerance window (revisions)
    due_end: date | None
    loss_mwh_per_day: float = 0.0   # from M2 (what you lose per day of waiting)
    p_fail_by_day: list[float] | None = None   # from M3
    fail_cost_mwh_eq: float = 0.0   # assumption (cost of a failure, in MWh equivalent)
    confidence: float = 1.0
    explanation_inputs: dict = field(default_factory=dict)   # for traceability
```

**Periodic revisions:** generated from `params.yaml`: every 6 months, the annual one longer, tolerance window ± N days. As an order of magnitude, a public O&M plan describes revisions every 6 months, with the annual one lasting 2–4 days.

**Cost of a job on day d** (for the hour interval h chosen on day d):

```
C(task, d, h) = Σ_{hours worked} cost_mwh[asset, hour]        # downtime (M4)
              + loss_mwh_per_day * (d − today)                # loss while waiting (M2)
              + p_fail(d) * fail_cost_mwh_eq                  # risk (M3)
```

**Constraints:**

- ≤ `CREWS_PER_DAY` crews, each with ≤ `HOURS_PER_CREW` hours;
- only `workable` hours;
- revisions within their window;
- one crew per asset at a time;
- jobs in the same park on the same day get a bonus (shared travel).

**Grouping with revisions:** if a repair and a revision are on the same asset and the revision is within the horizon, also evaluate "repair at revision" (the downtime is already paid for by the revision) and pick the cheaper option.

**MVP algorithm (greedy):**

1. Sort jobs by urgency: revisions close to `due_end`, high `p_fail`, high `loss_mwh_per_day`.
2. For each job, evaluate all feasible slots in the 7–10 day horizon and pick the minimum cost; reserve capacity.
3. Jobs with `confidence < CONF_MIN` or an uncertain forecast (large `cost_mwh_p90 − cost_mwh`) become `inspection` (drone or remote check) instead of `repair`.

**Optional:** CP-SAT model (OR-Tools) with the same costs; report the gain over greedy.

**Output (`schedule.json`)**, for each job:

```json
{
  "task_id": "T17-pitch-001",
  "asset_id": "FAN-T17",
  "date": "2026-11-12",
  "start_hour": 8,
  "crew": "crew-2",
  "cost_breakdown_mwh": {"downtime": 2.0, "waiting": 5.6, "risk": 0.0, "total": 7.6},
  "alternatives": [{"date": "2026-11-09", "total": 19.4, "why_not": "strong wind: 18 MWh downtime"}],
  "trace": {"m2_cause": "pitch_fault", "m2_proba": 0.81, "m2_loss_mwh_day": 1.4, "m4_forecast_run": "2026-11-08T06:00Z"}
}
```

### 9.7 Backtest (`src/decision/backtest.py`)

**Goal:** prove that the agent saves energy compared with simple policies.

1. Period: one test year (e.g. Kelmarsh 2020) that M1 has not seen.
2. Inject 20–40 faults (9.2) at random dates and generate revisions every 6 months.
3. Day-by-day simulation: each day, the agent sees only data up to that day. It runs M2 (detection), then M4 on the "forecast", then the optimizer, and executes the jobs scheduled for that day. A fault disappears when repaired.
4. **The "forecast":**
   - **(a)** perfect: the real weather of the following days;
   - **(b)** noisy, growing with horizon, e.g. `v_forecast = v_real * (1 + ε)`, with ε ~ N(0, σ_d), σ_d growing with the day (documented assumption). Report both.
5. **Policies compared:**
   - **(i)** "repair as soon as detected", ignoring weather;
   - **(ii)** "repair at the next revision";
   - **(iii)** the agent.
6. **Metrics:** total MWh lost (from downtime + underperformance + failures), number of trips, mean days until repair. Stacked bar chart across the three policies.

### 9.8 PV branch

- M1 and M2 identical in structure, on `pv_inverter.parquet` (Kaggle). With only 34 days: train the first 22 days, validate the next 5, test the last 7. **State explicitly in the README that the set is short.**
- PVDAQ (optional) for an M1 model on a long series.
- **PV jobs in the optimizer:** cleaning (when the daily soiling loss × days until forecast rain exceeds the cost of cleaning), string replacement, inverter check. Downtime cost = forecast PV production in the working hours; night is nearly zero.
- **Optional, images (PVF-10):** pre-trained ResNet18, fine-tuned (torchvision), on the fixed split in the archive. Metric: macro F1 (the set is imbalanced). Output: class + repairable/non-repairable, per the mapping in the dataset's README. Baseline: Random Forest on resized pixels.

---

## 10. LLM agent layer (`src/agent/`)

### 10.1 Principle

The LLM **orchestrates and explains**. All numbers come from tools. If there is no LLM API key, the application runs in **deterministic CLI mode**, so the demo does not depend on the network.

### 10.2 Exposed tools (`tools.py`)

Producers and shared tools. The consumers tools are listed in `README.md` section 3.3.

| Tool | Input | Output |
|---|---|---|
| `get_weather_forecast(site_id, days)` | site, days | hourly DataFrame (cached) |
| `predict_expected_production(asset_ids, start, end)` | | kW quantiles |
| `detect_underperformance(asset_ids, as_of)` | | `pi_72h, loss_mwh_per_day, cause, confidence` |
| `compute_cost_map(site_id, horizon_days)` | | hourly `cost_mwh, workable` |
| `list_tasks(as_of)` | | open jobs (revisions + detections) |
| `optimize_maintenance(constraints: dict)` | e.g. `{"crews_per_day": 2, "unavailable": [{"crew": "crew-2", "date": "2026-11-12"}]}` | `schedule.json` |
| `loss_overview(period)` | | losses by source, in MWh and RON, across both modules |
| `explain(item_id)` | | `cost_breakdown`, alternatives, `trace` |
| `update_from_field_report(text)` | free text from a technician | JSON validated against the schema (see 10.3), applied after confirmation |
| `create_work_orders(schedule_id)` | | ICS/JSON files in `results/work_orders/` |
| `send_alert(level, message, asset_id)` | | log in `results/alerts.log` (simulated) |

### 10.3 Field report schema

```json
{
  "task_id": "string",
  "status": "done | deferred | needs_part | not_found",
  "defer_days": "int | null",
  "new_finding": "string | null",
  "confidence": "float"
}
```

The LLM fills in the schema and the code **validates** it (pydantic) before applying it. If `confidence < 0.7` or fields are missing, the agent asks for human confirmation.

### 10.4 Demo scenarios that must work

1. "What do we do this week at Fântânele?" → `list_tasks` + `compute_cost_map` + `optimize_maintenance` → calendar + explanation.
2. "Why T17 on Friday and not tomorrow?" → `explain` → answer with the figures from `cost_breakdown`.
3. "Crew 2 is missing on Thursday." → `optimize_maintenance` with the new constraint → new plan + cost difference.
4. "I checked T05, the bearing is worn, the part arrives in 5 days." → `update_from_field_report` → confirmation → re-planning.
5. A low-confidence case → the agent proposes a drone check rather than scheduling a crew.

### 10.5 System prompt (skeleton)

```
You are the maintenance planning assistant for the PPC Renewables parks.
Rules:
- Do not compute costs or pick days yourself. Always use the tools.
- Every number in your answer must come from a tool result; name the tool.
- For any change to the plan, run optimize_maintenance and report the cost difference.
- If confidence is below the threshold or data is missing, say "I'm not sure" and propose a check.
- Answer in Romanian, concisely, with the table of options when explaining a decision.
```

---

## 11. Explainability

- **M1/M2:** SHAP values for individual predictions (`shap.TreeExplainer`); top 3 factors per alarm.
- **Decision:** `cost_breakdown` + rejected alternatives and the reason (see 9.6), shown in the interface.
- **Similar cases:** for M3 on CARE, the nearest historical events (k-NN on the deviation vector), with the fault description from `event_info`.

## 12. Robustness (checklist)

- [ ] Missing intervals: interpolate only short gaps (≤ 30 min); long gaps are marked, not filled.
- [ ] Stuck sensors and impossible values: filtered and reported.
- [ ] Turbine in service or derated: excluded from M2, not treated as a fault.
- [ ] Forecast missing or API unavailable: use the last cache and mark the plan as "stale forecast".
- [ ] Low confidence (M2) or wide interval (M1/M4): route to human verification.
- [ ] Bad API inputs (non-existent asset, past date): clear errors, no crash.
- [ ] Tests: `tests/test_schema.py` (columns and types), `tests/test_optimizer.py` (respects capacity and windows on a small hand-built case).

## 13. Reproducibility

`Makefile` (what each target does for users is described in `README.md` section 6). Every module runs as `python -m src.<package>.<module>` from the repository root:

```make
setup:      pip install -r requirements.txt
data:       python -m src.data.prepare --year 2019 && python -m src.data.prepare --pv
data-test:  python -m src.data.prepare --year 2020
train:      --train on expected_power, underperformance, downtime_cost, solar
eval:       --eval on the same four (validation) + python -m src.decision.backtest --dev
eval-final: --eval --final on the same four + python -m src.decision.backtest --final
plan:       python -m src.decision.optimizer --site fantanele calugareni
test:       python -m pytest
agent:      uvicorn src.agent.api:app        # when the agent exists
demo:       streamlit run src/app/dashboard.py  # when the dashboard exists
```

The README must also contain the error analysis (where M1/M2 go wrong, on which causes, what was tried) and what production would need from PPC.

## 14. 3-minute demo

1. Map: Fântânele turbines coloured by `pi_72h`.
2. Click a turbine: real production below the expected band, probable cause, MWh/day lost, SHAP.
3. Cost map over 7 days (real Open-Meteo weather), with calm days highlighted.
4. The generated calendar, with the table of options for a job.
5. LLM scenario: "Crew 2 is missing on Thursday" → new plan + cost difference.
6. Backtest chart: agent vs. "repair immediately" vs. "repair at revision".

## 15. Questions for PPC (at the opening presentation)

1. What SCADA signals do you have for turbines (10 minutes? component temperatures?) and how much history?
2. What service contracts do you have with manufacturers and what monitoring do they already include?
3. At what intervals do you do revisions and how long do they take?
4. What inverters and what monitoring system do you have in the PV parks?
5. How many field crews do you have per region and what are the wind limits for working at height?
6. Do you use the hourly energy price when planning outages?
