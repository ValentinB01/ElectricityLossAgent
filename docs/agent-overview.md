# Energy Loss Agent

**An agent that predicts and reduces energy losses at both ends of the chain: at producers and at consumers.**

Project for **Smarthack XI 2026**, PPC theme *"24h to Power PPC: AI in energy, from data to action"*.

> **Status:** planning stage, no code yet. This README describes the target system: the commands below and the files they use are created as the plan in [`docs/producers-plan.md`](producers-plan.md) progresses.

- **At producers** (wind turbines and PV parks): we estimate how much each piece of equipment should produce. We detect when it produces below normal and schedule revisions and checks in the windows with the lowest loss: low wind for turbines, little sun for panels.
- **At consumers** (households): we estimate how much each household should consume. We detect anomalies in the monthly consumption series (possible theft or a faulty meter) and prioritise inspections.
- **An LLM agent** puts all losses into a single view (MWh and RON), explains each decision and runs what-if scenarios. Decisions come from our models and optimizer, not from the LLM.

---

## Contents

1. [Problem](#1-problem)
2. [Architecture](#2-architecture)
3. [Modules](#3-modules)
4. [Results](#4-results)
5. [Data](#5-data)
6. [Install and run](#6-install-and-run)
7. [Repository structure](#7-repository-structure)
8. [Evaluation](#8-evaluation)
9. [Robustness and explainability](#9-robustness-and-explainability)
10. [Limitations](#10-limitations)
11. [Team](#11-team)
12. [Licences and citations](#12-licences-and-citations)

**Other documents:** [`docs/design.md`](design.md) explains why the system is designed this way. [`CLAUDE.md`](../CLAUDE.md) is the build specification.

---

## 1. Problem

Energy losses occur in two places that PPC covers with different divisions:

| Where | What is lost | Why it matters to PPC |
|---|---|---|
| **Production** (PPC Renewables) | Energy not produced when turbines or panels run below potential or are stopped for maintenance at times of high wind or sun | A portfolio of over 1.5 GW in Romania, including the Fântânele-Cogealac wind farm (600 MW, 240 GE 2.5xl turbines) and the Lumina PV park in Călugăreni (63 MW) |
| **Distribution** (Rețele Electrice) | Energy consumed but not billed: theft, faulty meters | A [2018 study](https://www.economica.net/furturile-de-energie-electrica-se-ridica-la-340-de-milioane-de-lei-in-fiecare-an-ce-e-de-facut_158325.html) estimated theft at approx. 3% of the energy entering distribution networks; Rețele Electrice has 2 million smart meters |

The PPC group's [AI programme](https://economedia.ro/georgios-stassis-ppc-vrem-o-cerere-noua-din-partea-centrelor-de-date-dar-trebuie-sa-evitam-ca-ai-sa-mareasca-factura-de-energie-a-clientului-final-corespondenta-eurelectric-power-summit-2.html) explicitly includes predictive maintenance and network loss reduction. Our project puts both problems under one agent, with one unit of measure: energy lost.

---

## 2. Architecture

![Two loss modules, one agent](img/agent-architecture.svg)

Three layers, each with its own job:

- **ML models** output physical units with uncertainty (kW, MWh/day, kWh/month, probabilities).
- **The decision layer** (maintenance optimizer, inspection queue) is deterministic and reproducible.
- **The LLM agent** coordinates and explains.

**Principle:** the LLM does not calculate or decide. Every displayed number comes from a tool, and every decision can be traced to the model prediction that produced it.

---

## 3. Modules

### 3.1 Producers: expected production and maintenance planning

| Component | What it does | Algorithm | Baseline |
|---|---|---|---|
| **M1** Expected production | How much the turbine or inverter would produce now, with no problems | Quantile LightGBM (10/50/90%), trained on normal periods only | Power curve on 0.5 m/s bins (wind); rated power × irradiance with temperature correction (PV) |
| **M2** Underperformance | Detects persistent loss, the probable cause and MWh lost per day | 72 h performance index + CUSUM + cause classifier (LightGBM) trained on injected faults | Fixed threshold on the performance index |
| **M3** Failure risk *(optional)* | Probability of failure in the coming days, from abnormal temperatures | Normal-behaviour model on temperatures (CARE to Compare) | Fixed temperature threshold; EnergyFaultDetector (Fraunhofer IEE) |
| **M4** Downtime cost | MWh (or RON) lost if the equipment is stopped in each hour of the next 7–10 days | M1 applied to the weather forecast | – |
| **Optimizer** | Picks the day and crew for each revision and repair | Greedy (optionally CP-SAT) on the cost: downtime + loss while waiting + risk | Policies: "repair immediately", "repair at the next revision" |

Causes detected by M2: yaw misalignment, pitch fault, faulty anemometer, icing (wind); soiling, string outage, shading, derated inverter (PV).

### 3.2 Consumers: anomalies in monthly consumption

| Component | What it does | Algorithm | Baseline |
|---|---|---|---|
| **C1** Expected consumption | How much the household should consume in a month | Quantile LightGBM, from its own history for the same month, trend, similar households, temperature | Median of the same month in previous years × trend factor |
| **C2** Consumption anomaly | Probability of theft, kWh lost per month, signal type (theft / faulty meter / uncertain) | Isolation Forest (unsupervised) + LightGBM (supervised, on SGCC labels and simulated attacks), calibrated | Rule "consumption more than X% below the same month last year" |
| **Inspection queue** | Which households get checked first | Value = P(theft) × kWh lost × recoverable months − travel cost; top K per crew, grouped geographically | Ordering by P(theft) alone |

The agent accuses no one: it schedules checks. Probably faulty meters go to the metering team, uncertain cases to a remote check.

### 3.3 LLM agent

| Module | Exposed tools |
|---|---|
| Producers | `get_weather_forecast`, `predict_expected_production`, `detect_underperformance`, `compute_cost_map`, `list_tasks`, `optimize_maintenance` |
| Consumers | `predict_expected_consumption`, `score_households`, `build_inspection_list` |
| Shared | `loss_overview`, `explain(item_id)`, `update_from_field_report(text)`, `create_work_orders`, `send_alert` |

Example questions:

- "What losses do we have this week?"
- "Why turbine T17 on Friday and not tomorrow?"
- "Crew 2 is missing on Thursday." (the agent re-plans and shows the cost difference)
- "Why is household X on today's list?"
- "I checked household X: faulty meter, not theft." (the agent asks for confirmation, closes the case and saves the label)

Without an LLM API key, the application runs in **deterministic CLI mode**, with the same tools.

---

## 4. Results

> All figures below are generated with `make eval` and saved in `results/`. Fill in the table after running.

| Component | Metric | Model | Baseline |
|---|---|---|---|
| M1 wind (Kelmarsh, test year) | MAE (% of rated) / 10–90% interval coverage | _TBD_ | _TBD_ |
| M1 PV (Kaggle, last 7 days) | MAE (% of rated) | _TBD_ | _TBD_ |
| M2 wind (injected faults) | detection % / delay (days) / false alarms per turbine-month / cause F1 | _TBD_ | _TBD_ |
| M3 (CARE to Compare, farm A) | CARE score | _TBD_ | _TBD_ |
| C2 consumers (monthly SGCC, test customers) | PR-AUC / precision in top K / kWh recovered in top K | _TBD_ | _TBD_ |
| **Maintenance backtest (one year)** | total MWh lost | **agent:** _TBD_ | **repair immediately:** _TBD_ · **at revision:** _TBD_ |

Charts (power curves, confusion matrix, backtest) are in `results/figures/`.

---

## 5. Data

We use only public and synthetic data. Data is **not** in the repository: `make data` downloads it.

### 5.1 Producers

| Dataset | What we use | Licence | Source |
|---|---|---|---|
| Kelmarsh wind farm data (Cubico) | 10-minute SCADA, 6 Senvion MM92 turbines, 1–2 years | CC-BY-4.0 | https://zenodo.org/records/16807551 |
| CARE to Compare (Fraunhofer IEE) | 95 SCADA datasets with labelled faults (for M3) | CC BY-SA 4.0 | https://zenodo.org/records/15846963 |
| Hill of Towie (RES / TRIG) *(optional)* | SCADA + alarm log; upgrades with known dates | CC-BY-4.0 | https://zenodo.org/records/14870023 |
| Penmanshiel (Cubico) *(reserve)* | 10-minute SCADA | CC-BY-4.0 | https://zenodo.org/records/16807304 |
| Solar Power Generation Data | 2 PV plants, 34 days, inverter data at 15 minutes | see the Kaggle page | https://www.kaggle.com/datasets/anikannal/solar-power-generation-data |
| PVDAQ (NREL) *(optional)* | Long PV production series | CC-BY | https://data.openei.org/submissions/4568 |
| PVF-10 *(optional)* | Thermal images of PV modules | see the repo | https://github.com/wangbobby1026/PVF-Dataset |
| InfraredSolarModules (Raptor Maps) *(optional)* | Thermal images for pre-training | MIT | https://github.com/RaptorMaps/InfraredSolarModules |
| Open-Meteo | Historical weather and forecast (wind, irradiance, temperature, pressure) | CC BY 4.0, non-commercial use | https://open-meteo.com |

### 5.2 Consumers

| Dataset | What we use | Licence | Source |
|---|---|---|---|
| SGCC electricity theft dataset | 42,372 customers, daily consumption 2014–2016, theft label; aggregated monthly | see the dataset page | Kaggle, "SGCC Electricity Theft Detection" (bensalem14); description: https://baselight.app/u/kaggle/dataset/bensalem14_sgcc_dataset |
| Smart Meters in London | 30-minute consumption, aggregated monthly, with simulated attacks | see the dataset page | https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london |
| Synthetic Data for Smart Meter Attack Detection *(verification)* | 159 users, 7 attack strategies | see the dataset page | https://ieee-dataport.org/documents/synthetic-data-smart-meter-attack-detection |

### 5.3 Synthetic faults and attacks

Public data has no labels for every case, so we generate them in a controlled, documented way (`src/simulate/`):

- **producers:** yaw misalignment, pitch, anemometer, icing, soiling, string outage, shading, derated inverter;
- **consumers:** the classic attack functions from the literature (multiplication by a constant, random constant, interval shutdown, etc.), applied to "honest" consumption.

All injections use a fixed seed, and labels are saved separately.

> Weather data by [Open-Meteo.com](https://open-meteo.com/).

---

## 6. Install and run

**Requirements:** Python 3.11, approx. 5 GB free space for the minimal set (over 10 GB if you also download CARE to Compare).

```bash
git clone <repo-url>
cd <repo>
python3.11 -m venv .venv && source .venv/bin/activate
make setup
```

**Kaggle token** (for the PV and consumption datasets): create `~/.kaggle/kaggle.json` from your Kaggle account → *Settings* → *API*.

**LLM** *(optional)*: set the environment variable for the chosen provider (see `config/agent.yaml`). Without it, the agent runs in CLI mode.

| Command | What it does |
|---|---|
| `make data` | Downloads the minimal set (Kelmarsh, Kaggle solar, SGCC, weather) and brings it to the common schema in `data/processed/` |
| `make data-full` | Also downloads the optional sets (CARE to Compare, Hill of Towie, PVDAQ) |
| `make train` | Trains M1, M2, C1, C2 (and M3 if the data exists) |
| `make eval` | Evaluates all models against baselines and runs the backtest; writes `results/*.json` and the charts |
| `make agent` | Starts the agent API (FastAPI) |
| `make demo` | Starts the interface (Streamlit) |
| `make test` | Runs the schema and optimizer tests |

All numeric assumptions (wind limit for work, failure cost, job durations, crew capacity, energy price) are in `config/params.yaml`.

---

## 7. Repository structure

| Path | Contents |
|---|---|
| `config/` | Numeric assumptions, site coordinates, LLM settings |
| `src/data/`, `src/features/` | Downloads, ingestion, weather, feature building |
| `src/producers/`, `src/consumers/` | The models of each module |
| `src/simulate/` | Synthetic faults and attacks |
| `src/decision/` | Optimizer and backtest |
| `src/agent/`, `src/app/` | LLM tools, API and dashboard |
| `results/` | JSON metrics, charts, generated work orders |
| `docs/` | Design, plan, data notes, decisions and error analysis |
| `data/`, `models/` | Git-ignored; recreated by `make data` and `make train` |

File-level layout: [`CLAUDE.md`](../CLAUDE.md) section 5.

---

## 8. Evaluation

- **Splits without information leakage:**
  - production series: split by time;
  - consumers: split by customer (no customer in both train and test) and by time;
  - the test set is fixed at the start.
- **Every model is compared with a simple baseline** (the tables in section 3).
- **Producers:** MAE and interval coverage (M1); detection, delay, false alarms and F1 by cause (M2); CARE score (M3).
- **Consumers:** PR-AUC, precision in top K and kWh recovered in top K, where K = daily inspection capacity.
- **The decision:** one-year backtest. The agent runs day by day, seeing only data up to that day, with a perfect forecast and with a noisy forecast. We compare it with "repair immediately" and "repair at the next revision".

Error analysis (where each model goes wrong, on which kinds of cases and what we tried) is in `docs/decisions.md` and in the presentation.

---

## 9. Robustness and explainability

**Robustness:** data gaps, stuck sensors, impossible values, assets in service, a stale forecast and invalid API input are each handled explicitly rather than treated as faults or crashes. With low confidence, the agent says "I'm not sure" and sends the case to a person (drone inspection, remote meter check).

**Explainability:**

- SHAP values for each alarm (top 3 reasons);
- for each scheduled job: the cost breakdown (downtime / waiting / risk) and the rejected options, with the reason;
- for each suspect household: the monthly curve against expectation and against similar households.

---

## 10. Limitations

- **The data is not from PPC.** Turbines (Senvion, Siemens) and PV plants (India, USA) differ from those in the PPC portfolio. At Fântânele-Cogealac and Călugăreni we use only real weather, for demonstration.
- **The Kaggle PV set has only 34 days.** PV results are indicative.
- **Faults and attacks are partly synthetic.** Performance on real cases may be lower; for theft, we test separately on the real SGCC labels.
- **Monthly granularity hides some kinds of theft** (for example shifting consumption between hours). Daily or hourly data from smart meters would improve detection.
- **New prosumers** (net consumption falling after installing panels) can look like theft if the "prosumer" attribute is missing.
- **Backtest forecasts are simulated** (real weather, optionally with noise), not archived forecasts.
- **Operational parameters are assumptions** (wind limits, costs, durations): in production they would come from manufacturers' manuals and PPC data.

For production, we would need from PPC: real SCADA, the service log, service contracts, inverter types, inspection history and outcomes.

---

## 11. Team

| Member | Role |
|---|---|
| Valentin | Producers module: expected production (wind + PV), downtime cost, maintenance optimizer, backtest |
| _Colleague_ | Consumers module: expected consumption, anomaly detection, inspection queue |
| _Whole team_ | LLM agent, interface, evaluation, presentation |

---

## 12. Licences and citations

**Code:** MIT (see `LICENSE`).

**Data:** each dataset keeps its own licence (section 5). Formal citations beyond the links in section 5:

- Gück, C., Roelofs, C. M. A., Faulstich, S. *CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine Data*. Data 2024, 9(12), 138. https://doi.org/10.3390/data9120138
- Cubico Sustainable Investments, *Kelmarsh wind farm data* and *Penmanshiel wind farm data*, Zenodo.
- RES on behalf of TRIG, *Hill of Towie wind farm open dataset*, Zenodo.
- NREL, *Photovoltaic Data Acquisition (PVDAQ) Public Datasets*. https://doi.org/10.25984/1846021
- Wang et al., *PVF-10*, Applied Energy 376 (2024). https://doi.org/10.1016/j.apenergy.2024.124187
- State Grid Corporation of China, *SGCC electricity theft dataset*, via Kaggle.
- *Smart Meters in London*, Kaggle (jeanmidev), data from London Datastore / UK Power Networks.

**Tools:**

- Open-Meteo.com Weather API, CC BY 4.0: [terms](https://open-meteo.com/en/terms)
- Fraunhofer IEE, *EnergyFaultDetector*: https://pypi.org/project/energy-fault-detector/
- NREL, *OpenOA* (BSD-3): https://github.com/NREL/OpenOA
- pvlib-python: https://pvlib-python.readthedocs.io
