# Producers module: implementation plan

Status: **not started.** Work through one step at a time; each step ends with a check and a commit. Specs are in [`CLAUDE.md`](../CLAUDE.md) (§ references below); this plan covers order, acceptance and the traps the spec does not mention.

## Scope

The producers half of the [architecture diagram](img/agent-architecture.svg), from data to a calendar.

| Diagram box | Step | Code | Output |
|---|---|---|---|
| SCADA, inverter | 1 (wind), 7 (PV) | `src/data/wind_ingest.py`, `pv_ingest.py` | `wind_scada.parquet`, `pv_inverter.parquet` |
| Weather forecast | 1, 4 | `src/data/weather.py` | `weather_hourly.parquet` |
| Expected production (M1) | 2, 7 | `src/producers/m1_expected.py` | `exp_p10/p50/p90_kw` |
| Underperformance (M2) | 3, 7 | `src/simulate/inject_faults.py`, `src/producers/m2_underperf.py` | `pi_72h`, `loss_mwh_per_day`, `cause`, `confidence` |
| Downtime cost per hour (M4) | 4, 7 | `src/producers/m4_cost.py` | `cost_mwh`, `cost_mwh_p90`, `workable` |
| Revisions, crews | 5 | `src/decision/tasks.py`, `config/params.yaml` | `tasks.json` |
| Optimizer | 5, 6 | `src/decision/optimizer.py`, `backtest.py` | `schedule.json`, `backtest.json` |

**Out of scope here:** the LLM agent, the dashboard, the consumers module and M3. M3 has no box in this diagram, so `p_fail_by_day` stays `None` and the risk term is 0: the optimizer's "risk × cost" is cost only until M3 exists. The interface stays in place so M3 can plug in later.

## Starting point (checked 2026-11-07)

| Item | Finding | Consequence |
|---|---|---|
| Kelmarsh, CARE links | Files, sizes and the CARE MD5 match the guide on Zenodo | Safe to script |
| Python | Machine default is 3.14; the guide requires 3.11 | Use a conda env (3.11 is available on conda-forge) |
| Kaggle token | None on this machine | **Blocks step 7 only.** Create it at Kaggle → Settings → API |
| Git remote | Local repo only; the brief requires a public repo with commits throughout the 24 h | Create the GitHub repo before step 1 |

## Decision to confirm before step 4

**M4 should use a weather-driven sibling of M1, not M1 itself.** §9.5 says to apply M1 to the forecast with the SCADA-only columns left out. But M1 learns from the nacelle anemometer, which sits behind the rotor and reads differently from a forecast's free-stream wind, and its turbulence feature has no forecast equivalent. Feeding it forecast wind would bias the cost map, and that cost map drives every scheduling decision.

Recommended: train **M1-W**, with the same algorithm and normal-period filter, mapping Open-Meteo archive weather to hourly mean power, as a fraction of rated. It costs about an hour and gets its own test-set score against a binned baseline. M1 stays as specified for M2. If you agree, §9.5 gets updated at step 4.

## Steps

Hours are the guide's budget (§8). Wind goes end to end first, so a working agent exists before PV starts.

### Step 0: Foundation (1 h, stage E0)

- **Build:** conda env, `requirements.txt`, `Makefile`, `.gitignore`, `LICENSE`, `config/params.yaml`, `config/sites.yaml`, package skeleton. Start the Kelmarsh downloads in the background.
- **Done when:** `make setup` works in a clean env, all key libraries import, downloads are running.

### Step 1: Wind data (3 h, E1)

- **Build:** `download.py` (with checksum verification), `wind_ingest.py`, `weather.py` with cache, `docs/data_notes.md`, `tests/test_schema.py`.
- **Done when:** `wind_scada.parquet` holds 6 turbines × 2 years; missing-value percentages are reported; Kelmarsh weather is cached.
- **Watch out:**
  - Derive `status_normal` from the real event files and record the logic. Everything downstream trusts this flag.
  - Check whether Kelmarsh timestamps mark the start or end of the interval before converting; the schema requires the end.

### Step 2: M1 wind (3 h, E2)

- **Build:** training filter, features, three quantile models, binned power-curve baseline, evaluation. Split: Jan–Oct 2019 train, Nov–Dec 2019 validation, 2020 test.
- **Done when:** `results/m1_wind.json` has test MAE/RMSE for model and baseline, and 10–90% coverage by wind bin; the real vs. expected chart exists.
- **Watch out:**
  - Predict **power as a fraction of rated**, then multiply by `rated_kw`. This is what lets a 2.05 MW Kelmarsh model say anything about a 2.5 MW Fântânele turbine. Specific power is close (about 308 vs. 318 W/m², assuming 92 m and 100 m rotors; confirm the GE figure), which is a defensible argument, not proof.
  - Quantile models can cross. Sort the three outputs per row.

### Step 3: Fault injection and M2 wind (4 h, E3)

- **Build:** `inject_faults.py`, performance index, CUSUM, cause classifier, fixed-threshold baseline.
- **Done when:** `results/m2_wind.json` has detection rate, delay, false alarms per turbine-month and macro F1, against the baseline; the confusion matrix exists.
- **Watch out:**
  - **`anemometer_bias` loses no energy.** Power is unchanged, so `exp_p50 − P_real` reports a phantom loss when `b > 0`, and the optimizer would send a crew to recover MWh that were never lost. When `b < 0`, PI goes above 1 and a one-sided CUSUM never fires. Use a two-sided test plus neighbour comparison, set the loss to 0 for this cause, and route it to `inspection`.
  - **Classifier training needs out-of-sample residuals.** M1's residuals on its own training months are too small, so a classifier trained there sees a distribution it will not meet in production. Use temporal cross-fitting.
  - **Humidity for `icing`** may not be in SCADA; take it from Open-Meteo `relative_humidity_2m`.
  - Set CUSUM thresholds on validation only. 2020 stays untouched.
  - Real, unlabelled faults in 2020 will count as false alarms. Inspect a sample and report this rather than hiding it.

### Step 4: M4 cost map (2 h, E4 part 1)

- **Build:** hub-height extrapolation, M1-W (see the decision above), `cost_mwh`, `cost_mwh_p90`, `workable`, assumed limits in `params.yaml`.
- **Done when:** the map is produced for Kelmarsh (archive) and Fântânele (live forecast); M1-W has its test score; a heatmap shows calm-day valleys.
- **Watch out:** check which wind heights the forecast API provides (§6.3). If the API is down, use the last cache and mark the plan as stale.

### Step 5: Jobs and greedy optimizer (1 h, E4 part 2)

- **Build:** `Task`, revision generator, greedy with grouping and low-confidence routing, `schedule.json` with `cost_breakdown`, `alternatives` and `trace`, `tests/test_optimizer.py`.
- **Done when:** the hand-built test confirms capacity, windows and workable hours are respected, and the cost example in [`design.md`](design.md) (7.6 vs. 19.4 MWh) is reproduced exactly.
- **Watch out:** annual revisions last 2–4 days, but the §9.6 cost formula assumes one day and one crew shift. Model them as consecutive workable days, or the longest jobs can never be placed.

### Step 6: Backtest (1 h, E4 part 3)

- **Build:** day-by-day replay of 2020, with 20–40 injected faults, three policies, two forecast modes.
- **Done when:** `results/backtest.json` and the stacked bar chart show total MWh lost, trips and mean days to repair per policy.
- **Watch out:**
  - **Decide on the forecast, charge on the actuals.** Otherwise the noisy-forecast run cannot show the cost of forecast error.
  - **All policies share the same crews, safety limits and fault set.** "Ignoring weather" means no cost-seeking, not working in unsafe wind. Otherwise the comparison overstates the agent's gain.
  - **Rolling-horizon procrastination:** each morning's plan may push the job to yet another calm day. Watch mean days to repair, and add a maximum deferral in `params.yaml` if it drifts.
  - 2020 is also M1's test set, so freeze all parameters before the first run, and do not tune afterwards.

**Checkpoint:** after this step, the wind path is complete from data to decision, and it is the result that matters most for the jury.

### Step 7: PV branch (3 h, E5)

- **Build:** `pv_ingest.py`, M1 and M2 on Kaggle solar (split 22/5/7 days), PV faults, PV M4, PV jobs in the same queue.
- **Done when:** M1/M2 PV metrics exist and PV jobs appear in the calendar next to wind jobs.
- **Watch out:**
  - **Plant location and time zone:** as far as I recall the dataset does not publish coordinates. pvlib and the rain-reset logic need them, so verify at inspection and, if absent, record an explicit assumption rather than guess silently (rule 10).
  - **The forecast has no module temperature or plant irradiance sensor.** The PV cost map needs the same weather-driven approach as M1-W: `shortwave_radiation` plus a pvlib temperature estimate.
  - **34 days is short** for slow soiling. Report it honestly (§9.8). Comparing the roughly 22 inverters within a plant gives a strong neighbour signal that partly compensates.

### Step 8: Exact optimizer, optional (if time allows)

CP-SAT with the same costs, reporting the gain over greedy (§9.6). Skip it if steps 0–7 ran long.

## After this plan

Stages E6–E8 in §8: M3, the LLM agent and the demo. One known issue for the demo: Fântânele has no public SCADA, so the "map coloured by `pi_72h`" (§14) must replay Kelmarsh turbines under demo IDs, and has to say so on screen.
