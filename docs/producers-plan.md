# Producers module: step-by-step plan

**Rule:** the repo holds only what the current step uses, and a step is finished only when its check passes on real data. Code drafted for later steps sits outside the repo in `../SmartHackPPC-parked/` and comes back, trimmed, when its step starts. Specs are in [`CLAUDE.md`](../CLAUDE.md). Commits are decided with Valentin.

| Step | Builds | Done when | Status |
|---|---|---|---|
| 1 | **Data**: [`prepare_data.py`](../prepare_data.py) and [`notebooks/explore_kelmarsh.ipynb`](../notebooks/explore_kelmarsh.ipynb) | Valentin has reviewed the data and its findings | Done |
| 2 | **M1** expected production: [`m1_expected.py`](../m1_expected.py) | On real data, beats the binned curve on Nov–Dec 2019 and 10–90% coverage is near 80% | **Done:** error 2.10% vs. 2.46% of rated, coverage 78.8% |
| 3 | **M2** underperformance: fault injection, CUSUM, cause classifier | Beats the fixed threshold on detection and false alarms, on validation | **Next** (drafted, parked) |
| 4 | **M4** downtime cost: weather-only M1-W, cost map | M1-W beats its baseline; a real forecast shows calm-day valleys | Drafted, parked |
| 5 | **Optimizer**: revisions, crews, calendar | Hand-built tests pass and the 7.6 vs. 19.4 MWh example is reproduced; a real calendar comes out | Drafted, parked |
| 6 | **Backtest**: one year under three policies | Results for perfect and noisy forecasts, run once with settings frozen | Not started |
| 7 | **PV**: M1, M2, M4 on the Kaggle data | PV jobs appear in the same calendar | Partly drafted, parked |

## What each step must not forget

- **Step 2:** the healthy filter keeps 97% of rows and must not use the setpoint. Validation also chose the inputs and the number of trees, so its score is slightly optimistic. The 2020 score is still to come: `--eval --final`, once, with settings frozen.
- **Step 3:** compare each turbine's deviation from expected power, not raw power, because healthy turbines differ by up to 12% a month against the fleet. An anemometer fault loses no energy, so its loss is 0 and it gets a check rather than a repair. The test must be two-sided, or a low-reading sensor never fires. Train the classifier on out-of-sample residuals. Healthy turbines' 72 h index spans 0.986–1.031 (median 1.012), partly because M1 runs up to 0.7% of rated low in winter and 0.3% high in summer. Against the fleet the span narrows to 0.984–1.018, so a yaw fault under about 8° (1.9% loss) may not show. M1 with the month input had lower bias: recheck it here.
- **Step 4:** hub height differs per turbine (68.5 m on turbines 3 and 6, 78.5 m on the rest). Forecast and archive come from different weather models. Average the weather readings at each end of the hour to match hourly turbine means. M1 can return slightly negative power for an idle turbine, which is real, so clip at 0 when turning it into a cost.
- **Step 5:** annual revisions span several days, so they take consecutive workable shifts. Real `Scheduled Maintenance` is about 20 h per turbine per year against our assumed 40 h, so revisit it. M3 is out of scope, so the risk term is 0.
- **Step 6:** decide on the forecast and charge on the actual weather. All policies share crews, safety limits and faults. 2020 is the test year, so freeze all settings before the first run.
- **Step 7:** the Kaggle plants have no known location and only 34 days of data. No public SCADA exists for PPC's parks, so demo detections are replayed and must be labelled as such.
