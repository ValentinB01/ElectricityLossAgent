# Producers module: step-by-step plan

**Rule:** the repo holds only what the current step uses, and a step is finished only when its check passes on real data. Code drafted for later steps sits outside the repo in `../SmartHackPPC-parked/` and comes back, trimmed, when its step starts. Specs are in [`CLAUDE.md`](../CLAUDE.md). Commits are decided with Valentin.

| Step | Builds | Done when | Status |
|---|---|---|---|
| 1 | **Data**: [`prepare_data.py`](../prepare_data.py) and [`notebooks/explore_kelmarsh.ipynb`](../notebooks/explore_kelmarsh.ipynb) | Valentin has reviewed the data and its findings | **For review** |
| 2 | **M1** expected production | On real data, beats the binned curve on Nov–Dec 2019 and 10–90% coverage is near 80% | Next |
| 3 | **M2** underperformance: fault injection, CUSUM, cause classifier | Beats the fixed threshold on detection and false alarms, on validation | Drafted, parked |
| 4 | **M4** downtime cost: weather-only M1-W, cost map | M1-W beats its baseline; a real forecast shows calm-day valleys | Drafted, parked |
| 5 | **Optimizer**: revisions, crews, calendar | Hand-built tests pass and the 7.6 vs. 19.4 MWh example is reproduced; a real calendar comes out | Drafted, parked |
| 6 | **Backtest**: one year under three policies | Results for perfect and noisy forecasts, run once with settings frozen | Not started |
| 7 | **PV**: M1, M2, M4 on the Kaggle data | PV jobs appear in the same calendar | Partly drafted, parked |

## What each step must not forget

- **Step 2:** the healthy filter keeps 97% of rows and must not use the setpoint. Nov–Dec is outside the training months, so the seasonal feature meets a season it hasn't seen: test with and without it. Most error will sit at 7–11 m/s.
- **Step 3:** compare each turbine's deviation from expected power, not raw power, because healthy turbines differ by up to 12% a month against the fleet. An anemometer fault loses no energy, so its loss is 0 and it gets a check rather than a repair. The test must be two-sided, or a low-reading sensor never fires. Train the classifier on out-of-sample residuals.
- **Step 4:** hub height differs per turbine (68.5 m on turbines 3 and 6, 78.5 m on the rest). Forecast and archive come from different weather models. Average the weather readings at each end of the hour to match hourly turbine means.
- **Step 5:** annual revisions span several days, so they take consecutive workable shifts. Real `Scheduled Maintenance` is about 20 h per turbine per year against our assumed 40 h, so revisit it. M3 is out of scope, so the risk term is 0.
- **Step 6:** decide on the forecast and charge on the actual weather. All policies share crews, safety limits and faults. 2020 is the test year, so freeze all settings before the first run.
- **Step 7:** the Kaggle plants have no known location and only 34 days of data. No public SCADA exists for PPC's parks, so demo detections are replayed and must be labelled as such.
