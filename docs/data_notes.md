# Data notes

What inspection found (CLAUDE.md rule 1). "Verified" means read from the real files or APIs; "unverified" means taken from the guide and checked by `make inspect` when the data arrives.

## Kelmarsh (verified 2026-11-07, via ranged reads of the Zenodo archives)

- Each annual zip holds 12 files: `Turbine_Data_Kelmarsh_<n>_...csv` and `Status_Kelmarsh_<n>_...csv` for 6 turbines. The turbine files are about 150 MB each.
- **Turbine files:** 299 columns, with an identical header across all turbines and both years. A 9-line `#` comment block comes first, and the header row starts `# Date and time`. Missing values are marked `NaN`.
- **Time:** UTC. The first row of a period starting at 00:00 is stamped 00:00, so timestamps mark the interval **start**. The canonical schema uses the end, so ingest adds 10 minutes.
- **Status files:** event logs with columns `Timestamp start, Timestamp end, Duration, Status, Code, Message, Comment, Service contract category, IEC category`. Many events have `-` as their end, meaning instantaneous.
- **2019 event counts:** `Status`: Informational 58,082, Stop 673, Warning 513, Communication 58. `IEC category`: Full Performance 36,010, Out of Environmental Specification 13,424, Technical Standby 9,363, blank 277, Forced outage 146, Scheduled Maintenance 64, Out of Electrical Specification 42. Top messages are routine (`Automatic start-up`, `System OK`, `Mains operation`), so "no event" cannot be the definition of normal.
- **Static file:** Senvion MM92, 2,050 kW, 92 m rotor. Hub height is 78.5 m, except 68.5 m on turbines 3 and 6.
- **Not in SCADA:** humidity and barometric pressure. These come from Open-Meteo.
- **Vendor outputs in the data:** `Potential power *`, `Lost Production *`. These are never model features, to avoid leaking a vendor's answer. They are used for filtering (curtailment, downtime) and as an independent cross-check.
- **Unit conflict:** the signal mapping lists `Power` in kWh, while the column is `Power (kW)`. `make inspect` compares it with `Energy Export (kWh)`.
- **`Turbine Power setpoint` is not a curtailment limit** (found on real data, 2026-11-07). On turbine 4 in 2019 it has a median of 1,250 kW, ranges from 205 to 2,060 kW, and is below 98% of rated in 84.5% of rows. Filtering on it kept only 16% of rows, mostly high-output ones, so it is not used.
- **`Lost Production to Curtailment`** is populated but was positive in 0.0% of rows on turbine 4 in 2019. Flat power below rated in high wind is caught separately.
- **`Power` is in kW:** median `Energy Export x 6 / Power` is 0.93 for power above 100 kW. It is not an exact match, but it rules out kWh.
- **`Turbine Power setpoint` follows output:** correlation with power is 0.88-0.89 on turbines 1 and 4, and its median rises with wind from about 900 kW to 2,060 kW.
- **Status clauses, 2019, turbines 1 and 4:** `Stop` events flag about 1% of rows and abnormal IEC events about 3%. `Warning` events flag 5-23% of rows at normal power (median about 260-290 kW), so they are not treated as abnormal.

- **What we keep:** `python -m src.data.prepare` keeps all 315,360 rows of 2019 but only 13 of the 299 columns, plus the status events, about 15 MB instead of 311 MB. Values are stored as 32-bit, within 0.0001 kW of the originals. To add a column, add it to `COLUMNS` in that script and run it again; it re-downloads the archive.
- **Analysis of 2019:** see [`notebooks/explore_kelmarsh.ipynb`](../notebooks/explore_kelmarsh.ipynb). 2020 stays unseen until final evaluation.

## Open-Meteo (verified 2026-11-07)

- **Forecast:** 240 hourly values (10 days); wind at 80, 120 and 180 m, gusts at 10 m. Units as requested: m/s, °C, hPa, %, mm, W/m².
- **Archive:** wind at 10 and 100 m, gusts at 10 m; pressure, humidity, precipitation and radiation available for 2019.
- Forecast and archive come from different weather models, so wind differs systematically between them. Reported as a limitation.
- **Archive grid point:** the API returns 52.408 N, 1.002 W, about 4 km from the farm. The values are modelled, so they can't see an individual turbine.
- **Time alignment:** both sources are in UTC. Weather values are spot readings at the hour, while hourly turbine means cover the hour after it, so averaging the readings at each end of the hour is the closest match.

## Kaggle Solar Power Generation Data (verified 2026-11-08)

Archive MD5 `132b49faba3e4a67627f7e19c841a308`, four CSVs. Two plants (4135001, 4136001), 22 inverters each, 15-minute rows from 2020-05-15 00:00 to 2020-06-17 23:45 (34 days), one weather sensor per plant.

- **Dates:** `Plant_1_Generation_Data.csv` alone is day-first (`15-05-2020 00:00`); the other three are ISO (`2020-05-15 00:00:00`). Times are local: irradiation starts about 06:00-06:15 and ends 18:15-18:30, consistent with India Standard Time. No location is given.
- **Power:** `AC_POWER` is kW: integrated over a day it equals `DAILY_YIELD` (ratio 1.000 and 0.999). `DC_POWER` in Plant 1 is 10.2 times AC, a unit error; Plant 2's DC/AC is 1.02. We use AC only.
- **Size:** highest AC per inverter 1,265-1,411 kW (Plant 1) and 1,173-1,385 kW (Plant 2). Rated power is estimated as each plant's highest AC output.
- **Irradiation** is kW/m2 (maximum 1.22 and 1.10), converted to W/m2. Whether it is horizontal or in the module plane is not stated.
- **Temperatures:** module 18-67 °C, ambient 20-39 °C.
- **Gaps:** Plant 1 is short on 20, 21 and 29 May (and its weather sensor on 16, 20, 21, 29 May); Plant 2 lacks 4 of its 22 inverters from 20 to 29 May.
- **Real faults:** in daylight, 17% of Plant 2's rows have zero AC; five inverters are at zero 23-30% of daytime. Some Plant 2 inverters produce up to 22% less per unit of irradiation than the plant median (Plant 1: within 9%). `TOTAL_YIELD` is unreliable (Plant 2 starts at 0; one counter reads 1.2 billion kWh) and is not used.
- **What we keep** (`pv_inverters.parquet`): ts, plant, inverter, ac_kw, irr_wm2, t_mod_c, t_amb_c, rated_kw, and `healthy` = weather present and not zero output above 50 W/m2.
- **Limits:** 34 days of one season, no location, no rain data: soiling resets by rain are seeded, which the README must state.

