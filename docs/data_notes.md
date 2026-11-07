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

- **What we keep:** `python prepare_data.py` keeps all 315,360 rows of 2019 but only 13 of the 299 columns, plus the status events, about 15 MB instead of 311 MB. Values are stored as 32-bit, within 0.0001 kW of the originals. To add a column, add it to `COLUMNS` in that script and run it again; it re-downloads the archive.
- **Analysis of 2019:** see [`notebooks/explore_kelmarsh.ipynb`](../notebooks/explore_kelmarsh.ipynb). 2020 stays unseen until final evaluation.

## Open-Meteo (verified 2026-11-07)

- **Forecast:** 240 hourly values (10 days); wind at 80, 120 and 180 m, gusts at 10 m. Units as requested: m/s, °C, hPa, %, mm, W/m².
- **Archive:** wind at 10 and 100 m, gusts at 10 m; pressure, humidity, precipitation and radiation available for 2019.
- Forecast and archive come from different weather models, so wind differs systematically between them. Reported as a limitation.
- **Archive grid point:** the API returns 52.408 N, 1.002 W, about 4 km from the farm. The values are modelled, so they can't see an individual turbine.
- **Time alignment:** both sources are in UTC. Weather values are spot readings at the hour, while hourly turbine means cover the hour after it, so averaging the readings at each end of the hour is the closest match.

## Kaggle Solar Power Generation Data (unverified: no token during development)

The guide's column hints are validated at read time. `make inspect` reports irradiance range, DC/AC ratio, thin days, and whether the irradiance peak fits the assumed time zone. Plant coordinates and time zone are assumptions in `config/sites.yaml`. They affect only solar-angle features, and rain resets are seeded since there is no location to fetch rain for.
