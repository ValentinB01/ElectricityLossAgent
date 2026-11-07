"""Build the small data files the project uses, from the public Kelmarsh archive and Open-Meteo.

    python prepare_data.py              # 2019: exploration and training
    python prepare_data.py --year 2020  # The test year. Only once the models are final.

Writes to data/kelmarsh/, then deletes the 300-500 MB zip, since it can be downloaded again:
    turbines_<year>.parquet   All 10-minute rows, only the columns below, plus a `healthy` flag
    events_<year>.parquet     Every status event
    weather_<year>.parquet    Hourly weather for the site
    Kelmarsh_WT_static.csv    Rating, hub height and location per turbine
Weather data by Open-Meteo.com. Kelmarsh data by Cubico Sustainable Investments, CC-BY-4.0.
"""
import argparse
import hashlib
import io
import json
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data" / "kelmarsh"
ZENODO = "https://zenodo.org/records/16807551/files/"
STATIC = ("Kelmarsh_WT_static.csv", "af3a038f0f7fddfc1608ad0bfc8cf5ba")
ARCHIVES = {2019: ("Kelmarsh_SCADA_2019_3085.zip", "14348a1572b722e1be10e53cd145c39a"),
            2020: ("Kelmarsh_SCADA_2020_3086.zip", "e189e02da98f32e74acc72e8559ef0db")}
LAT, LON = 52.4006, -0.9471
WEATHER = "wind_speed_100m,wind_speed_10m,wind_gusts_10m,precipitation,relative_humidity_2m,temperature_2m,surface_pressure"

TIME = "# Date and time"
COLUMNS = {
    # Model inputs and targets (M1, M2)
    "Power (kW)": "power_kw", "Wind speed (m/s)": "wind_ms", "Wind speed, Standard deviation (m/s)": "wind_std_ms",
    "Wind direction (°)": "wind_dir_deg", "Nacelle ambient temperature (°C)": "temp_c",
    # Diagnosing a cause: yaw from nacelle vs. wind direction, pitch faults from blade angle and rotor speed
    "Nacelle position (°)": "nacelle_deg", "Blade angle (pitch position) A (°)": "pitch_deg", "Rotor speed (RPM)": "rotor_rpm",
    # Component health in the notebook's oil filter check
    "Gear oil temperature (°C)": "gear_oil_c",
    # Vendor figures: filtering and cross-checks only, never model inputs
    "Lost Production to Downtime (kWh)": "lost_downtime_kwh", "Lost Production to Curtailment (Total) (kWh)": "lost_curtail_kwh",
    "Lost Production to Performance (kWh)": "lost_perf_kwh", "Potential power default PC (kW)": "vendor_potential_kw",
}
ABNORMAL_IEC = {"Forced outage", "Scheduled Maintenance", "Technical Standby", "Out of Electrical Specification"}


def download(name: str, md5: str) -> Path:
    dest = DATA / name
    if not dest.exists():
        print(f"downloading {name}...", flush=True)
        urllib.request.urlretrieve(f"{ZENODO}{name}?download=1", dest)
    if hashlib.md5(dest.read_bytes()).hexdigest() != md5:
        dest.unlink()
        raise SystemExit(f"{name}: checksum mismatch, file deleted. Run again.")
    return dest


def member(z: zipfile.ZipFile, kind: str, n: int) -> str:
    return next(x for x in z.namelist() if re.match(rf"{kind}_Kelmarsh_{n}_", x))


def read_events(z: zipfile.ZipFile, n: int, turbine: str) -> pd.DataFrame:
    text = z.read(member(z, "Status", n)).decode("utf8", "replace")
    skip = next(i for i, line in enumerate(text.split("\n")) if line.startswith("Timestamp start"))
    ev = pd.read_csv(io.StringIO(text), skiprows=skip, keep_default_na=False)
    start = pd.to_datetime(ev["Timestamp start"], errors="coerce")
    end = pd.to_datetime(ev["Timestamp end"].replace("-", ""), errors="coerce")
    out = pd.DataFrame({"turbine": turbine, "Status": ev["Status"], "Code": ev["Code"], "Message": ev["Message"], "IEC category": ev["IEC category"],
                        "start": start, "end": end, "hours": (end - start).dt.total_seconds() / 3600})
    return out.dropna(subset=["start"])


def read_turbine(z: zipfile.ZipFile, n: int, turbine: str, events: pd.DataFrame) -> pd.DataFrame:
    raw = z.read(member(z, "Turbine_Data", n))
    skip = next(i for i, line in enumerate(raw[:40000].decode("utf8", "replace").split("\n")) if line.startswith(TIME))
    d = pd.read_csv(io.BytesIO(raw), skiprows=skip, usecols=[TIME, *COLUMNS], na_values=["NaN"]).rename(columns=COLUMNS)
    d["ts"] = pd.to_datetime(d.pop(TIME))  # UTC, marks the START of the 10-minute interval.
    d = d.set_index("ts").sort_index()
    # Abnormal: overlaps a real stop or an abnormal IEC state, or the vendor reports downtime or curtailment loss.
    # The turbine's power setpoint is not used: it follows output, so it says nothing about curtailment.
    flag = np.zeros(len(d), dtype=bool)
    bad = events[(events["Status"] == "Stop") | events["IEC category"].isin(ABNORMAL_IEC)]
    grid, step = d.index.to_numpy(), np.timedelta64(10, "m")
    for s, e in zip(bad["start"].to_numpy(), bad["end"].fillna(bad["start"]).to_numpy()):
        a = np.searchsorted(grid, s - step, side="right")
        b = np.searchsorted(grid, e, side="left") if e > s else a + 1
        flag[a:max(b, a + 1)] = True
    d["healthy"] = ~(flag | d["power_kw"].isna().to_numpy() | d["wind_ms"].isna().to_numpy()
                     | d["lost_downtime_kwh"].fillna(0).gt(0).to_numpy() | d["lost_curtail_kwh"].fillna(0).gt(0).to_numpy())
    d["turbine"] = turbine
    numeric = list(COLUMNS.values())
    d[numeric] = d[numeric].astype("float32")
    return d.reset_index()


def get_weather(year: int) -> pd.DataFrame:
    query = urllib.parse.urlencode({"latitude": LAT, "longitude": LON, "start_date": f"{year}-01-01", "end_date": f"{year}-12-31", "hourly": WEATHER, "wind_speed_unit": "ms", "timezone": "UTC"})
    with urllib.request.urlopen(f"https://archive-api.open-meteo.com/v1/archive?{query}", timeout=120) as r:
        w = pd.DataFrame(json.load(r)["hourly"])
    w["ts_utc"] = pd.to_datetime(w.pop("time"), utc=True)
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2019, choices=sorted(ARCHIVES))
    ap.add_argument("--keep-zip", action="store_true", help="Do not delete the downloaded archive.")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    download(*STATIC)
    zip_path = download(*ARCHIVES[a.year])
    facts = pd.read_csv(DATA / STATIC[0], encoding="utf-8-sig")
    names = dict(zip(facts["Title"].str.extract(r"(\d+)$")[0].astype(int), facts["Alternative Title"]))
    with zipfile.ZipFile(zip_path) as z:
        events = pd.concat([read_events(z, n, t) for n, t in names.items()], ignore_index=True)
        turbines = pd.concat([read_turbine(z, n, t, events[events.turbine == t]) for n, t in names.items()], ignore_index=True)
    turbines.to_parquet(DATA / f"turbines_{a.year}.parquet", compression="zstd")
    events.to_parquet(DATA / f"events_{a.year}.parquet", compression="zstd")
    wx = DATA / f"weather_{a.year}.parquet"
    if not wx.exists() or set(WEATHER.split(",")) - set(pd.read_parquet(wx).columns):  # Fetched again when a later step needs a new variable.
        get_weather(a.year).to_parquet(wx)
    if not a.keep_zip:
        zip_path.unlink()
    print(f"{a.year}: {len(turbines):,} rows, {turbines.turbine.nunique()} turbines, {len(COLUMNS)} data columns, {turbines.healthy.mean():.1%} healthy, {len(events):,} events")
    for f in sorted(DATA.glob(f"*{a.year}*")):
        print(f"  {f.stat().st_size / 1e6:8.1f} MB  {f.name}")


if __name__ == "__main__":
    main()
