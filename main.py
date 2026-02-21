"""
EB Bill Prediction - FastAPI Backend v3
New in v3:
  Multi-tariff bill calculator (DOMESTIC / COMMERCIAL / COTTAGE / POWERLOOM etc.)
  Matches TNPDCL online calculator logic exactly
  POST /bill/calculate — standalone calculator (mirrors the HTML form)
  Appliance features fed into ML prediction
  Household stores tariff_category, billing_cycle, size_tag
  All existing endpoints preserved
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta, date
import sqlite3, json, pickle, uuid, os
import numpy as np
import pandas as pd
from contextlib import contextmanager

TARIFF_CONFIGS = {
    "DOMESTIC": {
        "display_name": "DOMESTIC",
        "slabs": [(0,100,0.0),(100,200,1.5),(200,500,3.0),(500,1000,5.0),(1000,float('inf'),7.0)],
        "fixed_charge": 30.0, "meter_rent": 10.0, "duty_rate": 0.16, "default_cycle": "Bi-monthly",
    },
    "COMMERCIAL": {
        "display_name": "COMMERCIAL",
        "slabs": [(0,100,3.5),(100,500,5.0),(500,float('inf'),7.0)],
        "fixed_charge": 75.0, "meter_rent": 20.0, "duty_rate": 0.20, "default_cycle": "Monthly",
    },
    "COTTAGE": {
        "display_name": "COTTAGE AND TINY INDUSTRIES",
        "slabs": [(0,100,0.0),(100,250,2.0),(250,float('inf'),3.5)],
        "fixed_charge": 20.0, "meter_rent": 10.0, "duty_rate": 0.16, "default_cycle": "Monthly",
    },
    "POWERLOOM": {
        "display_name": "POWERLOOM",
        "slabs": [(0,300,3.0),(300,float('inf'),4.5)],
        "fixed_charge": 50.0, "meter_rent": 15.0, "duty_rate": 0.16, "default_cycle": "Monthly",
    },
    "PUBLIC_WORSHIP": {
        "display_name": "ACTUAL PLACES OF PUBLIC WORSHIP",
        "slabs": [(0,100,0.0),(100,float('inf'),2.5)],
        "fixed_charge": 20.0, "meter_rent": 10.0, "duty_rate": 0.10, "default_cycle": "Bi-monthly",
    },
    "EDUCATION": {
        "display_name": "RECG-EDU-INSTN/INSTN-DECLARED BY GOVT.",
        "slabs": [(0,500,2.5),(500,float('inf'),5.0)],
        "fixed_charge": 40.0, "meter_rent": 15.0, "duty_rate": 0.16, "default_cycle": "Monthly",
    },
    "DOMESTIC_COMMON": {
        "display_name": "DOMESTIC COMMON SUPPLY",
        "slabs": [(0,100,0.0),(100,300,2.0),(300,float('inf'),4.0)],
        "fixed_charge": 25.0, "meter_rent": 10.0, "duty_rate": 0.16, "default_cycle": "Bi-monthly",
    },
}

BILLING_CYCLE_DAYS_MAP = {"Bi-monthly": 60, "Monthly": 30}


def calculate_bill(units: float, tariff: str = "DOMESTIC",
                   cycle: str = "Bi-monthly", contracted_load_kw: float = 0.0) -> dict:
    cfg = TARIFF_CONFIGS.get(tariff, TARIFF_CONFIGS["DOMESTIC"])
    divisor = 1 if cycle == "Bi-monthly" else 2
    slabs = [(lo/divisor, (hi/divisor if hi != float('inf') else float('inf')), rate)
             for (lo, hi, rate) in cfg["slabs"]]
    energy, rem = 0.0, max(0.0, units)
    slab_breakdown = []
    for lo, hi, rate in slabs:
        if rem <= 0:
            break
        chunk = min(rem, (hi - lo) if hi != float('inf') else rem)
        charge = chunk * rate
        energy += charge
        slab_breakdown.append({
            "slab": f"{int(lo)}-{int(hi) if hi != float('inf') else 'INF'}",
            "units": round(chunk, 2), "rate": rate, "charge": round(charge, 2),
        })
        rem -= chunk
    sub = energy + cfg["fixed_charge"] + cfg["meter_rent"]
    duty = round(sub * cfg["duty_rate"], 2)
    demand = round(contracted_load_kw * 30.0, 2) if contracted_load_kw > 0 and tariff not in ("DOMESTIC","PUBLIC_WORSHIP") else 0.0
    return {
        "units": round(units, 2), "tariff": tariff,
        "tariff_display": cfg.get("display_name", tariff), "cycle": cycle,
        "energy_charge": round(energy, 2), "fixed_charge": cfg["fixed_charge"],
        "meter_rent": cfg["meter_rent"], "electricity_duty": duty,
        "demand_charge": demand, "total_bill": round(sub + duty + demand, 2),
        "slab_breakdown": slab_breakdown,
    }


MONTHLY_SEASONAL = {1:8.0,2:8.5,3:10.5,4:13.0,5:14.5,6:11.0,7:9.5,8:9.0,9:9.5,10:10.0,11:8.5,12:7.5}
MONTHLY_TEMP = {1:26,2:27,3:30,4:33,5:35,6:34,7:33,8:33,9:32,10:29,11:27,12:25}

model = None
feature_cols = None
model_meta = {}
MODEL_PATH = "model_artifacts/xgb_model.pkl"
META_PATH  = "model_artifacts/metadata.json"


def load_model():
    global model, feature_cols, model_meta
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(META_PATH) as f:
            model_meta = json.load(f)
        feature_cols = model_meta["feature_cols"]
        print(f"Model loaded R2={model_meta.get('val_r2','N/A')} version={model_meta.get('model_version','v1')}")
    else:
        print("No model found - seasonal fallback active")


def _predict_one_day(features: dict) -> float:
    if model and feature_cols:
        X = pd.DataFrame([{c: features.get(c, 0) for c in feature_cols}])
        return float(model.predict(X)[0])
    return MONTHLY_SEASONAL.get(features.get("month", 6), 10.0)


def build_forecast(recent_kwh, kwh_already, days_remaining,
                   start_date=None, app_feats=None, tariff="DOMESTIC", cycle="Bi-monthly"):
    if start_date is None:
        start_date = date.today() + timedelta(days=1)
    if app_feats is None:
        app_feats = {}
    buf = list(recent_kwh) if recent_kwh else [10.0]
    cumulative = kwh_already
    daily_out = []
    for i in range(days_remaining):
        d = start_date + timedelta(days=i)
        mon = d.month
        dow = d.weekday()
        features = {
            "month": mon, "day_of_week": dow, "is_weekend": 1 if dow >= 5 else 0,
            "day_of_year": d.timetuple().tm_yday, "week_of_year": int(d.strftime("%W")),
            "quarter": (mon - 1) // 3 + 1,
            "is_summer": 1 if mon in [3,4,5] else 0,
            "is_monsoon": 1 if mon in [6,7,8,9,10] else 0,
            "is_winter": 1 if mon in [11,12,1,2] else 0,
            "avg_temperature": MONTHLY_TEMP.get(mon, 30),
            "lag_1d": buf[-1],
            "lag_7d": buf[-7] if len(buf) >= 7 else buf[0],
            "lag_30d": buf[-30] if len(buf) >= 30 else buf[0],
            "rolling_7": float(np.mean(buf[-7:])),
            "rolling_14": float(np.mean(buf[-14:])),
            "rolling_30": float(np.mean(buf[-30:])),
            "std_7": float(np.std(buf[-7:])) if len(buf) >= 2 else 0.5,
            "cycle_day": i + 1, "cycle_cumulative_kwh": cumulative,
            "projected_cycle_kwh": cumulative + float(np.mean(buf[-7:])) * (days_remaining - i),
            "num_appliance_types": app_feats.get("num_appliance_types", 0),
            "total_appliance_count": app_feats.get("total_appliance_count", 0),
            "total_rated_watts": app_feats.get("total_rated_watts", 0),
            "appliance_daily_kwh": app_feats.get("appliance_daily_kwh", 0),
            "ac_count": app_feats.get("ac_count", 0),
            "heater_count": app_feats.get("heater_count", 0),
            "fan_count": app_feats.get("fan_count", 0),
            "light_count": app_feats.get("light_count", 0),
            "heavy_appliance_count": app_feats.get("heavy_appliance_count", 0),
            "has_ac": app_feats.get("has_ac", 0),
            "has_heater": app_feats.get("has_heater", 0),
            "tariff_code": app_feats.get("tariff_code", 0),
            "size_code": app_feats.get("size_code", 1),
        }
        pred = max(0.0, _predict_one_day(features))
        cumulative += pred
        buf.append(pred)
        daily_out.append({
            "date": str(d), "predicted_kwh": round(pred, 3),
            "cumulative_kwh": round(cumulative, 3),
            "bill_if_stopped_today": calculate_bill(cumulative, tariff, cycle)["total_bill"],
        })
    return {
        "daily_forecast": daily_out,
        "projected_total_kwh": round(cumulative, 2),
        "projected_bill": calculate_bill(cumulative, tariff, cycle),
    }


DB_PATH = "eb_data.db"


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS households (
            id TEXT PRIMARY KEY, name TEXT, consumer_number TEXT,
            tariff_category TEXT DEFAULT 'DOMESTIC',
            billing_cycle TEXT DEFAULT 'Bi-monthly',
            size_tag TEXT DEFAULT 'medium',
            billing_cycle_start DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS iot_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT, household_id TEXT,
            timestamp TIMESTAMP, cumulative_kwh REAL, hourly_kwh REAL,
            voltage REAL, current_amps REAL, power_factor REAL,
            source TEXT DEFAULT 'iot', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS daily_summary (
            household_id TEXT, date DATE, daily_kwh REAL,
            peak_hourly_kwh REAL, avg_voltage REAL, reading_count INTEGER,
            PRIMARY KEY (household_id, date)
        );
        CREATE TABLE IF NOT EXISTS appliances (
            id TEXT PRIMARY KEY, household_id TEXT, name TEXT, icon TEXT DEFAULT '💡',
            watts REAL, hours_per_day REAL, quantity INTEGER DEFAULT 1,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS usage_limits (
            household_id TEXT PRIMARY KEY, daily_kwh_limit REAL,
            cycle_kwh_limit REAL, daily_bill_limit REAL, cycle_bill_limit REAL,
            alert_enabled INTEGER DEFAULT 1, updated_at TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS limit_breaches (
            id INTEGER PRIMARY KEY AUTOINCREMENT, household_id TEXT,
            breach_type TEXT, limit_value REAL, actual_value REAL,
            breach_date DATE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
    print("Database initialized")


class HouseholdCreate(BaseModel):
    id: str
    name: str
    consumer_number: Optional[str] = ""
    tariff_category: Optional[str] = "DOMESTIC"
    billing_cycle: Optional[str] = "Bi-monthly"
    size_tag: Optional[str] = "medium"
    billing_cycle_start: Optional[str] = None


class IoTReading(BaseModel):
    household_id: str
    cumulative_kwh: float
    voltage: Optional[float] = None
    current_amps: Optional[float] = None
    power_factor: Optional[float] = None
    timestamp: Optional[str] = None


class IFTTTWebhook(BaseModel):
    value1: str
    value2: Optional[str] = None
    value3: Optional[str] = None


class ApplianceCreate(BaseModel):
    id: Optional[str] = None
    household_id: str
    name: str
    icon: Optional[str] = "💡"
    watts: float
    hours_per_day: float
    quantity: Optional[int] = 1
    is_active: Optional[bool] = True


class ApplianceUpdate(BaseModel):
    name: Optional[str] = None
    icon: Optional[str] = None
    watts: Optional[float] = None
    hours_per_day: Optional[float] = None
    quantity: Optional[int] = None
    is_active: Optional[bool] = None


class UsageLimitSet(BaseModel):
    daily_kwh_limit: Optional[float] = None
    cycle_kwh_limit: Optional[float] = None
    daily_bill_limit: Optional[float] = None
    cycle_bill_limit: Optional[float] = None
    alert_enabled: Optional[bool] = True


class BillCalcRequest(BaseModel):
    units: float
    tariff: Optional[str] = "DOMESTIC"
    cycle: Optional[str] = "Bi-monthly"
    contracted_load_kw: Optional[float] = 0.0


app = FastAPI(title="EB Bill Prediction API v3", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    init_db()
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None,
            "model_version": model_meta.get("model_version", "none"),
            "timestamp": datetime.now().isoformat()}


# BILL CALCULATOR - mirrors TNPDCL HTML form
@app.post("/bill/calculate")
def bill_calculate(req: BillCalcRequest):
    """Standalone TNEB bill calculator. Matches TNPDCL online form logic."""
    if req.tariff not in TARIFF_CONFIGS:
        raise HTTPException(400, f"Unknown tariff '{req.tariff}'. Valid: {list(TARIFF_CONFIGS)}")
    if req.cycle not in ("Bi-monthly", "Monthly"):
        raise HTTPException(400, "cycle must be 'Bi-monthly' or 'Monthly'")
    return calculate_bill(req.units, req.tariff, req.cycle, req.contracted_load_kw or 0.0)


@app.get("/bill/tariffs")
def list_tariffs():
    """All tariff categories (matches TNPDCL dropdown)."""
    return {
        k: {
            "display_name": v["display_name"],
            "default_cycle": v["default_cycle"],
            "fixed_charge": v["fixed_charge"],
            "meter_rent": v["meter_rent"],
            "duty_rate_pct": round(v["duty_rate"] * 100),
            "slabs": [{"from": s[0], "to": s[1] if s[1] != float('inf') else None, "rate": s[2]}
                      for s in v["slabs"]],
        }
        for k, v in TARIFF_CONFIGS.items()
    }


@app.post("/household", status_code=201)
def create_household(h: HouseholdCreate):
    cycle_start = h.billing_cycle_start or date.today().replace(day=1).isoformat()
    with get_db() as conn:
        try:
            conn.execute(
                "INSERT INTO households (id,name,consumer_number,tariff_category,billing_cycle,size_tag,billing_cycle_start) VALUES (?,?,?,?,?,?,?)",
                (h.id, h.name, h.consumer_number or "", h.tariff_category or "DOMESTIC",
                 h.billing_cycle or "Bi-monthly", h.size_tag or "medium", cycle_start))
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Household ID exists")
    return {"status": "created", "id": h.id}


@app.get("/household/{household_id}")
def get_household(household_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM households WHERE id=?", (household_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Not found")
    return dict(row)


@app.post("/iot/push")
async def iot_push(reading: IoTReading, bg: BackgroundTasks):
    ts = reading.timestamp or datetime.now().isoformat()
    ts_dt = datetime.fromisoformat(ts)
    with get_db() as conn:
        prev = conn.execute(
            "SELECT cumulative_kwh FROM iot_readings WHERE household_id=? ORDER BY timestamp DESC LIMIT 1",
            (reading.household_id,)).fetchone()
        prev_cum = float(prev["cumulative_kwh"]) if prev else reading.cumulative_kwh
        hourly_kwh = max(0.0, reading.cumulative_kwh - prev_cum)
        conn.execute(
            "INSERT INTO iot_readings (household_id,timestamp,cumulative_kwh,hourly_kwh,voltage,current_amps,power_factor) VALUES (?,?,?,?,?,?,?)",
            (reading.household_id, ts, reading.cumulative_kwh, round(hourly_kwh, 5),
             reading.voltage, reading.current_amps, reading.power_factor))
        conn.commit()
    bg.add_task(_update_daily, reading.household_id, ts_dt.date())
    bg.add_task(_check_limits, reading.household_id, ts_dt.date())
    return {"status": "received", "hourly_kwh": round(hourly_kwh, 4), "timestamp": ts}


@app.post("/iot/webhook")
async def iot_webhook(payload: IFTTTWebhook, bg: BackgroundTasks):
    try:
        cum = float(payload.value2 or 0)
        v   = float(payload.value3) if payload.value3 else None
    except ValueError:
        raise HTTPException(400, "value2 must be numeric")
    return await iot_push(IoTReading(household_id=payload.value1, cumulative_kwh=cum, voltage=v), bg)


def _get_hh_or_404(conn, hid):
    row = conn.execute("SELECT * FROM households WHERE id=?", (hid,)).fetchone()
    if not row:
        raise HTTPException(404, f"Household '{hid}' not found")
    return dict(row)


def _cycle_rows(conn, hid, cycle_start):
    rows = conn.execute(
        "SELECT date, daily_kwh FROM daily_summary WHERE household_id=? AND date>=? ORDER BY date",
        (hid, cycle_start)).fetchall()
    return [dict(r) for r in rows]


def _recent_kwh(conn, hid, n=35):
    rows = conn.execute(
        "SELECT daily_kwh FROM daily_summary WHERE household_id=? ORDER BY date DESC LIMIT ?",
        (hid, n)).fetchall()
    return [float(r["daily_kwh"]) for r in reversed(rows)]


def _get_app_feats(conn, hid):
    rows = conn.execute(
        "SELECT * FROM appliances WHERE household_id=? AND is_active=1", (hid,)).fetchall()
    if not rows:
        return {}
    apps = [dict(r) for r in rows]
    return {
        "num_appliance_types": len(apps),
        "total_appliance_count": sum(a["quantity"] for a in apps),
        "total_rated_watts": sum(a["watts"] * a["quantity"] for a in apps),
        "appliance_daily_kwh": sum(a["watts"] * a["quantity"] * a["hours_per_day"] / 1000 for a in apps),
        "ac_count": sum(a["quantity"] for a in apps if "ac" in a["name"].lower() or "air_cond" in a["name"].lower()),
        "heater_count": sum(a["quantity"] for a in apps if "heater" in a["name"].lower() or "geyser" in a["name"].lower()),
        "fan_count": sum(a["quantity"] for a in apps if "fan" in a["name"].lower()),
        "light_count": sum(a["quantity"] for a in apps if any(x in a["name"].lower() for x in ("bulb","light","lamp"))),
        "heavy_appliance_count": sum(a["quantity"] for a in apps if a["watts"] >= 1000),
        "has_ac": 1 if any("ac" in a["name"].lower() or "air_cond" in a["name"].lower() for a in apps) else 0,
        "has_heater": 1 if any("heater" in a["name"].lower() for a in apps) else 0,
    }


TARIFF_CODE = {"DOMESTIC":0,"COMMERCIAL":1,"COTTAGE":2,"POWERLOOM":3}
SIZE_CODE   = {"small":0,"medium":1,"large":2,"commercial":3,"cottage":4}


@app.get("/summary/{household_id}")
def get_summary(household_id: str):
    with get_db() as conn:
        hh       = _get_hh_or_404(conn, household_id)
        tariff   = hh.get("tariff_category", "DOMESTIC")
        cycle    = hh.get("billing_cycle",   "Bi-monthly")
        size_tag = hh.get("size_tag",        "medium")
        cycle_days = BILLING_CYCLE_DAYS_MAP.get(cycle, 60)
        today = date.today()
        try:
            cs = date.fromisoformat(str(hh.get("billing_cycle_start") or today.replace(day=1).isoformat()))
        except Exception:
            cs = today.replace(day=1)
        days_elapsed   = max(1, (today - cs).days)
        days_remaining = max(0, cycle_days - days_elapsed)
        cycle_rows     = _cycle_rows(conn, household_id, str(cs))
        kwh_so_far     = sum(r["daily_kwh"] for r in cycle_rows)
        recent         = _recent_kwh(conn, household_id, 35)
        latest         = conn.execute(
            "SELECT cumulative_kwh, voltage, timestamp FROM iot_readings WHERE household_id=? ORDER BY timestamp DESC LIMIT 1",
            (household_id,)).fetchone()
        today_row = conn.execute(
            "SELECT COALESCE(SUM(hourly_kwh),0) as t FROM iot_readings WHERE household_id=? AND date(timestamp)=?",
            (household_id, str(today))).fetchone()
        today_kwh = float(today_row["t"]) if today_row else 0.0
        app_feats = _get_app_feats(conn, household_id)
        app_feats["tariff_code"] = TARIFF_CODE.get(tariff, 0)
        app_feats["size_code"]   = SIZE_CODE.get(size_tag, 1)
        limits = conn.execute("SELECT * FROM usage_limits WHERE household_id=?", (household_id,)).fetchone()

    bill_to_date = calculate_bill(kwh_so_far, tariff, cycle)
    forecast     = build_forecast(recent, kwh_so_far, days_remaining,
                                  today + timedelta(days=1), app_feats, tariff, cycle)
    return {
        "household_id": household_id, "tariff": tariff,
        "billing_cycle": cycle, "size_tag": size_tag,
        "cycle_start": str(cs), "days_elapsed": days_elapsed, "days_remaining": days_remaining,
        "live": {
            "cumulative_kwh": float(latest["cumulative_kwh"]) if latest else 0,
            "voltage": float(latest["voltage"]) if latest and latest["voltage"] else None,
            "last_reading_at": latest["timestamp"] if latest else None,
            "today_kwh": round(today_kwh, 3),
        },
        "cycle": {
            "kwh_so_far": round(kwh_so_far, 2),
            "daily_avg_kwh": round(kwh_so_far / days_elapsed, 3),
            "bill_to_date": bill_to_date,
        },
        "forecast": forecast,
        "limits": dict(limits) if limits else None,
        "appliance_features_used": bool(app_feats),
    }


@app.get("/dashboard/{household_id}")
def dashboard(household_id: str):
    return get_summary(household_id)


@app.get("/predict/{household_id}")
def predict(household_id: str, days_ahead: int = Query(default=30, ge=1, le=60)):
    with get_db() as conn:
        hh     = _get_hh_or_404(conn, household_id)
        tariff = hh.get("tariff_category", "DOMESTIC")
        cycle  = hh.get("billing_cycle",   "Bi-monthly")
        try:
            cs = date.fromisoformat(str(hh.get("billing_cycle_start") or date.today().replace(day=1).isoformat()))
        except Exception:
            cs = date.today().replace(day=1)
        kwh_so_far = sum(r["daily_kwh"] for r in _cycle_rows(conn, household_id, str(cs)))
        recent     = _recent_kwh(conn, household_id, 35)
        app_feats  = _get_app_feats(conn, household_id)
        app_feats["tariff_code"] = TARIFF_CODE.get(tariff, 0)
        app_feats["size_code"]   = SIZE_CODE.get(hh.get("size_tag","medium"), 1)
    forecast = build_forecast(recent, kwh_so_far, days_ahead,
                              date.today() + timedelta(days=1), app_feats, tariff, cycle)
    return {
        "household_id": household_id, "tariff": tariff, "billing_cycle": cycle,
        "kwh_already_in_cycle": round(kwh_so_far, 2),
        "forecast_days": days_ahead,
        "predicted_total_kwh": forecast["projected_total_kwh"],
        "predicted_bill": forecast["projected_bill"],
        "daily_forecast": forecast["daily_forecast"],
    }


@app.get("/history/{household_id}")
def history(household_id: str, days: int = Query(default=30, ge=1, le=365)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT date, daily_kwh, peak_hourly_kwh, avg_voltage, reading_count FROM daily_summary WHERE household_id=? AND date >= date('now', ?) ORDER BY date DESC",
            (household_id, f"-{days} days")).fetchall()
    return {"household_id": household_id, "days": days, "records": [dict(r) for r in rows]}


@app.get("/peak/{household_id}")
def peak_analysis(household_id: str, days: int = Query(default=90, ge=7, le=365)):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT date(timestamp) as d, strftime('%H',timestamp) as hr,
                   strftime('%w',timestamp) as dow, hourly_kwh
            FROM iot_readings WHERE household_id=? AND date(timestamp) >= date('now',?)
            ORDER BY timestamp
        """, (household_id, f"-{days} days")).fetchall()
    if not rows:
        return {"household_id": household_id, "message": "No IoT data"}
    df = pd.DataFrame([dict(r) for r in rows])
    df["hourly_kwh"] = df["hourly_kwh"].astype(float)
    df["hr"]  = df["hr"].astype(int)
    df["dow"] = df["dow"].astype(int)
    daily = df.groupby("d")["hourly_kwh"].sum()
    peak_day = {"date": daily.idxmax(), "kwh": round(float(daily.max()), 3)}
    by_hour  = {k: round(float(v), 4) for k, v in df.groupby("hr")["hourly_kwh"].mean().items()}
    dow_names = {0:"Sun",1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat"}
    by_dow   = {dow_names[int(k)]: round(float(v), 3) for k, v in df.groupby("dow")["hourly_kwh"].mean().items()}
    peak_h   = max(by_hour, key=by_hour.get)
    peak_d   = max(by_dow,  key=by_dow.get)
    return {
        "household_id": household_id, "analysis_days": days,
        "peak_day": peak_day,
        "peak_hour": {"hour": peak_h, "avg_kwh": by_hour[peak_h]},
        "peak_dow":  {"day": peak_d,  "avg_kwh": by_dow[peak_d]},
        "by_hour_of_day": by_hour, "by_day_of_week": by_dow,
        "insights": [
            f"Peak usage hour: {peak_h}:00-{peak_h+1}:00",
            f"Peak day of week: {peak_d}",
            f"Highest day: {peak_day['date']} ({peak_day['kwh']} kWh)",
        ],
    }


def _enrich_app(a, total_cycle_kwh, tariff, cycle):
    daily_kwh  = (a["watts"] * a["quantity"] * a["hours_per_day"]) / 1000
    cycle_days = BILLING_CYCLE_DAYS_MAP.get(cycle, 60)
    cycle_kwh  = daily_kwh * cycle_days
    pct        = round(cycle_kwh / total_cycle_kwh * 100, 1) if total_cycle_kwh > 0 else 0
    bill_with  = calculate_bill(total_cycle_kwh, tariff, cycle)["total_bill"]
    bill_wo    = calculate_bill(max(0, total_cycle_kwh - cycle_kwh), tariff, cycle)["total_bill"]
    return {**a, "daily_kwh": round(daily_kwh,4), "cycle_kwh": round(cycle_kwh,2),
            "pct_of_total": pct, "bill_contribution": round(bill_with-bill_wo,2),
            "peak_load_kw": round(a["watts"]*a["quantity"]/1000, 3)}


@app.get("/appliances/{household_id}")
def get_appliances(household_id: str):
    with get_db() as conn:
        hh   = _get_hh_or_404(conn, household_id)
        rows = conn.execute(
            "SELECT * FROM appliances WHERE household_id=? ORDER BY watts*quantity DESC",
            (household_id,)).fetchall()
    tariff = hh.get("tariff_category","DOMESTIC")
    cycle  = hh.get("billing_cycle","Bi-monthly")
    apps   = [dict(r) for r in rows]
    cycle_days  = BILLING_CYCLE_DAYS_MAP.get(cycle, 60)
    total_cycle = sum((a["watts"]*a["quantity"]*a["hours_per_day"]/1000)*cycle_days
                      for a in apps if a["is_active"])
    enriched = [_enrich_app(a, total_cycle, tariff, cycle) for a in apps]
    return {
        "household_id": household_id, "tariff": tariff, "billing_cycle": cycle,
        "appliances": enriched,
        "summary": {
            "total_daily_kwh": round(sum(a["daily_kwh"] for a in enriched if a["is_active"]),3),
            "total_cycle_kwh": round(total_cycle,1),
            "estimated_bill":  calculate_bill(total_cycle, tariff, cycle),
            "active_count":    sum(1 for a in enriched if a["is_active"]),
            "top_consumer":    max(enriched, key=lambda x: x["cycle_kwh"], default=None),
        },
    }


@app.post("/appliances", status_code=201)
def add_appliance(a: ApplianceCreate):
    aid = a.id or uuid.uuid4().hex[:8].upper()
    with get_db() as conn:
        try:
            conn.execute(
                "INSERT INTO appliances (id,household_id,name,icon,watts,hours_per_day,quantity,is_active) VALUES (?,?,?,?,?,?,?,?)",
                (aid, a.household_id, a.name, a.icon or "💡", a.watts, a.hours_per_day,
                 a.quantity or 1, 1 if a.is_active else 0))
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(400, "ID exists")
    return {"status":"created","id":aid}


@app.put("/appliances/{appliance_id}")
def update_appliance(appliance_id: str, u: ApplianceUpdate):
    fields = {k:v for k,v in u.dict().items() if v is not None}
    if not fields:
        raise HTTPException(400, "Nothing to update")
    if "is_active" in fields:
        fields["is_active"] = 1 if fields["is_active"] else 0
    set_clause = ", ".join(f"{k}=?" for k in fields)
    with get_db() as conn:
        conn.execute(f"UPDATE appliances SET {set_clause} WHERE id=?", (*fields.values(), appliance_id))
        conn.commit()
    return {"status":"updated","id":appliance_id}


@app.delete("/appliances/{appliance_id}")
def delete_appliance(appliance_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM appliances WHERE id=?", (appliance_id,))
        conn.commit()
    return {"status":"deleted"}


@app.get("/appliances/{household_id}/peak")
def appliance_peak(household_id: str):
    with get_db() as conn:
        hh      = _get_hh_or_404(conn, household_id)
        rows    = conn.execute("SELECT * FROM appliances WHERE household_id=? AND is_active=1",(household_id,)).fetchall()
        pk_row  = conn.execute("SELECT MAX(hourly_kwh) as mx FROM iot_readings WHERE household_id=?",(household_id,)).fetchone()
    tariff = hh.get("tariff_category","DOMESTIC")
    cycle  = hh.get("billing_cycle","Bi-monthly")
    apps   = [dict(r) for r in rows]
    cycle_days  = BILLING_CYCLE_DAYS_MAP.get(cycle,60)
    total_cycle = sum((a["watts"]*a["quantity"]*a["hours_per_day"]/1000)*cycle_days for a in apps)
    enriched    = sorted([_enrich_app(a,total_cycle,tariff,cycle) for a in apps],
                         key=lambda x: x["cycle_kwh"], reverse=True)
    actual_pk   = float(pk_row["mx"]) if pk_row and pk_row["mx"] else 0.0
    theoretical = sum(a["peak_load_kw"] for a in enriched)
    return {
        "household_id": household_id,
        "actual_peak_kwh": round(actual_pk,3),
        "theoretical_peak_kw": round(theoretical,3),
        "load_factor_pct": round(actual_pk/theoretical*100,1) if theoretical else 0,
        "appliance_peaks": enriched,
        "top_consumer": enriched[0] if enriched else None,
        "insight": (f"Top: {enriched[0]['name']} — {enriched[0]['cycle_kwh']:.0f} units = ₹{enriched[0]['bill_contribution']}" if enriched else "No data"),
    }


@app.get("/limits/{household_id}")
def get_limits(household_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM usage_limits WHERE household_id=?", (household_id,)).fetchone()
    return dict(row) if row else {"household_id": household_id, "message": "No limits set"}


@app.post("/limits/{household_id}")
def set_limits(household_id: str, limits: UsageLimitSet):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO usage_limits (household_id,daily_kwh_limit,cycle_kwh_limit,daily_bill_limit,cycle_bill_limit,alert_enabled,updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(household_id) DO UPDATE SET
              daily_kwh_limit=excluded.daily_kwh_limit, cycle_kwh_limit=excluded.cycle_kwh_limit,
              daily_bill_limit=excluded.daily_bill_limit, cycle_bill_limit=excluded.cycle_bill_limit,
              alert_enabled=excluded.alert_enabled, updated_at=excluded.updated_at
        """, (household_id, limits.daily_kwh_limit, limits.cycle_kwh_limit,
              limits.daily_bill_limit, limits.cycle_bill_limit,
              1 if limits.alert_enabled else 0, datetime.now().isoformat()))
        conn.commit()
    return {"status":"saved","household_id":household_id,**limits.dict()}


@app.delete("/limits/{household_id}")
def delete_limits(household_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM usage_limits WHERE household_id=?", (household_id,))
        conn.commit()
    return {"status":"deleted"}


@app.get("/limits/{household_id}/breaches")
def get_breaches(household_id: str, days: int = 30):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM limit_breaches WHERE household_id=? AND breach_date>=date('now',?) ORDER BY created_at DESC",
            (household_id, f"-{days} days")).fetchall()
    return {"household_id": household_id, "breaches": [dict(r) for r in rows]}


def _update_daily(household_id, for_date):
    with get_db() as conn:
        row = conn.execute("""
            SELECT SUM(hourly_kwh) as total, MAX(hourly_kwh) as peak,
                   AVG(voltage) as avg_v, COUNT(*) as cnt
            FROM iot_readings WHERE household_id=? AND date(timestamp)=?
        """, (household_id, str(for_date))).fetchone()
        if row and row["total"]:
            conn.execute("""
                INSERT OR REPLACE INTO daily_summary
                (household_id,date,daily_kwh,peak_hourly_kwh,avg_voltage,reading_count)
                VALUES (?,?,?,?,?,?)
            """, (household_id, str(for_date), round(float(row["total"]),4),
                  round(float(row["peak"]),4), row["avg_v"], row["cnt"]))
            conn.commit()


def _check_limits(household_id, for_date):
    with get_db() as conn:
        lim_row = conn.execute("SELECT * FROM usage_limits WHERE household_id=? AND alert_enabled=1",(household_id,)).fetchone()
        if not lim_row:
            return
        lim  = dict(lim_row)
        hh   = conn.execute("SELECT * FROM households WHERE id=?",(household_id,)).fetchone()
        tariff = dict(hh).get("tariff_category","DOMESTIC") if hh else "DOMESTIC"
        cycle  = dict(hh).get("billing_cycle","Bi-monthly") if hh else "Bi-monthly"
        today_kwh = float(conn.execute(
            "SELECT COALESCE(SUM(hourly_kwh),0) FROM iot_readings WHERE household_id=? AND date(timestamp)=?",
            (household_id, str(for_date))).fetchone()[0])

        def log(btype, lv, av):
            if not conn.execute("SELECT id FROM limit_breaches WHERE household_id=? AND breach_type=? AND breach_date=?",
                                (household_id, btype, str(for_date))).fetchone():
                conn.execute("INSERT INTO limit_breaches (household_id,breach_type,limit_value,actual_value,breach_date) VALUES (?,?,?,?,?)",
                             (household_id, btype, lv, av, str(for_date)))

        if lim.get("daily_kwh_limit") and today_kwh > lim["daily_kwh_limit"]:
            log("daily_kwh", lim["daily_kwh_limit"], today_kwh)
        if lim.get("daily_bill_limit"):
            daily_bill = calculate_bill(today_kwh * 30, tariff, cycle)["total_bill"] / 30
            if daily_bill > lim["daily_bill_limit"]:
                log("daily_bill", lim["daily_bill_limit"], daily_bill)
        conn.commit()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
