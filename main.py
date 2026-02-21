"""
EB Bill Prediction - FastAPI Backend
=====================================
Endpoints:
  POST /iot/push          - IoT device pushes reading directly
  POST /iot/webhook       - IFTTT/Webhook integration
  GET  /dashboard/{id}    - Full dashboard data
  GET  /predict/{id}      - Predict remaining bill for cycle
  GET  /forecast/{id}     - 7/30 day forecast
  GET  /history/{id}      - Historical readings
  POST /household         - Register new household
  GET  /health            - Health check
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timedelta, date
import sqlite3
import json
import pickle
import numpy as np
import pandas as pd
import os
from contextlib import contextmanager

# ─── TNEB CONSTANTS ──────────────────────────────────────────────────────────
TNEB_SLABS = [
    (0, 100, 0.0),
    (100, 200, 1.50),
    (200, 500, 3.00),
    (500, 1000, 5.00),
    (1000, float('inf'), 7.00),
]
FIXED_CHARGE = 30.0
METER_RENT = 10.0
ELECTRICITY_DUTY_RATE = 0.16
BILLING_CYCLE_DAYS = 60  # TNEB bimonthly


def calculate_tneb_bill(units: float) -> dict:
    energy_charge = 0.0
    remaining = max(0, units)
    slab_breakdown = []
    for lower, upper, rate in TNEB_SLABS:
        if remaining <= 0:
            break
        slab_units = min(remaining, upper - lower if upper != float('inf') else remaining)
        charge = slab_units * rate
        energy_charge += charge
        slab_breakdown.append({
            "slab": f"{lower}-{int(upper) if upper != float('inf') else '∞'}",
            "units": round(slab_units, 2),
            "rate": rate,
            "charge": round(charge, 2)
        })
        remaining -= slab_units
    subtotal = energy_charge + FIXED_CHARGE + METER_RENT
    duty = round(subtotal * ELECTRICITY_DUTY_RATE, 2)
    return {
        "units": round(units, 2),
        "energy_charge": round(energy_charge, 2),
        "fixed_charge": FIXED_CHARGE,
        "meter_rent": METER_RENT,
        "electricity_duty": duty,
        "total_bill": round(subtotal + duty, 2),
        "slab_breakdown": slab_breakdown
    }


# ─── DATABASE ─────────────────────────────────────────────────────────────────
DB_PATH = "eb_data.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS households (
            id TEXT PRIMARY KEY,
            name TEXT,
            consumer_number TEXT,
            tariff_category TEXT DEFAULT 'LT1A',
            billing_cycle_start DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS iot_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            household_id TEXT,
            timestamp TIMESTAMP,
            cumulative_kwh REAL,
            hourly_kwh REAL,
            voltage REAL,
            current_amps REAL,
            power_factor REAL,
            source TEXT DEFAULT 'iot',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (household_id) REFERENCES households(id)
        );

        CREATE TABLE IF NOT EXISTS daily_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            household_id TEXT,
            date DATE UNIQUE,
            daily_kwh REAL,
            peak_kwh REAL,
            off_peak_kwh REAL,
            avg_voltage REAL,
            temperature_c REAL,
            reading_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_readings_hh_ts ON iot_readings(household_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_daily_hh_date ON daily_summary(household_id, date);
        """)
    print("✅ Database initialized")


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ─── ML MODEL ─────────────────────────────────────────────────────────────────
MODEL_PATH = "C:/Users/ELCOT/Downloads/files/eb-prediction/model_artifacts/xgb_model.pkl"
METADATA_PATH = "C:/Users/ELCOT/Downloads/files/eb-prediction/model_artifacts/metadata.json"

model = None
feature_cols = None


def load_model():
    global model, feature_cols
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        feature_cols = meta["feature_cols"]
        print(f"✅ Model loaded (R²={meta['val_r2']})")
    else:
        print("⚠️  Model not found — using rule-based fallback")


def predict_daily_kwh(features: dict) -> float:
    """Predict daily kWh from features dict."""
    if model is None:
        # Fallback: seasonal average
        monthly_avg = {
            1: 8.0, 2: 8.5, 3: 10.5, 4: 13.0, 5: 14.5, 6: 11.0,
            7: 9.5, 8: 9.0, 9: 9.5, 10: 10.0, 11: 8.5, 12: 7.5
        }
        return monthly_avg.get(features.get("month", 6), 10.0)

    X = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])
    return float(model.predict(X)[0])


# ─── PYDANTIC MODELS ──────────────────────────────────────────────────────────
class IoTReading(BaseModel):
    household_id: str
    timestamp: Optional[datetime] = None
    cumulative_kwh: float
    voltage: Optional[float] = None
    current_amps: Optional[float] = None
    power_factor: Optional[float] = 1.0
    source: Optional[str] = "iot"


class IFTTTWebhook(BaseModel):
    """IFTTT Webhooks format"""
    value1: str   # household_id
    value2: str   # cumulative_kwh
    value3: Optional[str] = None  # timestamp or voltage


class HouseholdCreate(BaseModel):
    id: str
    name: str
    consumer_number: Optional[str] = ""
    tariff_category: Optional[str] = "LT1A"
    billing_cycle_start: Optional[date] = None


# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EB Bill Prediction API",
    description="TNEB EB Bill forecasting with IoT integration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    init_db()
    load_model()
    # Seed a demo household
    with get_db() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO households (id, name, consumer_number, billing_cycle_start)
            VALUES ('HH001', 'Demo Home', 'TN1234567', '2025-01-01')
        """)
        conn.commit()


# ─── HEALTH ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


# ─── IOT ENDPOINTS ────────────────────────────────────────────────────────────
@app.post("/iot/push")
async def iot_push(reading: IoTReading, background_tasks: BackgroundTasks):
    """
    IoT device pushes reading directly.
    Device should push every 15–60 minutes with current cumulative meter value.
    """
    ts = reading.timestamp or datetime.now()

    with get_db() as conn:
        # Get last reading to compute hourly_kwh
        last = conn.execute("""
            SELECT cumulative_kwh, timestamp FROM iot_readings
            WHERE household_id = ? ORDER BY timestamp DESC LIMIT 1
        """, (reading.household_id,)).fetchone()

        if last:
            hours_diff = max(0.01, (ts - datetime.fromisoformat(last['timestamp'])).total_seconds() / 3600)
            hourly_kwh = max(0, (reading.cumulative_kwh - last['cumulative_kwh']) / hours_diff)
            # Normalize to per-hour
            hourly_kwh = round(hourly_kwh, 4)
        else:
            hourly_kwh = 0.0

        conn.execute("""
            INSERT INTO iot_readings
            (household_id, timestamp, cumulative_kwh, hourly_kwh, voltage, current_amps, power_factor, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            reading.household_id, ts.isoformat(), reading.cumulative_kwh,
            hourly_kwh, reading.voltage, reading.current_amps,
            reading.power_factor, reading.source
        ))
        conn.commit()

    background_tasks.add_task(update_daily_summary, reading.household_id, ts.date())

    return {"status": "received", "hourly_kwh": hourly_kwh, "timestamp": ts.isoformat()}


@app.post("/iot/webhook")
async def ifttt_webhook(data: IFTTTWebhook, background_tasks: BackgroundTasks):
    """
    IFTTT Webhooks integration.
    Set up IFTTT applet: trigger → Webhooks → POST to this endpoint
    value1 = household_id
    value2 = cumulative_kwh reading
    value3 = optional voltage
    """
    reading = IoTReading(
        household_id=data.value1,
        cumulative_kwh=float(data.value2),
        voltage=float(data.value3) if data.value3 else None,
        source="ifttt"
    )
    return await iot_push(reading, background_tasks)


@app.post("/iot/bulk")
async def iot_bulk_push(readings: List[IoTReading]):
    """Push multiple readings at once (for catchup after offline period)."""
    results = []
    for reading in readings:
        ts = reading.timestamp or datetime.now()
        with get_db() as conn:
            last = conn.execute("""
                SELECT cumulative_kwh FROM iot_readings
                WHERE household_id = ? ORDER BY timestamp DESC LIMIT 1
            """, (reading.household_id,)).fetchone()
            hourly_kwh = max(0, reading.cumulative_kwh - (last['cumulative_kwh'] if last else 0))
            conn.execute("""
                INSERT INTO iot_readings
                (household_id, timestamp, cumulative_kwh, hourly_kwh, voltage, current_amps, power_factor, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                reading.household_id, ts.isoformat(), reading.cumulative_kwh,
                round(hourly_kwh, 4), reading.voltage, reading.current_amps,
                reading.power_factor or 1.0, reading.source or "bulk"
            ))
            conn.commit()
        results.append({"household_id": reading.household_id, "status": "ok"})
    return results


# ─── HOUSEHOLD ────────────────────────────────────────────────────────────────
@app.post("/household")
def create_household(hh: HouseholdCreate):
    with get_db() as conn:
        try:
            conn.execute("""
                INSERT INTO households (id, name, consumer_number, tariff_category, billing_cycle_start)
                VALUES (?, ?, ?, ?, ?)
            """, (hh.id, hh.name, hh.consumer_number, hh.tariff_category,
                  str(hh.billing_cycle_start or date.today())))
            conn.commit()
            return {"status": "created", "id": hh.id}
        except sqlite3.IntegrityError:
            raise HTTPException(400, "Household ID already exists")


@app.get("/household/{household_id}")
def get_household(household_id: str):
    with get_db() as conn:
        hh = conn.execute("SELECT * FROM households WHERE id = ?", (household_id,)).fetchone()
        if not hh:
            raise HTTPException(404, "Household not found")
        return dict(hh)


# ─── DASHBOARD ────────────────────────────────────────────────────────────────
@app.get("/dashboard/{household_id}")
def dashboard(household_id: str):
    """Full dashboard: current usage, bill estimate, trend, forecast."""
    with get_db() as conn:
        # Current cycle info
        today = date.today()

        # Last 60 days readings
        readings = conn.execute("""
            SELECT date(timestamp) as day, SUM(hourly_kwh) as daily_kwh
            FROM iot_readings
            WHERE household_id = ? AND timestamp >= date('now', '-60 days')
            GROUP BY date(timestamp)
            ORDER BY day
        """, (household_id,)).fetchall()

        # Latest reading
        latest = conn.execute("""
            SELECT * FROM iot_readings WHERE household_id = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (household_id,)).fetchone()

        # Today's usage
        today_usage = conn.execute("""
            SELECT COALESCE(SUM(hourly_kwh), 0) as kwh
            FROM iot_readings
            WHERE household_id = ? AND date(timestamp) = date('now')
        """, (household_id,)).fetchone()

        # Current month usage
        month_usage = conn.execute("""
            SELECT COALESCE(SUM(hourly_kwh), 0) as kwh
            FROM iot_readings
            WHERE household_id = ? AND strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now')
        """, (household_id,)).fetchone()

    days_data = [dict(r) for r in readings]
    total_kwh_so_far = sum(r['daily_kwh'] for r in days_data)
    days_elapsed = len(days_data)
    days_remaining = max(0, BILLING_CYCLE_DAYS - days_elapsed)

    # Projected total for cycle
    daily_avg = total_kwh_so_far / max(1, days_elapsed)
    projected_total = total_kwh_so_far + (daily_avg * days_remaining)

    # Bill estimates
    current_bill = calculate_tneb_bill(total_kwh_so_far)
    projected_bill = calculate_tneb_bill(projected_total)

    # Last 7 days trend
    last_7 = days_data[-7:] if len(days_data) >= 7 else days_data
    avg_last_7 = sum(r['daily_kwh'] for r in last_7) / max(1, len(last_7))

    return {
        "household_id": household_id,
        "today_kwh": round(float(today_usage['kwh']), 3),
        "month_kwh": round(float(month_usage['kwh']), 3),
        "cycle_days_elapsed": days_elapsed,
        "cycle_days_remaining": days_remaining,
        "cycle_kwh_so_far": round(total_kwh_so_far, 2),
        "projected_cycle_kwh": round(projected_total, 2),
        "current_bill_estimate": current_bill,
        "projected_bill": projected_bill,
        "daily_avg_kwh": round(daily_avg, 3),
        "last_7day_avg_kwh": round(avg_last_7, 3),
        "latest_reading": dict(latest) if latest else None,
        "daily_trend": days_data[-30:],  # Last 30 days for chart
        "alert": _generate_alert(daily_avg, projected_total),
    }


def _generate_alert(daily_avg: float, projected_total: float) -> dict:
    if projected_total > 1000:
        return {"level": "danger", "message": f"⚠️ Projected {projected_total:.0f} units — high bill expected! Reduce AC usage."}
    elif projected_total > 500:
        return {"level": "warning", "message": f"📊 Projected {projected_total:.0f} units — moderate usage, on track."}
    elif projected_total <= 100:
        return {"level": "success", "message": "✅ Under 100 units — FREE tier! Great savings."}
    else:
        return {"level": "info", "message": f"📈 Projected {projected_total:.0f} units this cycle."}


# ─── PREDICTION ───────────────────────────────────────────────────────────────
@app.get("/predict/{household_id}")
def predict_bill(household_id: str, days_ahead: int = 30):
    """
    Predict bill for the current/next billing cycle.
    Uses ML model for day-by-day forecast then applies TNEB slabs.
    """
    with get_db() as conn:
        # Get recent history for lag features
        history = conn.execute("""
            SELECT date(timestamp) as day, SUM(hourly_kwh) as daily_kwh,
                   AVG(voltage) as avg_voltage
            FROM iot_readings
            WHERE household_id = ?
            GROUP BY date(timestamp)
            ORDER BY day DESC
            LIMIT 35
        """, (household_id,)).fetchall()

    history_df = pd.DataFrame([dict(r) for r in history]).sort_values('day')
    recent_kwh = list(history_df['daily_kwh'].values) if len(history_df) > 0 else [10.0]

    # Build forecast day by day
    forecast = []
    today = date.today()
    cumulative = 0.0

    for i in range(1, days_ahead + 1):
        target_date = today + timedelta(days=i)
        month = target_date.month
        dow = target_date.weekday()

        # Temperature by month (seasonal estimate)
        monthly_temp = {1:26,2:27,3:30,4:33,5:35,6:34,7:33,8:33,9:32,10:29,11:27,12:25}

        features = {
            "month": month,
            "day_of_week": dow,
            "is_weekend": 1 if dow >= 5 else 0,
            "day_of_year": target_date.timetuple().tm_yday,
            "week_of_year": int(target_date.strftime("%W")),
            "quarter": (month - 1) // 3 + 1,
            "is_summer": 1 if month in [3, 4, 5] else 0,
            "is_monsoon": 1 if month in [6, 7, 8, 9, 10] else 0,
            "is_winter": 1 if month in [11, 12, 1, 2] else 0,
            "avg_temperature": monthly_temp.get(month, 30),
            "lag_1d": recent_kwh[-1] if recent_kwh else 10.0,
            "lag_7d": recent_kwh[-7] if len(recent_kwh) >= 7 else recent_kwh[0],
            "lag_30d": recent_kwh[-30] if len(recent_kwh) >= 30 else recent_kwh[0],
            "rolling_7": np.mean(recent_kwh[-7:]),
            "rolling_14": np.mean(recent_kwh[-14:]),
            "rolling_30": np.mean(recent_kwh[-30:]),
            "std_7": np.std(recent_kwh[-7:]) if len(recent_kwh) >= 2 else 0.5,
            "cycle_day": i,
            "cycle_cumulative_kwh": cumulative,
            "projected_cycle_kwh": cumulative + np.mean(recent_kwh[-7:]) * (days_ahead - i + 1),
        }

        predicted_kwh = predict_daily_kwh(features)
        cumulative += predicted_kwh
        recent_kwh.append(predicted_kwh)

        forecast.append({
            "date": str(target_date),
            "predicted_kwh": round(predicted_kwh, 3),
            "cumulative_kwh": round(cumulative, 3),
            "bill_if_stopped_today": calculate_tneb_bill(cumulative)["total_bill"]
        })

    final_bill = calculate_tneb_bill(cumulative)

    return {
        "household_id": household_id,
        "forecast_days": days_ahead,
        "predicted_total_kwh": round(cumulative, 2),
        "predicted_bill": final_bill,
        "daily_forecast": forecast,
        "model_used": "xgboost" if model else "rule_based_fallback"
    }


# ─── HISTORY ──────────────────────────────────────────────────────────────────
@app.get("/history/{household_id}")
def history(household_id: str, days: int = 30):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT date(timestamp) as day,
                   SUM(hourly_kwh) as daily_kwh,
                   MAX(cumulative_kwh) as end_cumulative,
                   AVG(voltage) as avg_voltage,
                   COUNT(*) as readings
            FROM iot_readings
            WHERE household_id = ? AND timestamp >= date('now', ?)
            GROUP BY date(timestamp)
            ORDER BY day
        """, (household_id, f'-{days} days')).fetchall()

    data = [dict(r) for r in rows]

    # Annotate with bill
    running = 0.0
    for row in data:
        running += row['daily_kwh'] or 0
        row['running_bill'] = calculate_tneb_bill(running)["total_bill"]
        row['running_kwh'] = round(running, 2)

    return {"household_id": household_id, "days": days, "history": data}


# ─── BACKGROUND: Update daily summary ─────────────────────────────────────────
def update_daily_summary(household_id: str, for_date: date):
    with get_db() as conn:
        row = conn.execute("""
            SELECT SUM(hourly_kwh) as total, MAX(hourly_kwh) as peak,
                   AVG(voltage) as avg_v, COUNT(*) as cnt
            FROM iot_readings
            WHERE household_id = ? AND date(timestamp) = ?
        """, (household_id, str(for_date))).fetchone()

        if row and row['total']:
            conn.execute("""
                INSERT OR REPLACE INTO daily_summary
                (household_id, date, daily_kwh, peak_kwh, avg_voltage, reading_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (household_id, str(for_date),
                  round(row['total'], 4), round(row['peak'], 4),
                  row['avg_v'], row['cnt']))
            conn.commit()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
