import pandas as pd
import duckdb
import logging
import requests

from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

LAT = 48.8566
LON = 2.3522

DB_PATH = Path("data/warehouse.duckdb")

def fetch_weather(start_date, end_date) -> pd.DataFrame:
    logger.info(f"Fetching meteo horaire de {start_date} to {end_date}")
    
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    
    all_data = []
    current = start
    
    while current <= end:
        year_end = min(datetime(current.year, 12, 31), end)
        
        logger.info(f"Fetching {current.date()} -> {year_end.date()}")
        
        params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": current.strftime("%Y-%m-%d"),
            "end_date": year_end.strftime("%Y-%m-%d"),
            "hourly": [
                "temperature_2m",
                "precipitation",
                "windspeed_10m",
                "weathercode",
            ],
            "timezone": "Europe/Paris"
        }
        
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data["hourly"])
            all_data.append(df)
        
        except Exception as e:
            logger.error(f"Erreur fetching année {current.year}: {e}")
            raise
        
        current = year_end + timedelta(days=1)
    
    result = pd.concat(all_data, ignore_index=True)
    
    result.rename(columns={
        "time": "datetime",
        "temperature_2m": "temp",
        "precipitation": "precip_mm",
        "windspeed_10m": "wind_kmh",
        "weathercode": "weather_code",
    }, inplace=True)
    
    result["datetime"] = pd.to_datetime(result["datetime"])
    result["date"] = result["datetime"].dt.date
    result["heure"] = result["datetime"].dt.hour
    result = result.drop(columns=["datetime"])
    
    result = result[["date", "heure", "temp", "precip_mm", "wind_kmh", "weather_code"]]
    
    logger.info(f"Fetched %d lignes (%d jours x 24h)", len(result), len(result) // 24)
    return result

def load_to_duckdb(df, db_path) -> None:
    logger.info(f"Loading weather dans DuckDB -> {db_path}")
    
    con = duckdb.connect(str(db_path))
    con.execute("""
                CREATE TABLE IF NOT EXISTS weather (
                    date DATE,
                    heure INTEGER,
                    temp DOUBLE,
                    precip_mm DOUBLE,
                    wind_kmh DOUBLE,
                    weather_code INTEGER,
                    PRIMARY KEY (date, heure)
                )
                """)
    con.register("df_view", df)
    
    con.execute("""
                INSERT OR REPLACE INTO weather
                SELECT * FROM df_view
                """)
    
    count = con.execute("SELECT COUNT(*) FROM weather").fetchone()[0]
    con.close()
    
    logger.info("✅ Weather chargée : %d lignes au total", count)

def run(start_date, end_date, db_path) -> pd.DataFrame:
    df = fetch_weather(start_date,end_date)
    load_to_duckdb(df, db_path)
    return df

if __name__== "__main__":   
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    df = run(
        start_date="2015-01-01",
        end_date="2025-12-31",
        db_path=DB_PATH
    )
    
    print(df.head(24))
    
    