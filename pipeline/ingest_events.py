import pandas as pd
import duckdb
import logging
import requests
import os

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openagenda.com/v2/agendas"
API_KEY = os.getenv("OPENAGENDA_PUBLIC_KEY")
AGENDA_UID = 56500817

DB_PATH = Path("data/warehouse.duckdb")

def fetch_events(start_date: str, end_date: str) -> pd.DataFrame:
    logger.info("Fetching événements Paris %s -> %s", start_date, end_date)
    if not API_KEY:
        raise ValueError("OPENAGENDA_PUBLIC_KEY manquante dans le .env")
    all_events = []
    after = None
    page = 0
    total_expected = None
    while True:
        page += 1
        out_of_range_count = 0
        params = {
            "key": API_KEY,
            "size": 100,
            "timings[gte]": start_date,
            "timings[lte]": end_date,
            "includeLabels": 1,
            "location[city]": "Paris",
        }
        if after:
            params["after"] = after
        try:
            response = requests.get(
                f"{BASE_URL}/{AGENDA_UID}/events",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("Erreur API OpenAgenda : %s", e)
            raise
        if total_expected is None:
            total_expected = data.get("total")
        events = data.get("events", [])
        if not events:
            logger.info("Fin pagination atteinte (page %d)", page)
            break
        dates_in_batch = []
        for event in events:
            first = event.get("firstTiming", {})
            begin = first.get("begin", "")
            end_t = first.get("end", "")
            if not begin:
                continue
            date_str = begin[:10]
            dates_in_batch.append(date_str)
            if not (start_date <= date_str <= end_date):
                out_of_range_count += 1
                continue
            all_events.append({
                "event_id": event.get("uid"),
                "titre": event.get("title", {}).get("fr", ""),
                "debut": begin,
                "fin": end_t,
            })
        current_total = len(all_events)
        if total_expected:
            progress = (current_total / total_expected) * 100
            logger.info(
                "[%s → %s] [Page %d] +%d valid | %d ignorés | Total: %d / %d (%.1f%%)",
                start_date,
                end_date,
                page,
                len(events) - out_of_range_count,
                out_of_range_count,
                current_total,
                total_expected,
                progress
            )
        else:
            logger.info(
                "[%s → %s] [Page %d] +%d valid | %d ignorés | Total: %d",
                start_date,
                end_date,
                page,
                len(events) - out_of_range_count,
                out_of_range_count,
                current_total
            )
        if out_of_range_count > 0:
            logger.warning(
                "⚠️  %d events hors plage ignorés sur cette page",
                out_of_range_count
            )
        if all_events:
            last_date = all_events[-1]["debut"][:10]
        else:
            last_date = "N/A"
        logger.info(
            "Fetched %d événements cumulés au %s",
            len(all_events),
            last_date
        )
        after = data.get("after")
        if not after:
            logger.info("Plus de curseur 'after' -> fin")
            break
    if not all_events:
        logger.warning("Aucun événement récupéré sur la période")
        return pd.DataFrame(columns=[
            "event_id", "titre", "debut", "fin",
            "date", "heure_debut", "heure_fin"
        ])
    df = pd.DataFrame(all_events)
    df["debut"] = pd.to_datetime(df["debut"], format='ISO8601', utc=True).dt.tz_convert("Europe/Paris")
    df["fin"] = pd.to_datetime(df["fin"], format='ISO8601', utc=True).dt.tz_convert("Europe/Paris")
    df["date"] = df["debut"].dt.date
    df["heure_debut"] = df["debut"].dt.hour
    df["heure_fin"] = df["fin"].dt.hour
    df = df.drop(columns=["debut", "fin"])
    df = df[["event_id", "titre", "date", "heure_debut", "heure_fin"]]
    logger.info("✅ %d occurrences d'événements récupérées", len(df))
    return df

def load_to_duckdb(df : pd.DataFrame, db_path: Path) -> None:
    logger.info("Loading events dans DuckDB -> %s", db_path)
    if df.empty:
        logger.warning("DataFrame vide, rien à charger")
        return
    con = duckdb.connect(str(db_path))
    con.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id BIGINT,
                    titre VARCHAR,
                    date DATE,
                    heure_debut INTEGER,
                    heure_fin INTEGER
                    )
                """)
    con.execute("DELETE FROM events")
    con.register("df_view", df)
    con.execute("INSERT INTO events SELECT * FROM df_view")
    count = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    con.close()
    logger.info("✅ Events chargés : %d lignes au total", count)

def run(start_date: str, end_date: str, db_path: Path) -> pd.DataFrame:
    df = fetch_events(start_date, end_date)
    load_to_duckdb(df, db_path)
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = run(start_date="2015-01-01", end_date="2025-12-31", db_path=DB_PATH)
    print(df.head())
    print(f"\n{len(df)} occurrences | {df['event_id'].nunique()} événements uniques")