import pickle
import math
import duckdb
import pandas as pd
import holidays
from datetime import datetime, timedelta
from pathlib import Path

VACANCES_ZONE_C = [
        ("2015-02-14", "2015-03-02"), ("2015-04-18", "2015-05-04"), ("2015-07-04", "2015-09-01"),
        ("2015-10-17", "2015-11-02"), ("2015-12-19", "2016-01-04"),
        ("2016-02-20", "2016-03-07"), ("2016-04-16", "2016-05-02"), ("2016-07-05", "2016-09-01"),
        ("2016-10-20", "2016-11-03"), ("2016-12-17", "2017-01-03"),
        ("2017-02-04", "2017-02-20"), ("2017-04-01", "2017-04-18"), ("2017-07-08", "2017-09-04"),
        ("2017-10-21", "2017-11-06"), ("2017-12-23", "2018-01-08"),
        ("2018-02-17", "2018-03-05"), ("2018-04-14", "2018-04-30"), ("2018-07-07", "2018-09-03"),
        ("2018-10-20", "2018-11-05"), ("2018-12-22", "2019-01-07"),
        ("2019-02-23", "2019-03-11"), ("2019-04-20", "2019-05-06"), ("2019-07-06", "2019-09-02"),
        ("2019-10-19", "2019-11-04"), ("2019-12-21", "2020-01-06"),
        ("2020-02-08", "2020-02-24"), ("2020-04-04", "2020-04-20"), ("2020-07-04", "2020-09-01"),
        ("2020-10-17", "2020-11-02"), ("2020-12-19", "2021-01-04"),
        ("2021-02-13", "2021-03-01"), ("2021-04-10", "2021-04-26"), ("2021-07-06", "2021-09-02"),
        ("2021-10-23", "2021-11-08"), ("2021-12-18", "2022-01-03"),
        ("2022-02-19", "2022-03-07"), ("2022-04-23", "2022-05-09"), ("2022-07-07", "2022-09-01"),
        ("2022-10-22", "2022-11-07"), ("2022-12-17", "2023-01-03"),
        ("2023-02-18", "2023-03-06"), ("2023-04-22", "2023-05-09"), ("2023-07-08", "2023-09-04"),
        ("2023-10-21", "2023-11-06"), ("2023-12-23", "2024-01-08"),
        ("2024-02-10", "2024-02-26"), ("2024-04-06", "2024-04-22"), ("2024-07-06", "2024-09-02"),
        ("2024-10-19", "2024-11-04"), ("2024-12-21", "2025-01-06"),
        ("2025-02-22", "2025-03-10"), ("2025-04-19", "2025-05-05"), ("2025-07-05", "2025-09-01"),
        ("2025-10-18", "2025-11-03"), ("2025-12-20", "2026-01-05"),
        ("2026-02-21", "2026-03-09"), ("2026-04-18", "2026-05-04"), ("2026-07-04", "2026-08-31"),
    ]

FEATURE_COLS = [
    "heure_sin", "heure_cos",
    "jour_sin", "jour_cos", "semaine_cos", "semaine_sin",
    "mois_sin", "mois_cos",
    "is_weekend", "is_jour_ferie", "is_vacances_scolaires",
    "is_greve", "is_covid",
    "temp", "precip_mm", "wind_kmh", "weather_code",
    "nb_events",
    "rang_station",
    "lag_1h", "lag_24h", "lag_7j",
]

class Predictor:
    def __init__(self, model_path: str, db_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.con = duckdb.connect(db_path, read_only=True)
        self.fr_holidays = holidays.France()
        self._load_station_meta()
        
    def _load_station_meta(self):
        """Charge rang de toutes les stations en mémoire."""
        df = self.con.execute("""
            SELECT DISTINCT station, rang_station,
            FROM dataset_enrichi
        """).df()
        self.station_meta = df.set_index("station").to_dict("index")
        self.known_stations = sorted(self.station_meta.keys())
        
    def _normalize_station(self, name: str) -> str:
        """Trouve le nom canonique le plus proche (upper + strip)."""
        upper = name.strip().upper()
        if upper in self.station_meta:
            return upper
        matches = [s for s in self.known_stations if upper in s]
        if matches:
            return matches[0]
        raise ValueError(f"Station inconnue : {name}")

    def _get_lag_features(self, station: str, dt: datetime) -> dict:
        """Récupère les 3 lags depuis DuckDB."""
        def fetch(target_dt):
            row = self.con.execute("""
                SELECT taux_congestion FROM dataset_enrichi
                WHERE station = ? AND date = ? AND heure = ?
                LIMIT 1
            """, [station, target_dt.date(), target_dt.hour]).fetchone()
            return row[0] if row else 0.0
        return {
            "lag_1h":  fetch(dt - timedelta(hours=1)),
            "lag_24h": fetch(dt - timedelta(hours=24)),
            "lag_7j":  fetch(dt - timedelta(days=7)),
        }
        
    def _is_vacances(self, dt: datetime) -> int:
        d = dt.date().isoformat()
        return int(any(s <= d <= e for s, e in VACANCES_ZONE_C))

    def build_features(self, station: str, dt: datetime, is_greve: int = 0) -> dict:
        """Construit le dictionnaire de features pour une station + datetime."""
        h = dt.hour
        dow = dt.weekday()
        week = dt.isocalendar().week
        month = dt.month
        
        features = {
            "heure_sin":  math.sin(2 * math.pi * h / 24),
            "heure_cos":  math.cos(2 * math.pi * h / 24),
            "jour_sin":   math.sin(2 * math.pi * dow / 7),
            "jour_cos":   math.cos(2 * math.pi * dow / 7),
            "mois_sin":   math.sin(2 * math.pi * month / 12),
            "mois_cos":   math.cos(2 * math.pi * month / 12),
            "semaine_sin":math.sin(2 * math.pi * week / 52),
            "semaine_cos":math.cos(2 * math.pi * week / 52),
            "is_weekend":           int(dow >= 5),
            "is_jour_ferie":        int(dt.date() in self.fr_holidays),
            "is_vacances_scolaires":self._is_vacances(dt),
            "is_greve" : is_greve,
            "is_covid": 0,
            "temp": 15.0, "precip_mm": 0.0, "wind_kmh": 10.0, "weather_code": 0,
            "nb_events": 0,
            "rang_station":        self.station_meta[station]["rang_station"],   
        }
        features.update(self._get_lag_features(station, dt))
        return features

    def predict(self, station: str, dt: datetime, is_greve: int = 0) -> dict:
        canonical = self._normalize_station(station)
        features = self.build_features(canonical, dt, is_greve)
        df = pd.DataFrame([features])[FEATURE_COLS]
        z_score = float(self.model.predict(df)[0])
        z_clipped = max(-5.0, min(5.0, z_score))
        if z_clipped >= 2.0: label = "très chargé"
        elif z_clipped >= 0.5: label = "chargé"
        elif z_clipped <= -1.5: label = "très faible"
        elif z_clipped <= -0.5: label = "faible"
        else: label = "normal"
        return {
            "station": canonical,
            "datetime": dt.isoformat(),
            "taux_congestion": round(z_clipped, 3),
            "label": label,
            "features_used": features,
        }