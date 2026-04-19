import duckdb
import logging
import holidays
import pandas as pd

from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("data/warehouse.duckdb")

ZONES_VACANCES = {
    "C": [
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
    ]
}

def build_vacances_set(zones: dict) -> set:
    vacances = set()
    for _, periodes in zones.items():
        for debut, fin in periodes:
            d = pd.Timestamp(debut)
            f = pd.Timestamp(fin)
            while d <= f:
                vacances.add(d.date())
                d += pd.Timedelta(days=1)
    return vacances

def build_jours_feries_set(years: list) -> set:
    feries = set()
    for year in years:
        for date in holidays.France(years=year).keys():
            feries.add(date)
    return feries

def compute_taux_congestion(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calcul du taux de congestion ...")
    
    # Attente d'occupation normale sur la station à telle heure tel jours
    baseline = (
        df.groupby(["station", "jour_semaine", "heure"])["nb_vald_heure"]
        .mean()
        .rename("baseline_mean")
        .reset_index()
    )
    # Pareil mais avec l'écart-type pour mesurer la variablité du créneau dans le temps    
    std = (
        df.groupby(["station", "jour_semaine", "heure"])["nb_vald_heure"]
        .std()
        .rename("baseline_std")
        .reset_index()
    )
    
    df = df.merge(baseline, on=["station", "jour_semaine", "heure"], how="left")
    df = df.merge(std, on=["station", "jour_semaine", "heure"], how="left")
    
    # Taux de congestion : écart à la moyenne normalisé par l'écart-type
    # Si std = 0 (station toujours au même niveau), taux = 0
    # Si taux = +2 : station chargée
    # Si taux = -2 : anormalement vide
    df["taux_congestion"] = (
        (df["nb_vald_heure"] - df["baseline_mean"])
        / df["baseline_std"].replace(0,1)
    ).round(4)
    
    logger.info("Taux de congestion calculé")
    return df

def detect_greve(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Détection des jours de grève ...")
    
    total_stations = (
        df.groupby("date")["station"]
        .nunique()
        .rename("total_stations")
        .reset_index()
    )
    
    stations_affectees = (
        df[df["taux_congestion"] < -1.55]
        .groupby("date")["station"]
        .nunique()
        .rename("stations_affectees")
        .reset_index()
    )
    
    greve = total_stations.merge(stations_affectees, on="date", how="left")
    greve["stations_affectees"] = greve["stations_affectees"].fillna(0)
    greve["pct_affectees"] = greve["stations_affectees"] / greve["total_stations"]
    greve["is_greve"] = (greve["pct_affectees"] >= 0.50).astype(int)
    
    df = df.merge(greve[["date", "is_greve"]], on="date", how="left")
    
    nb_jours_greve = greve["is_greve"].sum()
    logger.info("✅ %d jours de grève détectés", nb_jours_greve)
    return df

def compute_station_stats(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calcul variance historique et rang station ...")
    
    station_stats = (
        df.groupby("station")["nb_vald_heure"]
        .agg(
            variance_historique="var", # A quel point le trafic de cette station fluctue. Une station avec forte variance est imprévisible
            volume_moyen="mean", # Le trafic moyen toutes heures confondues
        )
        .reset_index()
    )
    
    station_stats["rang_station"] = (
        station_stats["volume_moyen"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    
    df = df.merge(station_stats[["station", "variance_historique", "rang_station"]], on="station", how="left")
    logger.info("Stats station calculées")
    return df

def join_weather(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    logger.info("Jointure avec weather ...")
    
    weather = con.execute("SELECT * FROM weather").df()
    weather["date"] = pd.to_datetime(weather["date"]).dt.date
    
    df = df.merge(weather, on=["date", "heure"], how="left")
    logger.info("Jointure weather OK : %d lignes", len(df))
    return df

def join_events(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    logger.info("Jointure avec events ...")
    
    events = con.execute("SELECT date, heure_debut, COUNT(*) as nb_events FROM events GROUP BY date, heure_debut").df()
    events["date"] = pd.to_datetime(events["date"]).dt.date
    events = events.rename(columns={"heure_debut": "heure"})
    
    df = df.merge(events, on=["date", "heure"], how="left")
    df["nb_events"] = df["nb_events"].fillna(0).astype(int)
    
    logger.info("Jointure events OK : %d lignes", len(df))
    return df

def add_calendar_features(df: pd.DataFrame, vacances_set: set, feries_set: set) -> pd.DataFrame:
    logger.info("Ajout features calendrier ...")
    
    df["is_jour_ferie"] = df["date"].apply(lambda d: int(d in feries_set))
    df["is_vacances_scolaires"] = df["date"].apply(lambda d: int(d in vacances_set))
    
    logger.info("Features calendrier ajoutées")
    return df

def run(db_path: Path = DB_PATH) -> pd.DataFrame:
    logger.info("=== Démarrage transform.py ===")
    
    con = duckdb.connect(str(db_path))
    
    logger.info("Chargement de la table validations ...")
    df = con.execute("SELECT * FROM validations").df()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    years = list(range(2015, 2026))
    vacances_set = build_vacances_set(ZONES_VACANCES)
    feries_set = build_jours_feries_set(years)
    
    df = compute_taux_congestion(df)
    df = detect_greve(df)
    df = compute_station_stats(df)
    df = join_weather(df, con)
    df = join_events(df, con)
    df = add_calendar_features(df, vacances_set, feries_set)
    
    logger.info("Sauvergarde dans DuckDB -> table dataset_enrichi ...")
    con.execute("DROP TABLE IF EXISTS dataset_enrichi")
    con.execute("CREATE TABLE dataset_enrichi AS SELECT * FROM df")
    
    count = con.execute("SELECT COUNT(*) FROM dataset_enrichi").fetchone()[0]
    cols = con.execute("DESCRIBE dataset_enrichi").df()["column_name"].tolist()
    con.close()
    
    logger.info("✅ dataset_enrichi : %d lignes | colonnes : %s", count, cols)
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")
    df = run()
    print(df.head())
    print(f"\nColonnes : {list(df.columns)}")