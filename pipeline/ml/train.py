import duckdb
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import logging
import shap
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH    = Path("data/warehouse.duckdb")
MODEL_PATH = Path("ml/models/lgbm_metrosignal.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "heure_sin", "heure_cos",
    "jour_sin", "jour_cos", "semaine_cos", "semaine_sin", "variance_historique",
    "mois_sin", "mois_cos",
    "is_weekend", "is_jour_ferie", "is_vacances_scolaires",
    "is_greve", "is_covid",
    "temp", "precip_mm", "wind_kmh", "weather_code",
    "nb_events",
    "rang_station",
    "lag_1h", "lag_24h", "lag_7j",
]
TARGET = "taux_congestion"

def load_data() -> pd.DataFrame:
    logger.info("Chargement dataset_enrichi depuis DuckDB ...")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT
            station, date, heure, jour_semaine,
            taux_congestion,
            heure_sin, heure_cos,
            jour_sin, jour_cos, semaine_sin, semaine_cos, variance_historique,
            mois_sin, mois_cos,
            is_weekend, is_jour_ferie, is_vacances_scolaires,
            is_greve, is_covid,
            temp, precip_mm, wind_kmh, weather_code,
            nb_events, rang_station
        FROM dataset_enrichi
        WHERE taux_congestion IS NOT NULL
        ORDER BY station, date, heure
    """).df()
    con.close()
    logger.info("Dataset chargé : %d lignes, %d stations", len(df), df["station"].nunique())
    return df

def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calcul des features lag ...")
    df = df.sort_values(["station", "date", "heure"]).reset_index(drop=True)

    df["lag_1h"]  = df.groupby("station")["taux_congestion"].shift(1)
    df["lag_24h"] = df.groupby("station")["taux_congestion"].shift(24)
    df["lag_7j"]  = df.groupby("station")["taux_congestion"].shift(24 * 7)

    n_avant = len(df)
    df = df.dropna(subset=["lag_1h", "lag_24h", "lag_7j"])
    logger.info("Lags calculés — lignes supprimées (NaN lag) : %d", n_avant - len(df))
    return df

def split_train_test(df: pd.DataFrame):
    cutoff = pd.Timestamp("2024-01-01")
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    logger.info("Train : %d lignes (%d–2023) | Test : %d lignes (2024–2025)",
                len(train), df["annee"].min() if "annee" in df.columns else 2015, len(test))
    return train, test

def train_model(train: pd.DataFrame):
    logger.info("Entraînement LightGBM ...")
    X_train = train[FEATURES]
    y_train = train[TARGET]
    
    params = {
        "objective":    "regression",
        "metric":       "mae",
        "learning_rate": 0.05,
        "num_leaves":   127,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":  5,
        "verbose":      -1,
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dtrain], 
        callbacks=[lgb.log_evaluation(10)],
    )
    return model

def evaluate(model, test: pd.DataFrame):
    logger.info("Évaluation sur le jeu de test ...")
    X_test = test[FEATURES]
    y_test = test[TARGET]
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    logger.info("MAE globale : %.4f | RMSE globale : %.4f", mae, rmse)
    
    #Evaluation par station (top 20 volume)
    
    test = test.copy()
    test["pred"] = preds
    station_metrics = (
        test.groupby("station")
        .apply(lambda g: pd.Series({
            "mae":  mean_absolute_error(g[TARGET], g["pred"]),
            "rmse": root_mean_squared_error(g[TARGET], g["pred"]),
            "n":    len(g),
        }))
        .sort_values("mae")
    )
    logger.info("Top 10 stations (meilleur MAE) :\n%s", station_metrics.head(10).to_string())
    logger.info("Top 10 stations (pire MAE) :\n%s",    station_metrics.tail(10).to_string())
    return station_metrics

def compute_shap(model, test: pd.DataFrame, station: str = "CHATELET"):
    logger.info("Calcul des SHAP values pour %s ...", station)
    
    sample = (
        test[
            (test["station"] == station) &
            (test["jour_semaine"] == 4) &
            (test["heure"] == 17)
        ]
    )
    
    if len(sample) == 0:
        logger.warning("Aucune donnée pour %s vendredi 17h", station)
        return
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample[FEATURES])
    
    shap_df = pd.DataFrame(
        np.abs(shap_values),
        columns=FEATURES
    ).mean().sort_values(ascending=False)
    
    logger.info("Importance SHAP - %s vendredi 17h : \n%s", station, shap_df.to_string())
    
    shap_path = Path("ml/models/shap_values.csv")
    pd.DataFrame(shap_values, columns=FEATURES).to_csv(shap_path, encoding="utf-8-sig")
    logger.info("✅ SHAP values sauvegardées : %s", shap_path)
    
    return shap_values, sample

def run_evaluate(test=None):
    logger.info("Chargement du modèle ...")
    model = joblib.load(MODEL_PATH)
    
    if test is None:
        df = load_data()
        df = add_lags(df)
        _, test = split_train_test(df)
    
    station_metrics = evaluate(model, test)
    metrics_path = Path("ml/models/station_metrics.csv")
    station_metrics.to_csv(metrics_path, encoding="utf-8-sig")
    logger.info("✅ Métriques par station : %s", metrics_path)
    
    compute_shap(model, test)

def run():
    df = load_data()
    df = add_lags(df)
    train, test = split_train_test(df)
    model = train_model(train)
    
    joblib.dump(model, MODEL_PATH)
    logger.info("✅ Modèle sauvegardé : %s", MODEL_PATH)
    
    del train, model
    
    run_evaluate(test)