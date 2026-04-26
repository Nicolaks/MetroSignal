from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from contextlib import asynccontextmanager
from api.predictor import Predictor
from api.schemas import PredictionResponse, HistoryResponse, HealthResponse, StationInfo

MODEL_PATH = "ml\models\lgbm_metrosignal.pkl"
DB_PATH = "data/warehouse.duckdb"

predictor: Predictor = None

print("Predictor importé :", Predictor)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code exécuté UNE SEULE FOIS au démarrage du serveur.
    On charge le modèle ici — pas à chaque requête.
    """
    global predictor
    predictor = Predictor(MODEL_PATH, DB_PATH)
    print(f"✅ Modèle chargé — {len(predictor.known_stations)} stations disponibles")
    yield
    predictor.con.close()

app = FastAPI(
    title="MetroSignal API",
    description="Prédiction de congestion du métro parisien",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
)

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
def health_check():
    """Vérifie que l'API est en ligne et le modèle chargé."""
    try:
        predictor.con.execute("SELECT 1")
        db_ok = True
    except:
        db_ok = False

    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "db_connected": db_ok,
        "stations_count": len(predictor.known_stations),
    }


@app.get("/predict", response_model=PredictionResponse)
def predict(
    station:  str = Query(..., example="Chatelet"),
    datetimeQ: str = Query(..., example="2025-06-13T17:00"),
    is_greve: int = 0,
):
    """
    Prédit le taux de congestion pour une station à une date/heure donnée.

    - **station** : nom de la station (partiel accepté, insensible à la casse)
    - **datetime** : format ISO 8601 `YYYY-MM-DDTHH:MM`
    """
    try:
        dt = datetime.fromisoformat(datetimeQ)
    except ValueError:
        raise HTTPException(422, "Format datetime invalide. Attendu : YYYY-MM-DDTHH:MM")

    try:
        result = predictor.predict(station, dt, is_greve)
    except ValueError as e:
        raise HTTPException(404, str(e))

    return result


@app.get("/stations", response_model=list[StationInfo])
def list_stations(limit: int = Query(50, ge=1, le=500)):
    """Retourne la liste des stations disponibles, triées par volume."""
    stations = []
    for name, meta in sorted(
        predictor.station_meta.items(),
        key=lambda x: x[1].get("rang_station", 9999)
    )[:limit]:
        stations.append({
            "name": name,
            "rang": meta.get("rang_station"),
            "variance": round(meta.get("variance_historique", 0), 3),
        })
    return stations


@app.get("/history/{station}", response_model=HistoryResponse)
def get_history(
    station:    str,
    start_date: str = Query("2024-01-01", example="2024-01-01"),
    end_date:   str = Query("2024-01-07", example="2024-01-07"),
):
    """Retourne l'historique réel d'une station sur une période."""
    try:
        canonical = predictor._normalize_station(station)
    except ValueError as e:
        raise HTTPException(404, str(e))

    rows = predictor.con.execute("""
        SELECT date, heure, nb_vald_heure, taux_congestion
        FROM dataset_enrichi
        WHERE station = ?
          AND date BETWEEN ? AND ?
        ORDER BY date, heure
        LIMIT 2000
    """, [canonical, start_date, end_date]).df()

    points = rows.to_dict("records")
    return {"station": canonical, "points": points, "count": len(points)}