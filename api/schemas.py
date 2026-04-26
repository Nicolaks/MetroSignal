from pydantic import BaseModel, Field
from typing import Optional

class PredictionResponse(BaseModel):
    station: str
    datetime: str
    taux_congestion: float
    label: str
    features_used: dict
    
class StationInfo(BaseModel):
    name: str
    rang: Optional[int]
    
class HistoryPoint(BaseModel):
    date: str
    heure: int
    nb_vald_heure: float
    taux_congestion: float
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool
    stations_count: int
    
class HistoryResponse(BaseModel):
    station: str
    points: list[HistoryPoint]
    count: int