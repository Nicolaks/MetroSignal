import requests
from dotenv import load_dotenv
import os
import duckdb

con = duckdb.connect("data/warehouse.duckdb", read_only=True)

# Doublons
n_dup = con.execute("""
    SELECT COUNT(*) FROM (
        SELECT station, date, heure, COUNT(*) n
        FROM dataset_enrichi GROUP BY 1,2,3 HAVING n > 1
    )
""").fetchone()[0]
print(f"Doublons dataset_enrichi : {n_dup}")  # → 0

# Stats globales
print(con.execute("""
    SELECT
        COUNT(*) AS nb_lignes,
        COUNT(DISTINCT station) AS nb_stations,
        MIN(taux_congestion) AS taux_min,
        MAX(taux_congestion) AS taux_max,
        AVG(taux_congestion) AS taux_moy
    FROM dataset_enrichi
""").df().to_string())

con.close()