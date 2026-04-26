import requests
from dotenv import load_dotenv
import os
import duckdb

con = duckdb.connect("data/warehouse.duckdb", read_only=True)

print(con.execute("""
            SELECT 
                CASE 
                    WHEN variance_historique < 0.2 THEN '0.0 - 0.2 (Très stable)'
                    WHEN variance_historique < 0.4 THEN '0.2 - 0.4 (Stable)'
                    WHEN variance_historique < 0.6 THEN '0.4 - 0.6 (Modéré)'
                    WHEN variance_historique < 0.8 THEN '0.6 - 0.8 (Variable)'
                    ELSE '0.8+ (Instable)'
                END AS segment_stabilite,
                COUNT(*) AS nb_lignes,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) || '%' AS proportion
            FROM dataset_enrichi
            GROUP BY 1
            ORDER BY 1
""").df())

con.close()