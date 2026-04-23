import requests
from dotenv import load_dotenv
import os
import duckdb

con = duckdb.connect("data/warehouse.duckdb", read_only=True)

# Stats globales
print(con.execute("""
    SELECT station, SUM(nb_vald_heure) as total
    FROM dataset_enrichi
    WHERE station LIKE '%DEFENSE%'
    GROUP BY station
    ORDER BY total DESC
""").df())

con.close()