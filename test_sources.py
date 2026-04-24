import requests
from dotenv import load_dotenv
import os
import duckdb

con = duckdb.connect("data/warehouse.duckdb", read_only=True)

# Stats globales
print(con.execute("DESCRIBE dataset_enrichi").df())

con.close()