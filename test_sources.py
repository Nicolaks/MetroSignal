import requests
from dotenv import load_dotenv
import os

load_dotenv()


url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 48.8566,
    "longitude": 2.3522,
    "start_date": "2024-01-01",
    "end_date": "2024-01-07",
    "hourly": "temperature_2m,precipitation",
    "timezone": "Europe/Paris",    
}

resp = requests.get(url, params=params, timeout=30)
print("Open-Meteo status: ", resp.status_code)
print(resp.json()["hourly"]["temperature_2m"][:5])



PUBLIC_KEY = os.getenv("OPENAGENDA_PUBLIC_KEY")
print("Clé chargée : ", PUBLIC_KEY[:4] + "...")

url = "https://api.openagenda.com/v2/agendas"
params = {
    "key": PUBLIC_KEY,
    "size": 5,
    "search": "Paris"
    }
resp = requests.get(url, params=params, timeout=30)

agenda = resp.json().get("agendas", [])

for a in agenda:
    print(a["uid"], "—", a["title"])

print("OpenAgenda status:", resp.status_code)
print(resp.json())