import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH     = Path("data/warehouse.duckdb")
OUTPUT_DIR  = Path("../img/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE       = "plotly_dark"
COLOR_PRIMARY  = "#00D4FF"
COLOR_ACCENT   = "#FF6B6B"
COLOR_WARNING  = "#FFD43B"
COLOR_BG       = "#0F1117"
COLORSCALE_DIV = "RdYlGn"
COLORSCALE_SEQ = "Blues"

LAYOUT_BASE = dict(
    template=TEMPLATE,
    font_family="monospace",
    paper_bgcolor=COLOR_BG,
    plot_bgcolor="rgba(255,255,255,0.03)",
    margin=dict(l=60, r=40, t=60, b=60),
)

figures = []

def save(fig: go.Figure, name: str, height: int = 600) -> None:
    fig.update_layout(height=height, title_text=f"{fig.layout.title.text}")
    figures.append(fig)


# ── Connexion ─────────────────────────────────────────────────────────────────
con = duckdb.connect(str(DB_PATH), read_only=True)
logger.info("Connecté à %s", DB_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# 1. APERÇU GÉNÉRAL
# ══════════════════════════════════════════════════════════════════════════════
logger.info("=== 1. Aperçu général ===")

for table in ["validations", "weather", "events", "dataset_enrichi"]:
    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    logger.info("  %-25s : %15d lignes", table, n)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DISTRIBUTIONS DE BASE
# ══════════════════════════════════════════════════════════════════════════════
logger.info("=== 2. Distributions de base ===")

# ── 2.1 Top 20 stations ───────────────────────────────────────────────────────
top_stations = con.execute("""
    SELECT station, SUM(nb_vald_heure) AS total_validations
    FROM dataset_enrichi
    GROUP BY station
    ORDER BY total_validations DESC
    LIMIT 20
""").df()

fig = px.bar(
    top_stations.sort_values("total_validations"),
    x="total_validations", y="station", orientation="h",
    title="🏆 Top 20 Stations — Volume Total de Validations",
    labels={"total_validations": "Validations totales", "station": ""},
    color="total_validations", color_continuous_scale=COLORSCALE_SEQ,
)
fig.update_layout(**LAYOUT_BASE, coloraxis_showscale=False)
fig.update_traces(marker_line_width=0)
save(fig, "2_1_top20_stations")

# ── 2.2 Heures de pointe ─────────────────────────────────────────────────────
hourly = con.execute("""
    SELECT heure, is_weekend, AVG(nb_vald_heure) AS moy
    FROM dataset_enrichi
    GROUP BY heure, is_weekend
    ORDER BY heure
""").df()
hourly["type_jour"] = hourly["is_weekend"].map({0: "Jour ouvré", 1: "Weekend"})

fig = px.line(
    hourly, x="heure", y="moy", color="type_jour",
    title="⏰ Validations Moyennes par Heure — Semaine vs Weekend",
    labels={"heure": "Heure", "moy": "Validations (moyenne)"},
    color_discrete_map={"Jour ouvré": COLOR_PRIMARY, "Weekend": COLOR_WARNING},
    markers=True,
)
fig.update_layout(**LAYOUT_BASE)
fig.update_xaxes(tickmode="linear", dtick=1)
save(fig, "2_2_heures_pointe")

# ── 2.3 Heatmap station × heure ──────────────────────────────────────────────
top30 = top_stations.head(30)["station"].tolist()
placeholder = ", ".join(["?"] * len(top30))
heatmap_data = con.execute(f"""
    SELECT station, heure, AVG(taux_congestion) AS taux_cong_moyen
    FROM dataset_enrichi
    WHERE station IN ({placeholder})
    GROUP BY station, heure
    ORDER BY station, heure
""", top30).df()

heatmap_pivot = heatmap_data.pivot(index="station", columns="heure", values="taux_cong_moyen")

fig = go.Figure(data=go.Heatmap(
    z=heatmap_pivot.values,
    x=[f"{h}h" for h in heatmap_pivot.columns],
    y=heatmap_pivot.index.tolist(),
    colorscale=COLORSCALE_DIV, zmid=0,
    colorbar=dict(title="Taux congestion<br>(z-score)", tickfont_color="white"),
))
fig.update_layout(
    **LAYOUT_BASE,
    title="🔥 Heatmap Taux de Congestion — Top 30 Stations × Heure",
    xaxis_title="Heure", yaxis_title="",
)
save(fig, "2_3_heatmap_station_heure", height=700)

# ── 2.4 Distribution taux_congestion par jour de la semaine ──────────────────
dist_jour = con.execute("""
    SELECT jour_semaine,
           PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY taux_congestion) AS q1,
           PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY taux_congestion) AS median,
           PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY taux_congestion) AS q3,
           AVG(taux_congestion) AS mean,
           MIN(taux_congestion) AS min_val,
           MAX(taux_congestion) AS max_val
    FROM dataset_enrichi
    WHERE taux_congestion BETWEEN -5 AND 5
    GROUP BY jour_semaine
    ORDER BY jour_semaine
""").df()

jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
dist_jour["jour_label"] = dist_jour["jour_semaine"].map(dict(enumerate(jours)))

fig = go.Figure()
for _, row in dist_jour.iterrows():
    fig.add_trace(go.Box(
        name=row["jour_label"],
        q1=[row["q1"]], median=[row["median"]], q3=[row["q3"]],
        mean=[row["mean"]], lowerfence=[row["min_val"]], upperfence=[row["max_val"]],
        boxmean=True,
    ))
fig.update_layout(
    **LAYOUT_BASE,
    title="📊 Distribution du Taux de Congestion par Jour de la Semaine",
    yaxis_title="Taux congestion (z-score)",
    showlegend=False,
)
save(fig, "2_4_distribution_congestion")


# ══════════════════════════════════════════════════════════════════════════════
# 3. OUTLIERS & ÉVÉNEMENTS HISTORIQUES
# ══════════════════════════════════════════════════════════════════════════════
logger.info("=== 3. Outliers & événements historiques ===")

# ── 3.1 Série temporelle globale ─────────────────────────────────────────────
daily_global = con.execute("""
    SELECT
        date,
        SUM(nb_vald_heure)          AS total_vald,
        MEDIAN(taux_congestion)     AS taux_cong_median,
        MAX(is_greve)               AS is_greve,
        MAX(is_vacances_scolaires)  AS is_vacances,
        MAX(is_jour_ferie)          AS is_ferie
    FROM dataset_enrichi
    GROUP BY date
    ORDER BY date
""").df()
daily_global["date"] = pd.to_datetime(daily_global["date"])
daily_global["rolling7"] = daily_global["total_vald"].rolling(7, center=True).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily_global["date"], y=daily_global["total_vald"],
    mode="lines", name="Validations journalières",
    line=dict(color="rgba(0,212,255,0.3)", width=1),
))
fig.add_trace(go.Scatter(
    x=daily_global["date"], y=daily_global["rolling7"],
    mode="lines", name="Moyenne mobile 7j",
    line=dict(color=COLOR_PRIMARY, width=2),
))
greve_days = daily_global[daily_global["is_greve"] == 1]
fig.add_trace(go.Scatter(
    x=greve_days["date"], y=greve_days["total_vald"],
    mode="markers", name="Grève détectée",
    marker=dict(color=COLOR_ACCENT, size=6, symbol="x"),
))
fig.add_vrect(x0="2020-03-17", x1="2020-05-11",
    fillcolor="rgba(255,107,107,0.15)", line_width=0,
    annotation_text="COVID Confinement 1", annotation_position="top left",
    annotation_font_color=COLOR_ACCENT)
fig.add_vrect(x0="2024-07-26", x1="2024-08-11",
    fillcolor="rgba(255,212,59,0.12)", line_width=0,
    annotation_text="JO Paris 2024", annotation_position="top left",
    annotation_font_color=COLOR_WARNING)
fig.update_layout(
    **LAYOUT_BASE,
    title="📈 Série Temporelle Globale — Validations Journalières 2015-2025",
    xaxis_title="Date", yaxis_title="Total validations", hovermode="x unified",
)
save(fig, "3_1_serie_temporelle", height=500)

# ── 3.2 Grèves par année ─────────────────────────────────────────────────────
greve_par_an = con.execute("""
    SELECT annee, COUNT(DISTINCT date) AS nb_jours_greve
    FROM dataset_enrichi
    WHERE is_greve = 1
    GROUP BY annee
    ORDER BY annee
""").df()

fig = px.bar(
    greve_par_an, x="annee", y="nb_jours_greve",
    title="🚨 Nombre de Jours de Grève Détectés par Année",
    labels={"annee": "Année", "nb_jours_greve": "Jours de grève"},
    color="nb_jours_greve", color_continuous_scale="Reds",
)
fig.update_layout(**LAYOUT_BASE, coloraxis_showscale=False)
save(fig, "3_2_greves_par_an")

# ── 3.3 Indice fréquentation mensuel base 100 = 2019 ─────────────────────────
monthly = con.execute("""
    SELECT annee, mois, SUM(nb_vald_heure) AS total
    FROM dataset_enrichi
    GROUP BY annee, mois
    ORDER BY annee, mois
""").df()

baseline_2019 = monthly[monthly["annee"] == 2019].set_index("mois")["total"]
monthly["indice"] = monthly.apply(
    lambda r: r["total"] / baseline_2019.get(r["mois"], float("nan")) * 100,
    axis=1
)

fig = go.Figure()
fig.add_hline(y=100, line_dash="dash", line_color="gray",
              annotation_text="Niveau 2019 (base 100)")
colors = px.colors.qualitative.Set2
for i, annee in enumerate(sorted(monthly["annee"].unique())):
    sub = monthly[monthly["annee"] == annee].sort_values("mois")
    lw = 3 if annee in [2020, 2021] else 1.5
    fig.add_trace(go.Scatter(
        x=sub["mois"], y=sub["indice"],
        mode="lines+markers", name=str(annee),
        line=dict(color=colors[i % len(colors)], width=lw),
    ))
fig.update_layout(
    **LAYOUT_BASE,
    title="🦠 Indice de Fréquentation Mensuel (base 100 = 2019)",
    xaxis=dict(title="Mois", tickmode="linear", dtick=1,
               ticktext=["Jan","Fév","Mar","Avr","Mai","Jun",
                         "Jul","Aoû","Sep","Oct","Nov","Déc"],
               tickvals=list(range(1, 13))),
    yaxis_title="Indice",
)
save(fig, "3_3_indice_frequentation", height=500)

# ── 3.4 JO 2024 vs été 2023 ──────────────────────────────────────────────────
jo_compare = con.execute("""
    WITH jo AS (
        SELECT station, AVG(taux_congestion) AS jo_2024
        FROM dataset_enrichi
        WHERE date BETWEEN '2024-07-26' AND '2024-08-11'
        GROUP BY station
    ),
    ref AS (
        SELECT station, AVG(taux_congestion) AS ref_2023
        FROM dataset_enrichi
        WHERE date BETWEEN '2023-07-26' AND '2023-08-11'
        GROUP BY station
    )
    SELECT jo.station, jo_2024, ref_2023, (jo_2024 - ref_2023) AS delta
    FROM jo JOIN ref ON jo.station = ref.station
    ORDER BY delta DESC
    LIMIT 20
""").df()

fig = px.bar(
    jo_compare, x="delta", y="station", orientation="h",
    title="🏅 JO Paris 2024 — Stations avec la Plus Forte Hausse vs Été 2023",
    labels={"delta": "Δ Taux Congestion", "station": ""},
    color="delta", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
)
fig.update_layout(**LAYOUT_BASE, coloraxis_showscale=False)
save(fig, "3_4_jo_2024", height=600)


# ══════════════════════════════════════════════════════════════════════════════
# 4. INSIGHT CLÉ — MÉTÉO VS HEURE
# ══════════════════════════════════════════════════════════════════════════════
logger.info("=== 4. Insight clé — Météo vs Heure ===")

# Corrélation Spearman par station via DuckDB
# On calcule corr(precip_mm, taux_congestion) et corr(heure, taux_congestion)
station_corr = con.execute("""
    SELECT
        station,
        CORR(precip_mm, taux_congestion)  AS r_precip,
        CORR(temp, taux_congestion)       AS r_temp,
        CORR(wind_kmh, taux_congestion)   AS r_wind,
        CORR(heure, taux_congestion)      AS r_heure,
        COUNT(*)                          AS n_obs
    FROM dataset_enrichi
    WHERE precip_mm IS NOT NULL AND taux_congestion IS NOT NULL
    GROUP BY station
    HAVING COUNT(*) >= 100
    ORDER BY station
""").df()

station_corr["r_precip"]    = station_corr["r_precip"].abs()
station_corr["r_temp"]      = station_corr["r_temp"].abs()
station_corr["r_wind"]      = station_corr["r_wind"].abs()
station_corr["r_heure"]     = station_corr["r_heure"].abs()
station_corr["r_meteo_max"] = station_corr[["r_precip", "r_temp", "r_wind"]].max(axis=1)
station_corr["meteo_vs_heure"] = station_corr["r_meteo_max"] / station_corr["r_heure"].replace(0, 0.001)

top_meteo = station_corr[station_corr["meteo_vs_heure"] >= 1.0]
logger.info("Stations météo > heure : %d / %d", len(top_meteo), len(station_corr))

station_corr["color_label"] = station_corr["meteo_vs_heure"].apply(
    lambda x: "🌦️ Météo > Heure" if x >= 1.0 else "⏰ Heure > Météo"
)

fig = px.scatter(
    station_corr,
    x="r_heure", y="r_meteo_max",
    color="color_label", size="n_obs",
    hover_name="station",
    title="🔍 Insight Clé : Météo vs Heure — Corrélation avec le Taux de Congestion",
    labels={
        "r_heure": "Corrélation |Pearson| Heure → Congestion",
        "r_meteo_max": "Corrélation |Pearson| Météo → Congestion",
        "color_label": "",
    },
    color_discrete_map={
        "🌦️ Météo > Heure": COLOR_ACCENT,
        "⏰ Heure > Météo": COLOR_PRIMARY,
    },
)
max_val = max(station_corr["r_heure"].max(), station_corr["r_meteo_max"].max()) + 0.05
fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
              line=dict(dash="dot", color="gray", width=1))
fig.update_layout(**LAYOUT_BASE)
save(fig, "4_1_meteo_vs_heure", height=550)

# ── 4.2 Impact pluie par intensité ───────────────────────────────────────────
precip_bins = con.execute("""
    SELECT
        CASE
            WHEN precip_mm = 0        THEN '0_Sec'
            WHEN precip_mm < 0.1      THEN '1_Trace'
            WHEN precip_mm < 1        THEN '2_Légère'
            WHEN precip_mm < 3        THEN '3_Modérée'
            WHEN precip_mm < 7        THEN '4_Forte'
            ELSE                           '5_Très forte'
        END AS intensite,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY taux_congestion) AS q1,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY taux_congestion) AS median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY taux_congestion) AS q3,
        AVG(taux_congestion) AS mean,
        MIN(taux_congestion) AS min_val,
        MAX(taux_congestion) AS max_val
    FROM dataset_enrichi
    WHERE taux_congestion BETWEEN -5 AND 5
    GROUP BY intensite
    ORDER BY intensite
""").df()

fig = go.Figure()
for _, row in precip_bins.iterrows():
    fig.add_trace(go.Box(
        name=row["intensite"].split("_", 1)[1],
        q1=[row["q1"]], median=[row["median"]], q3=[row["q3"]],
        mean=[row["mean"]], lowerfence=[row["min_val"]], upperfence=[row["max_val"]],
        boxmean=True,
    ))
fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Normal (z=0)")
fig.update_layout(
    **LAYOUT_BASE,
    title="🌧️ Intensité des Précipitations vs Taux de Congestion",
    xaxis_title="Intensité pluie", yaxis_title="Taux congestion (z-score)",
    showlegend=False,
)
save(fig, "4_2_pluie_intensite")

# ── 4.3 Delta congestion pluie vs sec par heure ───────────────────────────────
delta_heure = con.execute("""
    SELECT
        heure,
        AVG(CASE WHEN precip_mm > 0 THEN taux_congestion END) AS avec_pluie,
        AVG(CASE WHEN precip_mm = 0 THEN taux_congestion END) AS sans_pluie
    FROM dataset_enrichi
    GROUP BY heure
    ORDER BY heure
""").df()
delta_heure["delta"] = delta_heure["avec_pluie"] - delta_heure["sans_pluie"]

fig = go.Figure(go.Bar(
    x=delta_heure["heure"], y=delta_heure["delta"],
    marker_color=[COLOR_ACCENT if v > 0 else COLOR_PRIMARY for v in delta_heure["delta"]],
))
fig.add_hline(y=0, line_color="gray")
fig.update_layout(
    **LAYOUT_BASE,
    title="☔ Impact de la Pluie par Heure — Δ Taux Congestion (Pluie − Sec)",
    xaxis=dict(title="Heure", tickmode="linear", dtick=1),
    yaxis_title="Δ z-score congestion",
)
save(fig, "4_3_delta_pluie_heure")


# ══════════════════════════════════════════════════════════════════════════════
# 5. STATIONS IMPRÉVISIBLES
# ══════════════════════════════════════════════════════════════════════════════
logger.info("=== 5. Stations imprévisibles ===")

# ── 5.1 Top 30 variance ───────────────────────────────────────────────────────
station_stats = con.execute("""
    SELECT
        station,
        ANY_VALUE(variance_historique) AS variance,
        ANY_VALUE(rang_station)        AS rang,
        AVG(nb_vald_heure)             AS vol_moyen,
        AVG(is_greve)                  AS taux_greve
    FROM dataset_enrichi
    GROUP BY station
""").df().dropna(subset=["variance"])

top_imprev = station_stats.nlargest(30, "variance")

fig = px.bar(
    top_imprev.sort_values("variance"),
    x="variance", y="station", orientation="h",
    title="🎲 Top 30 Stations les Plus Imprévisibles (Variance du z-score)",
    labels={"variance": "Variance historique (z-score)", "station": ""},
    color="variance", color_continuous_scale="Reds",
)
fig.update_layout(**LAYOUT_BASE, coloraxis_showscale=False)
save(fig, "5_1_stations_imprev", height=700)

# ── 5.2 Volume vs Variance ────────────────────────────────────────────────────
fig = px.scatter(
    station_stats,
    x="vol_moyen", y="variance",
    hover_name="station", color="taux_greve", size="vol_moyen",
    title="📍 Volume vs Variance — Stations Stratégiques pour le ML",
    labels={
        "vol_moyen": "Volume moyen (validations/heure)",
        "variance": "Variance historique (z-score)",
        "taux_greve": "Taux jours grève",
    },
    color_continuous_scale="Reds",
)
fig.add_vline(x=station_stats["vol_moyen"].median(), line_dash="dot", line_color="gray")
fig.add_hline(y=station_stats["variance"].median(), line_dash="dot", line_color="gray")
fig.update_layout(**LAYOUT_BASE)
save(fig, "5_2_volume_vs_variance", height=550)

# ── 5.3 Profil horaire station imprévisible vs stable ─────────────────────────
station_imprev = station_stats.nlargest(1, "variance")["station"].values[0]
station_stable = station_stats.nsmallest(1, "variance")["station"].values[0]
logger.info("Station imprévisible : %s | Stable : %s", station_imprev, station_stable)

profils = con.execute(f"""
    SELECT station, heure,
           AVG(nb_vald_heure) AS mean_vald,
           STDDEV(nb_vald_heure) AS std_vald
    FROM dataset_enrichi
    WHERE station IN ('{station_imprev}', '{station_stable}')
    GROUP BY station, heure
    ORDER BY station, heure
""").df()

fig = go.Figure()
FILLS = {
    COLOR_ACCENT: "rgba(255, 107, 107, 0.15)",
    COLOR_PRIMARY: "rgba(0, 212, 255, 0.15)",
}

for station, color in [(station_imprev, COLOR_ACCENT), (station_stable, COLOR_PRIMARY)]:
    sub = profils[profils["station"] == station]
    label = f"🎲 {station}" if station == station_imprev else f"📏 {station}"
    fig.add_trace(go.Scatter(
        x=sub["heure"], y=sub["mean_vald"],
        mode="lines+markers", name=label,
        line=dict(color=color, width=2),
    ))
    upper = sub["mean_vald"] + sub["std_vald"]
    lower = sub["mean_vald"] - sub["std_vald"]
    fig.add_trace(go.Scatter(
        x=list(sub["heure"]) + list(sub["heure"])[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself",
        fillcolor=FILLS[color],  # ← rgba valide
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
fig.update_layout(
    **LAYOUT_BASE,
    title="📊 Profil Horaire : Station Imprévisible vs Station Stable (±1σ)",
    xaxis=dict(title="Heure", tickmode="linear", dtick=1),
    yaxis_title="Validations/heure",
)
save(fig, "5_3_profil_horaire_imprev")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SYNTHÈSE — CORRÉLATIONS FEATURES VS CIBLE
# ══════════════════════════════════════════════════════════════════════════════
logger.info("=== 6. Synthèse ===")

feature_corr = con.execute("""
    SELECT
        CORR(heure,                 taux_congestion) AS heure,
        CORR(jour_semaine,          taux_congestion) AS jour_semaine,
        CORR(mois,                  taux_congestion) AS mois,
        CORR(is_weekend,            taux_congestion) AS is_weekend,
        CORR(temp,                  taux_congestion) AS temp,
        CORR(precip_mm,             taux_congestion) AS precip_mm,
        CORR(wind_kmh,              taux_congestion) AS wind_kmh,
        CORR(weather_code,          taux_congestion) AS weather_code,
        CORR(nb_events,             taux_congestion) AS nb_events,
        CORR(is_jour_ferie,         taux_congestion) AS is_jour_ferie,
        CORR(is_vacances_scolaires, taux_congestion) AS is_vacances_scolaires,
        CORR(is_greve,              taux_congestion) AS is_greve
    FROM dataset_enrichi
""").df().T.reset_index()

feature_corr.columns = ["feature", "correlation"]
feature_corr = feature_corr.sort_values("correlation", key=abs, ascending=False)

fig = px.bar(
    feature_corr,
    x="correlation", y="feature", orientation="h",
    title="🤖 Corrélation Pearson des Features avec taux_congestion",
    labels={"correlation": "Corrélation Pearson", "feature": "Feature"},
    color="correlation",
    color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
)
fig.update_layout(**LAYOUT_BASE, coloraxis_showscale=False)
save(fig, "6_1_feature_correlations")

logger.info("Corrélations avec taux_congestion :")
for _, row in feature_corr.iterrows():
    bar = "█" * int(abs(row["correlation"]) * 30)
    sign = "+" if row["correlation"] > 0 else "-"
    logger.info("  %-25s  %s%s %.3f", row["feature"], sign, bar, row["correlation"])

# ── Fin ───────────────────────────────────────────────────────────────────────
con.close()
logger.info("=== EDA terminée — figures dans %s ===", OUTPUT_DIR)

html_parts = [pio.to_html(f, full_html=False, include_plotlyjs="cdn") for f in figures]
html = f"""
<html>
<head><meta charset="utf-8"><title>MetroSignal EDA</title>
<style>body {{ background: {COLOR_BG}; }} </style>
</head>
<body>
{"".join(html_parts)}
</body>
</html>
"""
output_path = Path("outputs/eda/eda_report.html")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(html, encoding="utf-8")
logger.info("✅ Rapport EDA : %s", output_path)

import webbrowser
webbrowser.open(str(output_path.resolve()))
