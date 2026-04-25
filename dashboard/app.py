import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from plotly.subplots import make_subplots
import os
import requests

GEO_CSV_PATH = os.getenv("GEO_CSV_PATH", "data/raw/emplacement-des-gares-idf.csv")
METRICS_CSV_PATH = os.getenv("METRICS_CSV_PATH", "ml/models/station_metrics.csv")
DB_PATH = os.getenv("DB_PATH", "data/warehouse.duckdb")
API_URL = os.getenv("API_URL", "http://localhost:8000")

COLORS = {
    "bg":      "#0e1117",
    "surface": "#1a1f2e",
    "border":  "#2d3748",
    "accent":  "#00d4ff",
    "accent2": "#ff6b6b",
    "accent3": "#ffd93d",
    "text":    "#e2e8f0",
    "muted":   "#718096",
}

#Config page
st.set_page_config(
    page_title="MetroSignal",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
    )

st.markdown(f"""
        <style>
            .stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}
            [data-testid="stSidebar"] {{
                background-color: {COLORS['surface']};
                border-right: 1px solid {COLORS['border']};
            }}
            [data-testid="metric-container"] {{
                background: {COLORS['surface']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 16px;
            }}
            h1, h2, h3 {{ color: {COLORS['accent']} !important; }}
            hr {{ border-color: {COLORS['border']}; }}
            .stAlert {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; }}
        </style>
        """, unsafe_allow_html=True)

#Connexion

@st.cache_resource
def get_connection():
    return duckdb.connect(DB_PATH, read_only=True)

def query(sql: str, params: list = None) -> pd.DataFrame:
    con = get_connection()
    return con.execute(sql, params or []).df()

def apply_dark_theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["accent"], size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text"], family="monospace"),
        xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"],
                   tickfont=dict(color=COLORS["muted"])),
        yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"],
                   tickfont=dict(color=COLORS["muted"])),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"]),
        margin=dict(l=40, r=20, t=50, b=40),        
    )
    return fig

#Données de base 

@st.cache_data(ttl=3600)
def get_stations() -> list:
    df = query("SELECT station, SUM(nb_vald_heure) AS total FROM dataset_enrichi GROUP BY station ORDER BY total DESC")
    return df["station"].tolist()

@st.cache_data(ttl=3600)
def get_years() -> list:
    return query("SELECT DISTINCT annee FROM dataset_enrichi ORDER BY annee")["annee"].tolist()

#Sidebar

with st.sidebar:
    st.markdown("## 🚇 MetroSignal")
    st.markdown("*Analyse du trafic IDFM rail 2015–2025*")
    st.divider()
    
    view = st.radio("Vue", [
        "🗺️ Heatmap réseau",
        "🌧️ Météo & Trafic",
        "🎭 Événements & Pics",
        "📊 Stations imprévisibles",
        "🔮 Prédiction ML",
        "🌍 Carte géographique",
        "🚨 Grèves & Anomalies",
        "📈 Performance modèle",
    ])
    st.divider()
    
    stations_list = get_stations()
    years_list = get_years()
    
    selected_station = st.selectbox("Station", stations_list, index=0, help="Classées par volume total décroissant")
    selected_year    = st.selectbox("Année", years_list, index=len(years_list) - 1)
    
    st.divider()
    st.caption("Données : IDFM · Open-Meteo · OpenAgenda")
    st.caption(f"BDD : '{DB_PATH}'")

# ─────────────────────────────────────────────────────────────────────────────
# VUE 1 — HEATMAP RÉSEAU
# ─────────────────────────────────────────────────────────────────────────────

if view == "🗺️ Heatmap réseau":
    st.title("🗺️ Heatmap trafic — Réseau")
    st.markdown("Taux de congestion moyen (z-score) par **heure** et **jour de la semaine**.")

    @st.cache_data(ttl=3600)
    def get_network_metrics(year):
        return query("""
            SELECT COUNT(DISTINCT station) AS nb_stations,
                SUM(nb_vald_heure)      AS total_validations,
                ROUND(AVG(taux_congestion), 3) AS congestion_moyenne,
                -- On compte uniquement les dates uniques ayant subi une grève
                COUNT(DISTINCT CASE WHEN is_greve = 1 THEN date END) AS jours_greve
            FROM dataset_enrichi WHERE annee = ?
        """, [year])

    m = get_network_metrics(selected_year).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stations actives",        f"{int(m.nb_stations)}")
    c2.metric("Validations totales",     f"{int(m.total_validations):,}".replace(",", " "))
    c3.metric("Congestion moyenne",      f"{m.congestion_moyenne:.3f} σ")
    c4.metric("Jours de grève détectés", f"{int(m.jours_greve)}")
    st.divider()

    col_left, col_right = st.columns([3, 1])
    with col_right:
        top_n      = st.slider("Top N stations", 10, 50, 20, step=5)
        excl_greve = st.checkbox("Exclure jours de grève", value=True)
        excl_covid = st.checkbox("Exclure COVID (2020)", value=True)

    @st.cache_data(ttl=3600)
    def get_heatmap_data(year, top_n, excl_greve, excl_covid):
        where  = ["annee = ?", "taux_congestion IS NOT NULL"]
        params = [year]
        if excl_greve: where.append("is_greve = 0")
        if excl_covid: where.append("NOT (annee = 2020 AND mois BETWEEN 3 AND 5)")
        w      = " AND ".join(where)
        top_q  = f"SELECT station FROM (SELECT station, SUM(nb_vald_heure) v FROM dataset_enrichi WHERE {w} GROUP BY station ORDER BY v DESC LIMIT {top_n})"
        return query(f"""
            SELECT heure,
                   CASE jour_semaine WHEN 0 THEN 'Lun' WHEN 1 THEN 'Mar' WHEN 2 THEN 'Mer'
                       WHEN 3 THEN 'Jeu' WHEN 4 THEN 'Ven' WHEN 5 THEN 'Sam' WHEN 6 THEN 'Dim'
                   END AS jour_label,
                   jour_semaine,
                   ROUND(AVG(taux_congestion), 3) AS congestion_moy
            FROM dataset_enrichi
            WHERE {w} AND station IN ({top_q})
            GROUP BY heure, jour_semaine, jour_label
            ORDER BY jour_semaine, heure
        """, params * 2)

    df_heat = get_heatmap_data(selected_year, top_n, excl_greve, excl_covid)
    if not df_heat.empty:
        pivot = df_heat.pivot(index="jour_label", columns="heure", values="congestion_moy")
        pivot = pivot.reindex([j for j in ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"] if j in pivot.index])
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values, x=[f"{h}h" for h in pivot.columns], y=pivot.index,
            colorscale=[[0,"#1a237e"],[0.3,"#0d47a1"],[0.5,"#00b4d8"],
                        [0.65,"#ffffff"],[0.8,"#ff6b6b"],[1,"#b71c1c"]],
            zmid=0,
            colorbar=dict(title=dict(text="Z-score", font=dict(color=COLORS["muted"])
                        ),
                        tickfont=dict(color=COLORS["muted"])
                    ),
            hovertemplate="<b>%{y} %{x}</b><br>Congestion : %{z:.2f} σ<extra></extra>",
        ))
        apply_dark_theme(fig_heat, f"Congestion réseau {selected_year} — Top {top_n} stations")
        with col_left:
            st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
    st.subheader(f"Profil horaire — {selected_station}")

    @st.cache_data(ttl=3600)
    def get_station_profile(station, year):
        return query("""
            SELECT heure, is_weekend,
                   ROUND(AVG(taux_congestion), 3)    AS moy,
                   ROUND(STDDEV(taux_congestion), 3) AS std
            FROM dataset_enrichi
            WHERE station = ? AND annee = ? AND taux_congestion IS NOT NULL
            GROUP BY heure, is_weekend ORDER BY is_weekend, heure
        """, [station, year])

    df_prof = get_station_profile(selected_station, selected_year)
    if not df_prof.empty:
        fig_p = go.Figure()
        for w, label, color, fill in [
            (0, "Semaine", COLORS["accent"],  "rgba(0,212,255,0.12)"),
            (1, "Weekend", COLORS["accent2"], "rgba(255,107,107,0.12)"),
        ]:
            d = df_prof[df_prof["is_weekend"] == w]
            if d.empty: continue
            fig_p.add_trace(go.Scatter(
                x=list(d["heure"]) + list(d["heure"].iloc[::-1]),
                y=list(d["moy"] + d["std"]) + list((d["moy"] - d["std"]).iloc[::-1]),
                fill="toself", fillcolor=fill, line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip"))
            fig_p.add_trace(go.Scatter(
                x=d["heure"], y=d["moy"], name=label,
                line=dict(color=color, width=2), mode="lines+markers", marker=dict(size=4),
                hovertemplate=f"<b>{label} %{{x}}h</b><br>%{{y:.2f}} σ<extra></extra>"))
        fig_p.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], opacity=0.5)
        apply_dark_theme(fig_p, f"Profil horaire — {selected_station} ({selected_year})")
        fig_p.update_layout(xaxis_title="Heure", yaxis_title="Congestion (z-score)")
        st.plotly_chart(fig_p, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# VUE 2 — MÉTÉO & TRAFIC
# ─────────────────────────────────────────────────────────────────────────────

elif view == "🌧️ Météo & Trafic":
    st.title("🌧️ Impact de la météo sur le trafic")

    col1, col2 = st.columns([2, 1])
    with col2:
        selected_hour  = st.selectbox("Heure", list(range(6, 23)), index=2,
                                      format_func=lambda h: f"{h}h")
        weather_metric = st.radio("Variable météo", ["precip_mm", "temp", "wind_kmh"])
        sample_size    = st.select_slider("Échantillon",
                                          [10_000, 50_000, 100_000, 500_000], value=100_000,
                                          format_func=lambda n: f"{n:,}".replace(",", " "))

    @st.cache_data(ttl=3600)
    def get_weather_scatter(station, hour, metric, n):
        return query(f"""
            SELECT {metric}, taux_congestion, is_weekend, annee
            FROM dataset_enrichi
            WHERE station = ? AND heure = ?
              AND taux_congestion IS NOT NULL AND {metric} IS NOT NULL AND is_greve = 0
            USING SAMPLE {n} ROWS
        """, [station, hour])

    df_w = get_weather_scatter(selected_station, selected_hour, weather_metric, sample_size)
    lbl  = {"precip_mm": "Précipitations (mm)", "temp": "Température (°C)", "wind_kmh": "Vent (km/h)"}
    if not df_w.empty:
        with col1:
            fig_s = px.scatter(df_w, x=weather_metric, y="taux_congestion",
                               color="is_weekend",
                               color_discrete_map={0: COLORS["accent"], 1: COLORS["accent2"]},
                               opacity=0.3, trendline="lowess",
                               trendline_color_override=COLORS["accent3"],
                               labels={weather_metric: lbl[weather_metric],
                                       "taux_congestion": "Congestion (σ)", "is_weekend": "Weekend"})
            apply_dark_theme(fig_s, f"{lbl[weather_metric]} vs Congestion — {selected_station} {selected_hour}h")
            st.plotly_chart(fig_s, use_container_width=True)

    st.divider()
    st.subheader("Distribution par intensité de pluie")

    @st.cache_data(ttl=3600)
    def get_rain_dist(station, year):
        return query("""
            SELECT CASE WHEN precip_mm < 0.1 THEN '☀️ Sec'
                        WHEN precip_mm < 2   THEN '🌦️ Légère'
                        WHEN precip_mm < 7   THEN '🌧️ Modérée'
                        ELSE '⛈️ Forte' END AS cat,
                   taux_congestion
            FROM dataset_enrichi
            WHERE station = ? AND annee = ?
              AND taux_congestion IS NOT NULL AND precip_mm IS NOT NULL AND is_greve = 0
            USING SAMPLE 200000 ROWS
        """, [station, year])

    df_r = get_rain_dist(selected_station, selected_year)
    if not df_r.empty:
        fig_b = px.box(df_r, x="cat", y="taux_congestion",
                       category_orders={"cat": ["☀️ Sec","🌦️ Légère","🌧️ Modérée","⛈️ Forte"]},
                       color="cat",
                       color_discrete_sequence=[COLORS["accent3"],COLORS["accent"],
                                                COLORS["accent"],COLORS["accent2"]],
                       labels={"taux_congestion": "Congestion (σ)", "cat": ""})
        apply_dark_theme(fig_b, f"Congestion par intensité de pluie — {selected_station} ({selected_year})")
        fig_b.update_layout(showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# VUE 3 — ÉVÉNEMENTS & PICS
# ─────────────────────────────────────────────────────────────────────────────

elif view == "🎭 Événements & Pics":
    st.title("🎭 Événements parisiens et pics de trafic")

    col1, col2 = st.columns(2)
    with col1:
        start_m = st.selectbox("Mois début", range(1, 13), index=0,
                               format_func=lambda m: datetime(2000, m, 1).strftime("%B"))
    with col2:
        end_m   = st.selectbox("Mois fin", range(1, 13), index=11,
                               format_func=lambda m: datetime(2000, m, 1).strftime("%B"))

    @st.cache_data(ttl=3600)
    def get_events_timeline(station, year, m0, m1):
        return query("""
            SELECT date, AVG(taux_congestion) AS congestion, SUM(nb_events) AS nb_events,
                   MAX(is_greve) AS is_greve, MAX(is_jour_ferie) AS is_ferie
            FROM dataset_enrichi
            WHERE station = ? AND annee = ? AND mois BETWEEN ? AND ?
              AND taux_congestion IS NOT NULL
            GROUP BY date ORDER BY date
        """, [station, year, m0, m1])

    df_ev = get_events_timeline(selected_station, selected_year, start_m, end_m)
    if not df_ev.empty:
        fig_tl = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=("Congestion (σ)", "Événements parisiens"),
                                row_heights=[0.7, 0.3])
        fig_tl.add_trace(go.Scatter(x=df_ev["date"], y=df_ev["congestion"],
                                    line=dict(color=COLORS["accent"], width=1.5),
                                    fill="tozeroy", fillcolor="rgba(0,212,255,0.1)",
                                    name="Congestion"), row=1, col=1)
        grv = df_ev[df_ev["is_greve"] == 1]
        if not grv.empty:
            fig_tl.add_trace(go.Scatter(x=grv["date"], y=grv["congestion"], mode="markers",
                                        name="Grève", marker=dict(color=COLORS["accent2"], size=7, symbol="x")),
                             row=1, col=1)
        fig_tl.add_trace(go.Bar(x=df_ev["date"], y=df_ev["nb_events"],
                                marker=dict(color=COLORS["accent"], opacity=0.6), name="Événements"),
                         row=2, col=1)
        fig_tl.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=COLORS["surface"],
            font=dict(color=COLORS["text"]), hovermode="x unified",
            margin=dict(l=40,r=20,t=60,b=40),
            title=dict(text=f"Timeline {selected_station} ({selected_year})",
                       font=dict(color=COLORS["accent"], size=16)))
        for r in [1, 2]:
            fig_tl.update_xaxes(gridcolor=COLORS["border"], row=r, col=1)
            fig_tl.update_yaxes(gridcolor=COLORS["border"], row=r, col=1)
        st.plotly_chart(fig_tl, use_container_width=True)

        st.subheader("Top 10 jours — impact événements")
        top10 = (df_ev[df_ev["nb_events"] > 0]
                 .assign(score=lambda d: d["nb_events"] * d["congestion"].clip(lower=0))
                 .nlargest(10, "score")[["date", "nb_events", "congestion", "is_greve"]])
        top10.columns = ["Date","Nb événements","Congestion (σ)","Grève"]
        st.dataframe(top10, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# VUE 4 — STATIONS IMPRÉVISIBLES
# ─────────────────────────────────────────────────────────────────────────────

elif view == "📊 Stations imprévisibles":
    st.title("📊 Stations imprévisibles")
    st.markdown("Variance du z-score de congestion sur 2015–2025.")

    @st.cache_data(ttl=3600)
    def get_variance_ranking():
        return query("""
            SELECT station, ROUND(AVG(variance_historique), 4) AS variance_moy,
                   ROUND(AVG(nb_vald_heure), 1) AS volume_moyen, rang_station
            FROM dataset_enrichi WHERE taux_congestion IS NOT NULL
            GROUP BY station, rang_station ORDER BY variance_moy DESC
        """)

    df_var = get_variance_ranking()
    col1, col2 = st.columns([3, 1])
    with col2:
        n_st     = st.slider("Stations à afficher", 10, 50, 25)
        color_by = st.radio("Couleur", ["variance_moy", "volume_moyen"])
    with col1:
        fig_bar = px.bar(df_var.head(n_st).sort_values("variance_moy"),
                         x="variance_moy", y="station", orientation="h", color=color_by,
                         color_continuous_scale=["#1a237e","#00d4ff","#ff6b6b"],
                         labels={"variance_moy":"Variance","station":""},
                         hover_data=["rang_station","volume_moyen"])
        apply_dark_theme(fig_bar, f"Top {n_st} stations imprévisibles")
        fig_bar.update_coloraxes(colorbar=dict(tickfont=dict(color=COLORS["muted"]), 
                                               title=dict(font=dict(color=COLORS["muted"])) # Structure correcte
                                               ))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    fig_sc = px.scatter(df_var, x="volume_moyen", y="variance_moy",
                        color="variance_moy", color_continuous_scale=["#1a237e","#00d4ff","#ff6b6b"],
                        labels={"volume_moyen":"Volume (val/h)","variance_moy":"Variance"})
    fig_sc.add_vline(x=df_var["volume_moyen"].median(), line_dash="dash",
                     line_color=COLORS["muted"], opacity=0.5)
    fig_sc.add_hline(y=df_var["variance_moy"].median(), line_dash="dash",
                     line_color=COLORS["muted"], opacity=0.5)
    apply_dark_theme(fig_sc, "Volume vs Variance — Quadrant ML")
    fig_sc.update_coloraxes(showscale=False)
    st.plotly_chart(fig_sc, use_container_width=True)

    with st.expander("📋 Table complète"):
        st.dataframe(df_var, use_container_width=True)
        st.download_button("⬇️ CSV", df_var.to_csv(index=False),
                           "stations_variance.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# VUE 5 — PRÉDICTION ML (API FastAPI)
# ─────────────────────────────────────────────────────────────────────────────

elif view == "🔮 Prédiction ML":
    st.title("🔮 Prédiction en temps réel")
    st.markdown(f"Interface vers l'API FastAPI. Lance `python flow.py serve` avant d'utiliser cette vue.")

    @st.cache_data(ttl=30)
    def check_api():
        try:
            return requests.get(f"{API_URL}/stations", timeout=2).status_code == 200
        except Exception:
            return False

    api_ok = check_api()
    if api_ok:
        st.success(f"✅ API connectée — `{API_URL}`")
    else:
        st.error(f"❌ API non disponible — `{API_URL}` · Lance `python flow.py serve`")

    col1, col2 = st.columns(2)
    with col1:
        pred_station  = st.selectbox("Station", stations_list, key="pred_s")
        pred_date     = st.date_input("Date", datetime.today())
        pred_hour     = st.slider("Heure", 0, 23, datetime.now().hour)
        manual_strike = st.checkbox("Simuler une grève")

    with col2:
        st.markdown("#### Résultat")
        if st.button("🔮 Lancer la prédiction", disabled=not api_ok):
            try:
                resp = requests.get(f"{API_URL}/predict",
                                    params={"station": pred_station,
                                            "datetime": f"{pred_date}T{pred_hour:02d}:00:00",
                                            "is_greve": int(manual_strike)},
                                    timeout=10)
                if resp.status_code == 200:
                    z = resp.json().get("taux_congestion_predit", 0)
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=z,
                        title={"text": "Congestion prédite (σ)", "font": {"color": COLORS["accent"]}},
                        gauge={"axis": {"range": [-3, 3]},
                               "bar": {"color": COLORS["accent"]},
                               "bgcolor": COLORS["surface"],
                               "steps": [{"range":[-3,-1.5],"color":"#1a237e"},
                                         {"range":[-1.5,1.5],"color":"#2d3748"},
                                         {"range":[1.5,3],"color":"#7f1d1d"}]},
                        number={"font": {"color": COLORS["accent"], "size": 40}, "suffix": " σ"},
                    ))
                    fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                        font=dict(color=COLORS["text"]),
                                        height=280, margin=dict(l=20,r=20,t=30,b=10))
                    st.plotly_chart(fig_g, use_container_width=True)
                    for lo, hi, msg in [(-3,-1.5,"🟦 Trafic très faible"),(-1.5,-0.5,"🟩 En dessous normale"),
                                        (-0.5,0.5,"⬜ Normal"),(0.5,1.5,"🟧 Au-dessus normale"),(1.5,3,"🟥 Très chargé")]:
                        if lo <= z < hi: st.info(msg); break
                else:
                    st.error(f"Erreur {resp.status_code}")
            except Exception as e:
                st.error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# VUE 6 — CARTE GÉOGRAPHIQUE
# ─────────────────────────────────────────────────────────────────────────────

elif view == "🌍 Carte géographique":
    st.title("🌍 Carte du réseau parisien")
    st.markdown("Chaque point = une station. **Taille** = volume. **Couleur** = métrique choisie.")

    @st.cache_data(ttl=86400)
    def load_geo_csv(path):
        """
        Lit le CSV géo IDFM.
        Colonne 'Geo Point' au format "lat, lon" → on split sur la virgule.
        On dédoublonne par nom de station (plusieurs lignes par station physique = plusieurs modes).
        """
        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep=";", encoding="latin-1")
        coords      = df["Geo Point"].str.split(",", expand=True)
        df["lat"]   = pd.to_numeric(coords[0].str.strip(), errors="coerce")
        df["lon"]   = pd.to_numeric(coords[1].str.strip(), errors="coerce")
        # Clé de jointure : upper + strip + suppression accents
        df["station_key"] = (df["nom_ZdC"].str.upper().str.strip()
                             .str.normalize("NFKD")
                             .str.encode("ascii", errors="ignore").str.decode("ascii"))
        return (df.dropna(subset=["lat","lon"])
                  .drop_duplicates(subset=["station_key"])
                [["station_key","nom_ZdC","lat","lon","mode","res_com","exploitant"]]
                .reset_index(drop=True))

    @st.cache_data(ttl=3600)
    def get_station_metrics_for_map(year):
        df = query("""
            SELECT station, ROUND(AVG(variance_historique),4) AS variance_moy,
                   ROUND(AVG(nb_vald_heure),1) AS volume_moyen, rang_station
            FROM dataset_enrichi WHERE annee = ? AND taux_congestion IS NOT NULL
            GROUP BY station, rang_station
        """, [year])
        df["station_key"] = (df["station"].str.upper().str.strip()
                             .str.normalize("NFKD")
                             .str.encode("ascii", errors="ignore").str.decode("ascii"))
        return df

    @st.cache_data(ttl=3600)
    def load_ml_metrics(path):
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()
        if not {"station","mae","rmse"}.issubset(df.columns):
            return None
        df["station_key"] = (df["station"].str.upper().str.strip()
                             .str.normalize("NFKD")
                             .str.encode("ascii", errors="ignore").str.decode("ascii"))
        return df

    try:
        df_geo = load_geo_csv(GEO_CSV_PATH)
    except FileNotFoundError:
        st.error(f"CSV géo introuvable : `{GEO_CSV_PATH}`. Définis la variable `GEO_CSV_PATH`.")
        st.stop()

    df_met = get_station_metrics_for_map(selected_year)
    df_ml  = load_ml_metrics(METRICS_CSV_PATH)
    df_map = df_geo.merge(df_met, on="station_key", how="inner")
    if df_ml is not None:
        df_map = df_map.merge(df_ml[["station_key","mae","rmse"]], on="station_key", how="left")
    else:
        df_map["mae"] = df_map["rmse"] = np.nan

    st.caption(f"✅ {len(df_map)} stations géolocalisées sur {len(df_geo)} dans le CSV")

    col_ctrl, col_map = st.columns([1, 4])
    with col_ctrl:
        color_metric = st.radio("Couleur", ["Variance","MAE modèle","Volume"])
        map_style    = st.selectbox("Fond de carte",
                                   ["carto-darkmatter","open-street-map","carto-positron"])
        min_vol      = st.slider("Volume minimum", 0, 500, 50)

    color_col = {"Variance":"variance_moy","MAE modèle":"mae","Volume":"volume_moyen"}[color_metric]
    color_lbl = {"variance_moy":"Variance z-score","mae":"MAE (σ)","volume_moyen":"Validations/h"}

    df_f = df_map[df_map["volume_moyen"] >= min_vol].copy()
    sv   = df_f["volume_moyen"].fillna(df_f["volume_moyen"].median())
    df_f["_size"] = 5 + 35 * (sv - sv.min()) / (sv.max() - sv.min() + 1e-9)

    if color_col == "mae" and df_f["mae"].isna().all():
        st.warning(f"Métriques ML non trouvées dans `{METRICS_CSV_PATH}`. Change la métrique de couleur.")
    else:
        with col_map:
            fig_map = px.scatter_mapbox(
                df_f, lat="lat", lon="lon",
                color=color_col, size="_size", size_max=35,
                color_continuous_scale=["#1a237e","#00b4d8","#ffffff","#ff6b6b","#b71c1c"],
                hover_name="station",
                hover_data={"lat":False,"lon":False,"_size":False,
                            "variance_moy":":.3f","volume_moyen":":.0f",
                            "mae":":.3f","rang_station":True,"res_com":True},
                mapbox_style=map_style, zoom=10,
                center={"lat":48.8566,"lon":2.3522}, height=600,
                labels={color_col: color_lbl[color_col]},
            )
            fig_map.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                margin=dict(l=0, r=0, t=0, b=0),
                coloraxis_colorbar=dict(
                    title=dict(
                        text=color_lbl[color_col],
                        font=dict(color=COLORS["muted"])
                    ),
                    tickfont=dict(color=COLORS["muted"])
                )
            )
            st.plotly_chart(fig_map, use_container_width=True)

        st.subheader(f"Top 10 — {color_metric}")
        top10 = (df_f.dropna(subset=[color_col]).nlargest(10, color_col)
                 [["station","variance_moy","volume_moyen","mae","rang_station","res_com"]])
        top10.columns = ["Station","Variance","Volume moy.","MAE","Rang","Ligne"]
        st.dataframe(top10, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# VUE 7 — GRÈVES & ANOMALIES
# ─────────────────────────────────────────────────────────────────────────────

elif view == "🚨 Grèves & Anomalies":
    st.title("🚨 Grèves & Anomalies historiques")
    st.markdown(
        "352 jours de grève détectés sur 2015–2025. "
        "Le flag `is_greve` capture toute baisse généralisée — y compris les confinements COVID."
    )

    @st.cache_data(ttl=3600)
    def get_strike_summary():
        return query("""
            SELECT annee,
                   COUNT(DISTINCT CASE WHEN is_greve=1 THEN date END) AS jours_greve,
                   ROUND(AVG(CASE WHEN is_greve=1 THEN taux_congestion END), 3) AS cong_greve,
                   ROUND(AVG(CASE WHEN is_greve=0 THEN taux_congestion END), 3) AS cong_normale
            FROM dataset_enrichi WHERE taux_congestion IS NOT NULL
            GROUP BY annee ORDER BY annee
        """)

    df_s = get_strike_summary()
    df_s["type"] = df_s["annee"].apply(
        lambda y: "⚠️ COVID + Grève" if y in [2020,2021] else "🚇 Grève sociale")

    fig_bars = px.bar(df_s, x="annee", y="jours_greve", color="type",
                      color_discrete_map={"🚇 Grève sociale":COLORS["accent2"],
                                          "⚠️ COVID + Grève":COLORS["accent3"]},
                      text="jours_greve",
                      labels={"annee":"Année","jours_greve":"Jours détectés","type":""})
    fig_bars.update_traces(textposition="outside", textfont=dict(color=COLORS["text"]))
    apply_dark_theme(fig_bars, "Jours de grève détectés par année")
    st.plotly_chart(fig_bars, use_container_width=True)

    st.divider()
    st.subheader(f"Profil horaire — Grève vs Normal · {selected_station}")

    @st.cache_data(ttl=3600)
    def get_strike_profile(station):
        return query("""
            SELECT heure, is_greve,
                   ROUND(AVG(taux_congestion), 3)    AS moy,
                   ROUND(STDDEV(taux_congestion), 3) AS std,
                   COUNT(*) AS n
            FROM dataset_enrichi
            WHERE station = ? AND taux_congestion IS NOT NULL AND annee NOT IN (2020,2021)
            GROUP BY heure, is_greve ORDER BY is_greve, heure
        """, [station])

    df_sp = get_strike_profile(selected_station)
    if not df_sp.empty:
        fig_sp = go.Figure()
        for flag, label, color, fill in [
            (0,"Jour normal",COLORS["accent"],"rgba(0,212,255,0.12)"),
            (1,"Jour de grève",COLORS["accent2"],"rgba(255,107,107,0.12)"),
        ]:
            d = df_sp[df_sp["is_greve"] == flag]
            if d.empty: continue
            fig_sp.add_trace(go.Scatter(
                x=list(d["heure"])+list(d["heure"].iloc[::-1]),
                y=list(d["moy"]+d["std"])+list((d["moy"]-d["std"]).iloc[::-1]),
                fill="toself", fillcolor=fill, line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip"))
            fig_sp.add_trace(go.Scatter(
                x=d["heure"], y=d["moy"], name=label,
                line=dict(color=color, width=2.5), mode="lines+markers", marker=dict(size=5),
                hovertemplate=f"<b>{label} %{{x}}h</b><br>%{{y:.2f}} σ (n=%{{customdata}})<extra></extra>",
                customdata=d["n"]))
        fig_sp.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], opacity=0.4)
        apply_dark_theme(fig_sp, f"Grève vs Normal — {selected_station} (hors COVID)")
        fig_sp.update_layout(xaxis_title="Heure", yaxis_title="Congestion (z-score)")
        st.plotly_chart(fig_sp, use_container_width=True)

    st.divider()
    st.subheader("Stations les plus impactées par les grèves")

    @st.cache_data(ttl=3600)
    def get_strike_impact():
        return query("""
            SELECT station,
                   ROUND(AVG(CASE WHEN is_greve=1 THEN taux_congestion END)
                       - AVG(CASE WHEN is_greve=0 THEN taux_congestion END), 3) AS impact,
                   ROUND(AVG(nb_vald_heure), 0) AS volume_moyen
            FROM dataset_enrichi
            WHERE taux_congestion IS NOT NULL AND annee NOT IN (2020,2021)
            GROUP BY station
            HAVING COUNT(CASE WHEN is_greve=1 THEN 1 END) > 100
            ORDER BY impact ASC LIMIT 30
        """)

    df_imp = get_strike_impact()
    if not df_imp.empty:
        col1, col2 = st.columns([3, 1])
        with col2:
            n_show = st.slider("Nb stations", 10, 30, 20)
        with col1:
            fig_imp = px.bar(df_imp.head(n_show).sort_values("impact"),
                             x="impact", y="station", orientation="h", color="impact",
                             color_continuous_scale=["#b71c1c","#ff6b6b","#2d3748"],
                             text="impact", labels={"impact":"Impact grève (σ)","station":""},
                             hover_data=["volume_moyen"])
            fig_imp.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                                  textfont=dict(color=COLORS["muted"], size=9))
            apply_dark_theme(fig_imp, f"Top {n_show} stations — Impact grèves (hors COVID)")
            fig_imp.update_coloraxes(showscale=False)
            fig_imp.add_vline(x=0, line_dash="dash", line_color=COLORS["muted"], opacity=0.5)
            st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    st.subheader("🔬 Zoom 2020 — Décomposition grève vs confinement")

    @st.cache_data(ttl=3600)
    def get_covid_2020():
        return query("""
            SELECT date, AVG(taux_congestion) AS congestion, MAX(is_greve) AS is_greve
            FROM dataset_enrichi WHERE annee = 2020 AND taux_congestion IS NOT NULL
            GROUP BY date ORDER BY date
        """)

    df_2020 = get_covid_2020()
    if not df_2020.empty:
        fig_2020 = go.Figure()
        for start, end, label in [
            ("2020-03-17","2020-05-11","Confinement 1"),
            ("2020-10-30","2020-12-15","Confinement 2"),
        ]:
            fig_2020.add_vrect(x0=start, x1=end, fillcolor="rgba(255,211,61,0.12)",
                               layer="below", line_width=0,
                               annotation_text=label, annotation_position="top left",
                               annotation_font=dict(color=COLORS["accent3"], size=10))
        fig_2020.add_trace(go.Scatter(x=df_2020["date"], y=df_2020["congestion"],
                                      fill="tozeroy", fillcolor="rgba(0,212,255,0.1)",
                                      line=dict(color=COLORS["accent"], width=1.5),
                                      name="Congestion réseau"))
        grv20 = df_2020[df_2020["is_greve"] == 1]
        fig_2020.add_trace(go.Scatter(x=grv20["date"], y=grv20["congestion"],
                                      mode="markers", name="is_greve=1",
                                      marker=dict(color=COLORS["accent2"], size=5, opacity=0.7)))
        fig_2020.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], opacity=0.4)
        apply_dark_theme(fig_2020, "Trafic réseau 2020 — Grèves détectées + Confinements")
        fig_2020.update_layout(hovermode="x unified")
        st.plotly_chart(fig_2020, use_container_width=True)

        n_tot   = len(grv20)
        n_covid = int(grv20[grv20["date"].between("2020-03-17","2020-05-11") |
                             grv20["date"].between("2020-10-30","2020-12-15")].shape[0])
        c1,c2,c3 = st.columns(3)
        c1.metric("Jours is_greve=1 en 2020",           str(n_tot))
        c2.metric("→ Pendant confinements (faux +)",     str(n_covid), delta_color="inverse")
        c3.metric("→ Hors confinements (grèves réelles)", str(n_tot - n_covid))


# ─────────────────────────────────────────────────────────────────────────────
# VUE 8 — PERFORMANCE DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

elif view == "📈 Performance modèle":
    st.title("📈 Performance du modèle LightGBM")
    st.markdown("MAE globale **0.371 σ** / RMSE **0.636 σ**. Split train 2015–2023 → test 2024–2025.")

    @st.cache_data(ttl=3600)
    def load_station_metrics(path):
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()
        return df if {"station","mae","rmse"}.issubset(df.columns) else None

    df_met = load_station_metrics(METRICS_CSV_PATH)

    if df_met is None:
        st.warning(f"Fichier de métriques non trouvé : `{METRICS_CSV_PATH}`. Lance `python flow.py evaluate`.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("MAE globale",           f"{df_met['mae'].mean():.3f} σ")
        c2.metric("RMSE globale",          f"{df_met['rmse'].mean():.3f} σ")
        c3.metric("Meilleure MAE",         f"{df_met['mae'].min():.3f} σ",
                  delta=df_met.loc[df_met['mae'].idxmin(),'station'], delta_color="off")
        c4.metric("Pire MAE",              f"{df_met['mae'].max():.3f} σ",
                  delta=df_met.loc[df_met['mae'].idxmax(),'station'], delta_color="off")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            fig_h = px.histogram(df_met, x="mae", nbins=40,
                                 color_discrete_sequence=[COLORS["accent"]],
                                 labels={"mae":"MAE (σ)"})
            fig_h.add_vline(x=df_met["mae"].mean(), line_dash="dash",
                            line_color=COLORS["accent3"],
                            annotation_text=f"Moy. {df_met['mae'].mean():.3f}",
                            annotation_font=dict(color=COLORS["accent3"]))
            apply_dark_theme(fig_h, "Distribution des MAE")
            st.plotly_chart(fig_h, use_container_width=True)

        with col2:
            fig_sc2 = px.scatter(df_met, x="mae", y="rmse", hover_name="station",
                                 color="mae", color_continuous_scale=["#1a237e","#00d4ff","#ff6b6b"],
                                 labels={"mae":"MAE (σ)","rmse":"RMSE (σ)"}, opacity=0.7)
            mx = max(df_met["mae"].max(), df_met["rmse"].max())
            fig_sc2.add_trace(go.Scatter(x=[0,mx],y=[0,mx], mode="lines",
                                         line=dict(color=COLORS["muted"],dash="dash",width=1),
                                         name="RMSE=MAE"))
            apply_dark_theme(fig_sc2, "MAE vs RMSE")
            fig_sc2.update_coloraxes(showscale=False)
            st.plotly_chart(fig_sc2, use_container_width=True)

        st.divider()
        col_b, col_w = st.columns(2)
        with col_b:
            top15 = df_met.nsmallest(15,"mae")
            fig_b = px.bar(top15.sort_values("mae"), x="mae", y="station", orientation="h",
                           color="mae", color_continuous_scale=["#00d4ff","#1a237e"],
                           text="mae", labels={"mae":"MAE","station":""})
            fig_b.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                                textfont=dict(color=COLORS["muted"],size=9))
            apply_dark_theme(fig_b, "✅ Top 15 — meilleures stations")
            fig_b.update_coloraxes(showscale=False)
            st.plotly_chart(fig_b, use_container_width=True)
        with col_w:
            bot15 = df_met.nlargest(15,"mae")
            fig_w = px.bar(bot15.sort_values("mae"), x="mae", y="station", orientation="h",
                           color="mae", color_continuous_scale=["#ff6b6b","#b71c1c"],
                           text="mae", labels={"mae":"MAE","station":""})
            fig_w.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                                textfont=dict(color=COLORS["muted"],size=9))
            apply_dark_theme(fig_w, "❌ Top 15 — pires stations")
            fig_w.update_coloraxes(showscale=False)
            st.plotly_chart(fig_w, use_container_width=True)

    st.divider()



