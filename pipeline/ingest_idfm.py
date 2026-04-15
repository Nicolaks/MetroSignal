import pandas as pd
import duckdb
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#Configuration
RAW_DIR = Path("data/raw")
DB_PATH = Path("data/warehouse.duckdb")

COLUMN_MAP = {
    "JOUR": "date",
    "CODE_STIF_TRNS": "code_transporteur",
    "CODE_STIF_RES": "code_reseau",
    "CODE_STIF_ARRET": "code_arret",
    "LIBELLE_ARRET": "station",
    "ID_ZDC": "id_zdc",
    "CATEGORIE_TITRE": "categorie_titre",
    "NB_VALD": "nb_vald",
}

def load_csv(path: Path) -> pd.DataFrame:
    logger.info("Lecture du fichier %s ....", path.name)
    
    for encoding in ["latin-1", "utf-16", "utf-8", 'utf-16-le', 'utf-16-be']:
        for sep in ["\t", ";", ","]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
                df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
                
                if "JOUR" in df.columns:
                    logger.info("Encodage détecté : %s", encoding)
                    break
            except Exception:
                continue
        else:
            continue
        break
    else:
        raise ValueError(f"Impossible de lire {path.name} avec les encodages testés")
    
    df = df.rename(columns=COLUMN_MAP)
    df = df[[col for col in COLUMN_MAP.values() if col in df.columns]]
    df["code_arret"] = (
        df["code_arret"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .replace(["ND", "NaN", "None", ""], None)
    )
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True, errors="coerce")
    df["nb_vald"] = (
        df["nb_vald"]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )
    
    df = df.dropna(subset=["date", "station"])
    
    logger.info("Chargé : %d lignes, %d stations uniques", len(df), df["station"].nunique())
    return df

def load_profil(path: Path) -> pd.DataFrame:
    logger.info("Lecture du profil horaire %s ...", path.name)
    
    df = None
    
    for encoding in ["latin-1", "utf-8", "utf-8-sig", "utf-16", "utf-16-le"]:
        for sep in [";", "\t", ","]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
                df.columns = (df.columns.str.replace("\ufeff", "", regex=False).str.strip())
            
                if "TRNC_HORR_60" in df.columns:
                    logger.info("Profil OK (encoding=%s, sep=%s)", encoding, sep)
                    break
            
            except Exception:
                continue
        if df is not None and "TRNC_HORR_60" in df.columns:
            break
        
    if df is None or "TRNC_HORR_60" not in df.columns:
        raise ValueError(f"Impossible de lire le profil {path.name}")
    
    time_col = next(
        (c for c in ["TRNC_HORR_60", "TRNC_HORAIRE", "HEURE"] if c in df.columns),
        None
    )
    
    if time_col is None:
        raise ValueError(f"Aucunne colonne horaire trouvée: {df.columns}")
                
    df["heure"] = (
        df[time_col]
        .astype(str)
        .str.extract(r"^(\d+)H")
        .squeeze()
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )
    
    df = df.rename(columns={
        "CODE_STIF_TRNS": "code_transporteur",
        "CODE_STIF_RES": "code_reseau",
        "CODE_STIF_ARRET": "code_arret",
        "LIBELLE_ARRET": "station",
        "ID_ZDC": "id_zdc",
        "ID_REFA_LDA" : "id_zdc",
        "CAT_JOUR": "cat_jour",
        "pourc_validations": "pct_validations", # format S1/S2 et avant 2024
        "Pourcentage_validations": "pct_validations", # format T3/T4 2024
    })
    
    df["code_arret"] = (
        df["code_arret"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .replace(["ND", "NaN", "None", ""], None)
    )
    
    df["pct_validations"] = (
        df["pct_validations"]
        .astype(str)
        .str.replace(",", ".", regex=False)  # au cas où format français
        .pipe(pd.to_numeric, errors="coerce")
)

    df = df[["code_arret", "cat_jour", "heure", "pct_validations"]]
    logger.info("Colonnes profil détectées: %s", list(df.columns))
    logger.info("Profil chargé : %d lignes", len(df))
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame :
    logger.info("Ajout des features temporelles ...")
    
    df["heure"] = df["date"].dt.hour
    df["jour_semaine"] = df["date"].dt.dayofweek
    df["semaine_annee"] = df["date"].dt.isocalendar().week.astype(int)
    df["mois"] = df["date"].dt.month
    df["annee"] = df["date"].dt.year
    df["is_weekend"] = df["jour_semaine"].isin([5,6]).astype(int)
    
    logger.info("Features temporelles ajoutées")
    return df

def merge_nb_profil(df_nb: pd.DataFrame, df_profil: pd.DataFrame) -> pd.DataFrame:
    logger.info("Jointure NB_FER x PROFIL_FER ...")
    
    def get_cat_jour(row):
        if row["is_weekend"] == 0:
            return "JOHV"
        elif row["jour_semaine"] == 5:
            return "SAHV"
        else:
            return "DIJFP"
        
    df_nb["cat_jour"] = df_nb.apply(get_cat_jour, axis=1)
    
    df = df_nb.merge(
        df_profil,
        on=["code_arret", "cat_jour", "heure"],
        how="left"
    )
    
    df["pct_validations"] = df["pct_validations"].fillna(0)
    
    df["nb_vald_heure"] = (
        df["nb_vald"].astype(float) * df["pct_validations"] / 100
    ).round().astype(int)
    
    logger.info("Résultat : %d lignes", len(df))
    return df

def aggregate_by_station_hour(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Agrégation par station / jour / heure ...")
    
    df_agg = (
        df.groupby(["date", "heure", "station", "code_arret", "jour_semaine", "semaine_annee", "mois", "annee", "is_weekend"])
        .agg(nb_vald_heure=("nb_vald_heure", "sum"))
        .reset_index()
    )
    
    logger.info("Agrégé : %d lignes, %d stations uniques", len(df_agg), df_agg["station"].nunique())
    return df_agg

def load_to_duckdb(df: pd.DataFrame, db_path: Path, annee: int) -> None:
    logger.info("Chargement dans DuckDB -> %s ...", db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS validations (
            date DATE,
            heure INTEGER,
            station VARCHAR,
            code_arret VARCHAR,
            jour_semaine INTEGER,
            semaine_annee INTEGER,
            mois INTEGER,
            annee INTEGER,
            is_weekend INTEGER,
            nb_vald_heure INTEGER
            )
        """)
    
    con.execute(f"DELETE FROM validations WHERE annee = {annee}")
    con.execute("INSERT INTO validations SELECT * FROM df")
    
    count = con.execute(f"SELECT COUNT(*) FROM validations WHERE annee = {annee}").fetchone()[0]
    total = con.execute("SELECT COUNT(*) FROM validations").fetchone()[0]
    con.close()
    
    logger.info("✅ Année %d : %d lignes | Total BDD : %d lignes", annee, count, total)

def run(annee: int, raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> pd.DataFrame:
    dossier = raw_dir / f"data-rf-{annee}"
    
    if annee == 2025:
        fichiers_nb = sorted(dossier.glob("*nombre-validations*.csv"))
        fichiers_profil = sorted(dossier.glob("*profils-horaires*.csv"))
        
    else:
        fichiers_nb = sorted(
            list(dossier.glob("*NB_FER*"))
        )
        
        fichiers_profil = sorted(
            list(dossier.glob("*PROFIL_FER*"))
        )
    
    if not fichiers_nb:
        raise FileNotFoundError(f"Aucun fichier NB_FER trouvé dans {dossier}")
    
    frames = []
    
    if fichiers_profil:
        for nb_path, profil_path in zip(fichiers_nb, fichiers_profil):
            df_nb = load_csv(nb_path)
            df_nb = add_time_features(df_nb)
            
            df_profil = load_profil(profil_path)
            
            df_merged = merge_nb_profil(df_nb, df_profil)
            frames.append(aggregate_by_station_hour(df_merged))
    else:
        logger.warning(f"Aucun profil trouvé pour {annee} -> fallback journalier")
        
        for nb_path in fichiers_nb:
            df_nb = load_csv(nb_path)
            df_nb = add_time_features(df_nb)
            
            df_nb["heure"] = 0
            df_nb["nb_vald_heure"] = df_nb["nb_vald"]
            
            frames.append(aggregate_by_station_hour(df_nb))
        
        
    df_final = pd.concat(frames, ignore_index=True)
    load_to_duckdb(df_final, db_path, annee)
    
    return df_final

if __name__== "__main__":
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    con = duckdb.connect(str(DB_PATH))
    
    #con.execute("DROP TABLE IF EXISTS validations")
    for annee in [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        df = run(annee=annee)
        
    print(con.execute("""
                      SELECT annee, COUNT(*) as nb_lignes
                      FROM validations
                      GROUP BY annee
                      ORDER BY annee
                      """).df())
    con.close()
