import pandas as pd
import duckdb
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#Configuration
RAW_DIR = Path("data/raw/data-rf-2024")
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
    
    df = pd.read_csv(path, sep="\t", encoding="latin-1", low_memory=False)
    df = df.rename(columns=COLUMN_MAP)
    df = df[[col for col in COLUMN_MAP.values() if col in df.columns]]
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
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
    
    df = pd.read_csv(path, sep="\t", encoding="latin-1", low_memory=False)
    
    df["heure"] = (
        df["TRNC_HORR_60"]
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
        "CAT_JOUR": "cat_jour",
        "pourc_validations": "pct_validations",
    })
    
    df["pct_validations"] = (
        df["pct_validations"]
        .astype(str)
        .str.replace(",", ".", regex=False)  # au cas où format français
        .pipe(pd.to_numeric, errors="coerce")
)
    
    df = df[["code_arret", "cat_jour", "heure", "pct_validations"]]
    
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

def load_to_duckdb(df: pd.DataFrame, db_path: Path) -> None:
    logger.info("Chargement dans DuckDB -> %s ...", db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    con = duckdb.connect(str(db_path))
    con.execute("DROP TABLE IF EXISTS validations")
    con.execute("CREATE TABLE validations AS SELECT * FROM df")
    count = con.execute("SELECT COUNT(*) FROM validations").fetchone()[0]
    con.close()
    
    logger.info("✅ Table 'validations' : %d lignes chargées", count)

def run(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> pd.DataFrame:
    df_nb = load_csv(raw_dir / "2024_S1_NB_FER.txt")
    df_nb = add_time_features(df_nb)
    df_profil = load_profil(raw_dir / "2024_S1_PROFIL_FER.txt")
    df_merged = merge_nb_profil(df_nb, df_profil)
    df_final = aggregate_by_station_hour(df_merged)
    load_to_duckdb(df_final, db_path)
    return df_final

if __name__== "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = run()
    con = duckdb.connect(str(DB_PATH))
    print(con.execute("SELECT station, heure, nb_vald_heure FROM validations LIMIT 5").df())
    con.close()
