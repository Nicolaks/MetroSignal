import argparse
import logging
import duckdb

from pathlib import Path

DB_PATH = Path("data/warehouse.duckdb")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def cmd_idfm(args):
    from pipeline.ingest_idfm import run
    
    annees = list(range(args.start, args.end + 1))
    logger.info("Ingestion IDFM pour les années %s", annees)
    for annee in annees:
        run(annee=annee)
        
def cmd_weather(args):
    from pipeline.ingest_weather import run
    
    run(start_date=args.start_date, end_date=args.end_date, db_path=DB_PATH)
    
def cmd_transform(args):
    from pipeline.transform import run
    
    run()
    
def cmd_train(args):
    from pipeline.ml.train import run
    
    run()
    
def cmd_evaluate(args):
    from pipeline.ml.train import run_evaluate
    run_evaluate()
    
def cmd_events(args):
    from pipeline.ingest_events import run
    
    run(start_date=args.start_date, end_date=args.end_date, db_path=DB_PATH)
    
def cmd_inspect(args):
    con = duckdb.connect(str(DB_PATH))
    tables = con.execute("SHOW TABLES").df()
    
    if tables.empty:
        print("Aucune table dans la base.")
        con.close()
        return
    
    print(f"\n{'='*60}")
    print(f" Base : {DB_PATH}")
    print(f"{'='*60}")
    
    for table in tables["name"].tolist():
        
        if args.table and args.table != table:
            continue
        
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        schema = con.execute(f"DESCRIBE {table}").df()
        sample = con.execute(f"SELECT * FROM {table} USING SAMPLE 20").df()

        print(f"\n📦 {table.upper()} — {count:,} lignes")
        print(f"{'─'*40}")

        for _, row in schema.iterrows():
            print(f"  {row['column_name']:<25} {row['column_type']}")

        
        print(f"\n  Sample (10 lignes) :")
        print(sample.to_string(index=False))
        print()

    con.close()
    
def cmd_all(args):
    logger.info("=== Lancement pipeline complet ===")
    
    cmd_idfm(args)
    cmd_weather(args)
    cmd_events(args)
    cmd_transform(args)
    
    logger.info("=== Pipeline complet terminé ===")
    
def main():
    parser = argparse.ArgumentParser(
        description="MetroSignal — Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Exemples :
                python flow.py idfm --start 2023 --end 2025
                python flow.py weather --start-date 2023-01-01 --end-date 2025-12-31
                python flow.py events --start-date 2023-01-01 --end-date 2025-12-31
                python flow.py transform
                python flow.py inspect
                python flow.py inspect --table validations --sample
                """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- IDFM ---
    p_idfm = subparsers.add_parser("idfm", help="Ingestion données IDFM")
    p_idfm.add_argument("--start", type=int, default=2015, help="Année de début (défaut: 2015)")
    p_idfm.add_argument("--end",   type=int, default=2025, help="Année de fin (défaut: 2025)")
    p_idfm.set_defaults(func=cmd_idfm)

    # --- WEATHER ---
    p_weather = subparsers.add_parser("weather", help="Ingestion météo Open-Meteo")
    p_weather.add_argument("--start-date", default="2015-01-01", help="Date début (défaut: 2015-01-01)")
    p_weather.add_argument("--end-date",   default="2025-12-31", help="Date fin (défaut: 2025-12-31)")
    p_weather.set_defaults(func=cmd_weather)

    # --- EVENTS ---
    p_events = subparsers.add_parser("events", help="Ingestion événements OpenAgenda")
    p_events.add_argument("--start-date", default="2015-01-01", help="Date début (défaut: 2015-01-01)")
    p_events.add_argument("--end-date",   default="2025-12-31", help="Date fin (défaut: 2025-12-31)")
    p_events.set_defaults(func=cmd_events)

    # --- TRANSFORM ---
    p_transform = subparsers.add_parser("transform", help="Calcul dataset_enrichi")
    p_transform.set_defaults(func=cmd_transform)
    
    # --- ALL ---
    p_all = subparsers.add_parser("all", help="Lancer le pipeline complet dans l'ordre")
    p_all.add_argument("--start",      type=int, default=2015,       help="Année début IDFM")
    p_all.add_argument("--end",        type=int, default=2025,       help="Année fin IDFM")
    p_all.add_argument("--start-date", default="2015-01-01",         help="Date début météo/events")
    p_all.add_argument("--end-date",   default="2025-12-31",         help="Date fin météo/events")
    p_all.set_defaults(func=cmd_all)

    # --- INSPECT ---
    p_inspect = subparsers.add_parser("inspect", help="Inspecter la base DuckDB")
    p_inspect.add_argument("--table",  type=str, default=None, help="Filtrer sur une table spécifique")
    p_inspect.add_argument("--sample", action="store_true",    help="Afficher 5 lignes de sample")
    p_inspect.set_defaults(func=cmd_inspect)
    
    p_train = subparsers.add_parser("train", help="Entraînement modèle LightGBM")
    p_train.set_defaults(func=cmd_train)
    
    p_evaluate = subparsers.add_parser("evaluate", help="Évaluation du modèle sauvegardé")
    p_evaluate.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
        
        