import os
import sys
from sqlalchemy import text

# Allow running as a standalone script from /app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_connection.database import engine

# SQL files are copied into the image at /app/database/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_FILE = os.path.join(BASE_DIR, "database", "01_schema.sql")
SEED_FILE = os.path.join(BASE_DIR, "database", "02_seed.sql")


def run_sql_file(connection, filepath: str) -> None:
    print(f"Running {filepath} ...")
    with open(filepath, "r", encoding="utf-8") as f:
        sql = f.read()
    connection.execute(text(sql))
    print(f"  ✓ {os.path.basename(filepath)} applied successfully.")


def init_db() -> None:
    print("=== Database initialisation started ===")

    for path in (SCHEMA_FILE, SEED_FILE):
        if not os.path.exists(path):
            print(f"ERROR: SQL file not found: {path}", file=sys.stderr)
            sys.exit(1)

    try:
        with engine.begin() as conn:
            run_sql_file(conn, SCHEMA_FILE)
            run_sql_file(conn, SEED_FILE)
    except Exception as exc:
        print(f"ERROR: Database initialisation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=== Database initialisation complete ===")


if __name__ == "__main__":
    init_db()
