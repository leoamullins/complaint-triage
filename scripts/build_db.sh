#!/usr/bin/env bash
# build the complaints database from the raw CFPB files, in order
# needs: data/complaints.csv (CFPB bulk export) and data/narratives/*.zip
#        (scripts/download_narratives.sh), and an existing database
set -euo pipefail
DB=${PGDATABASE:-complaints}
cd "$(dirname "$0")/.."

run() { echo "== $1"; psql -v ON_ERROR_STOP=1 "$DB" -f "$1"; }

run sql/01_schema.sql
echo "== loading data/complaints.csv"
psql -v ON_ERROR_STOP=1 "$DB" -c "\copy raw_complaints FROM 'data/complaints.csv' CSV HEADER"
run sql/02_complaints.sql

run sql/03_raw_narratives.sql
scripts/load_narratives.sh
run sql/04_narratives.sql

run sql/05_complaints_text.sql
run sql/06_model_data.sql
echo "done; next: uv run python scripts/export.py"
