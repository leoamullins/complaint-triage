#!/usr/bin/env bash
set -euo pipefail
for z in data/narratives/*.zip; do
  echo "Loading $z"
  unzip -p "$z" | psql complaints -c "\copy raw_narratives FROM STDIN CSV HEADER"
done