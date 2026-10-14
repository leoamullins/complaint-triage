#!/usr/bin/env bash
set -euo pipefail
BASE=https://files.consumerfinance.gov/f/documents
mkdir -p data/narratives
for f in \
  CCDB_Export_1_December_2011_through_April_2018 \
  CCDB_Export_2_May_2018_through_April_2021 \
  CCDB_Export_3_May_2021_through_October_2022 \
  CCDB_Export_4_November_2022_through_August_2023 \
  CCDB_Export_5_September_2023_through_March_2024 \
  CCDB_Export_6_April_2024_through_July_2024 \
  CCDB_Export_7_August_2024_through_October_2024 \
  CCDB_Export_8_November_2024_through_December_2024 \
  CCDB_Export_9_January_2025_through_February_2025 \
  CCDB_Export_10_March_2025_through_April_2025 \
  CCDB_Export_11_May_2025_through_June_2025 \
  CCDB_Export_12_July_2025_through_August_2025 \
  CCDB_Export_13_September_2025_through_October_2025 \
  CCDB_Export_14_November_2025_through_December_2025 \
  CCDB_Export_15_January_2026_through_February_2026 \
  CCDB_Export_16_March_2026 \
  CCDB_Export_17_April_2026 \
  CCDB_Export_18_May_2026 \
  CCDB_Export_19_June_2026 \
  CCDB_Export_20_July_2026 \
  CCDB_Export_21_August_2026
do
  curl -L -o "data/narratives/$f.zip" "$BASE/$f.zip"
done