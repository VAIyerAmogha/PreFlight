# Error analysis (tuning set, 153 files, run run_20261030_154227)

## Column type misses: 149

| Labelled | Detected | Misses | Flagged for review | Top files |
|---|---|---|---|---|
| ID | numeric | 112 | 111 | DAAG_cfseal.csv (3), COUNT_fishing.csv (3), AER_USGasB.csv (3) |
| categorical | text | 22 | 17 | week7_starwars.csv (5), carData_Chile.csv (2), tdf_winners.csv (2) |
| categorical | numeric | 5 | 5 | passwords.csv (1), big_epa_cars.csv (1), volcano.csv (1) |
| numeric | ID | 3 | 3 | rainfall.csv (2), DAAG_antigua.csv (1) |
| text | categorical | 2 | 1 | week4_australian_salary.csv (1), weather-check.csv (1) |
| ID | categorical | 2 | 1 | big_epa_cars.csv (1), national_parks.csv (1) |
| categorical | constant | 1 | 1 | biopics.csv (1) |
| ID | datetime | 1 | 1 | test1.csv (1) |
| numeric | text | 1 | 1 | horror_movies.csv (1) |

## Dialect misses: 13 files

Wrong part counts: encoding 7, delimiter 4, header_row_index 2

- bank-full.csv: delimiter: ';' -> ','
- biopics.csv: encoding: 'iso-8859-1' -> 'utf-8'
- bom.csv: encoding: 'utf-8-sig' -> 'utf-8'
- food-world-cup-data.csv: encoding: 'iso-8859-1' -> 'utf-8'
- iris.data: header_row_index: None -> 0
- issue-1009-semicolon.csv: delimiter: ';' -> ','
- police_killings.csv: encoding: 'iso-8859-1' -> 'utf-8'
- skip-rows-before-headers.csv: header_row_index: 1 -> 0
- table-delimiter.csv: delimiter: ';' -> ','
- table-with-accents.csv: encoding: 'iso-8859-1' -> 'utf-8'
- week4_australian_salary.csv: encoding: 'iso-8859-1' -> 'utf-8'
- week7_starwars.csv: encoding: 'iso-8859-1' -> 'utf-8'
- winequality-red.csv: delimiter: ';' -> ','
