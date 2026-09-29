#!/usr/bin/env python3
"""
Full Corpus Acquisition and Ground-Truth Labeling Pipeline for Phase 1 (M0).

Sources 200+ real-world CSV files across diverse open repositories:
- Frictionless messy CSV benchmarks (MIT)
- pandas test fixtures (BSD-3-Clause)
- UCI Machine Learning Repository (CC BY 4.0)
- Seaborn benchmark datasets (BSD-3-Clause)
- FiveThirtyEight open datasets (MIT / CC BY 4.0)
- TidyTuesday messy open datasets (MIT / CC0)
- Rdatasets CRAN packages (GPL-2 / GPL-3 / CC0 / MIT)

Generates:
- corpus/raw/<file_id>.<ext>
- corpus/labels/<file_id>.json
- corpus/labels/sources.csv
"""

import os
import sys
import json
import csv
import io
import re
import zipfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

RAW_DIR = Path("corpus/raw")
LABELS_DIR = Path("corpus/labels")
SOURCES_FILE = LABELS_DIR / "sources.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PreFlight-Benchmark/1.0"


def fetch_url(url: str, timeout: int = 20) -> bytes:
    """Download raw bytes with User-Agent and timeout."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# -------------------------------------------------------------
# Curated Dataset Catalog
# -------------------------------------------------------------

def build_dataset_manifest() -> List[Dict[str, Any]]:
    manifest = []

    # 1. Frictionless Messy Benchmarks (MIT)
    frictionless_base = "https://raw.githubusercontent.com/frictionlessdata/frictionless-py/main/data"
    fl_items = [
        ("fl_latin1", "latin1.csv", f"{frictionless_base}/latin1.csv", "iso-8859-1", ",", 0, "Frictionless Data", "MIT", "Non-UTF8 Latin1 encoding with special character (c)"),
        ("fl_semicolon", "issue-1009-semicolon.csv", f"{frictionless_base}/issue-1009-semicolon.csv", "utf-8", ";", 0, "Frictionless Data", "MIT", "Semicolon delimiter with leading-zero ID"),
        ("fl_bom", "bom.csv", f"{frictionless_base}/bom.csv", "utf-8-sig", ",", 0, "Frictionless Data", "MIT", "UTF-8 byte order mark (BOM)"),
        ("fl_skip_rows", "skip-rows-before-headers.csv", f"{frictionless_base}/skip-rows-before-headers.csv", "utf-8", ",", 1, "Frictionless Data", "MIT", "Comment preamble row before header"),
        ("fl_tsv", "table.tsv", f"{frictionless_base}/table.tsv", "utf-8", "\t", 0, "Frictionless Data", "MIT", "Tab-delimited file with escaped null markers"),
        ("fl_missing_sentinels", "table-infer-missing-values.csv", f"{frictionless_base}/table-infer-missing-values.csv", "utf-8", ",", 0, "Frictionless Data", "MIT", "Disguised missing values with hyphens '-'"),
        ("fl_doublequote", "doublequote.csv", f"{frictionless_base}/doublequote.csv", "utf-8", ",", 0, "Frictionless Data", "MIT", "Escaped double quotes inside text fields"),
        ("fl_unicode_headers", "table-unicode-headers.csv", f"{frictionless_base}/table-unicode-headers.csv", "utf-8", ",", 0, "Frictionless Data", "MIT", "Unicode characters in header names"),
        ("fl_accents", "table-with-accents.csv", f"{frictionless_base}/table-with-accents.csv", "utf-8", ",", 0, "Frictionless Data", "MIT", "Accented European characters in fields"),
        ("fl_delimiter_custom", "table-delimiter.csv", f"{frictionless_base}/table-delimiter.csv", "utf-8", ";", 0, "Frictionless Data", "MIT", "Alternative delimiter"),
        ("fl_blank_headers", "blank-headers.csv", f"{frictionless_base}/blank-headers.csv", "utf-8", ",", 0, "Frictionless Data", "MIT", "Missing header names in some columns"),
    ]
    for fid, fn, url, enc, sep, hdr, src, lic, note in fl_items:
        manifest.append({
            "file_id": fid, "filename": fn, "url": url, "encoding": enc,
            "delimiter": sep, "header_row_index": hdr, "source": src,
            "license": lic, "license_notes": note, "type": "direct"
        })

    # 2. pandas test fixtures (BSD-3-Clause)
    pd_base = "https://raw.githubusercontent.com/pandas-dev/pandas/main/pandas/tests/io/data/csv"
    pd_items = [
        ("pd_banklist", "banklist.csv", f"{pd_base}/banklist.csv", "utf-8", ",", 0, "pandas-dev/pandas", "BSD-3-Clause", "Real-world failed bank list with dates, states, cert IDs"),
        ("pd_test1", "test1.csv", f"{pd_base}/test1.csv", "utf-8", ",", 0, "pandas-dev/pandas", "BSD-3-Clause", "Multi-type test table with floats, integers, and text"),
    ]
    for fid, fn, url, enc, sep, hdr, src, lic, note in pd_items:
        manifest.append({
            "file_id": fid, "filename": fn, "url": url, "encoding": enc,
            "delimiter": sep, "header_row_index": hdr, "source": src,
            "license": lic, "license_notes": note, "type": "direct"
        })

    # 3. Seaborn benchmark datasets (BSD-3-Clause)
    sb_base = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master"
    sb_items = [
        ("sb_penguins", "penguins.csv", f"{sb_base}/penguins.csv", "utf-8", ",", 0, "Seaborn Datasets", "BSD-3-Clause", "Palmer penguins morphological measurements with missing values"),
        ("sb_titanic", "titanic.csv", f"{sb_base}/titanic.csv", "utf-8", ",", 0, "Seaborn Datasets", "BSD-3-Clause", "Titanic passenger manifest with missing ages, cabins, and text names"),
        ("sb_tips", "tips.csv", f"{sb_base}/tips.csv", "utf-8", ",", 0, "Seaborn Datasets", "BSD-3-Clause", "Restaurant tipping records with currency floats and factors"),
        ("sb_flights", "flights.csv", f"{sb_base}/flights.csv", "utf-8", ",", 0, "Seaborn Datasets", "BSD-3-Clause", "Monthly airline passengers with years, month text, passenger counts"),
        ("sb_planets", "planets.csv", f"{sb_base}/planets.csv", "utf-8", ",", 0, "Seaborn Datasets", "BSD-3-Clause", "Exoplanet discovery records with sparse orbital metrics"),
        ("sb_diamonds", "diamonds.csv", f"{sb_base}/diamonds.csv", "utf-8", ",", 0, "Seaborn Datasets", "BSD-3-Clause", "Diamond prices with cut, clarity, color factors and numeric specs"),
        ("sb_exercise", "exercise.csv", f"{sb_base}/exercise.csv", "utf-8", ",", 0, "Seaborn Datasets", "BSD-3-Clause", "Heart rate exercise trials with categorical regimes"),
        ("sb_car_crashes", "car_crashes.csv", f"{sb_base}/car_crashes.csv", "utf-8", ",", 0, "Seaborn Datasets", "BSD-3-Clause", "US state highway crash statistics with state IDs"),
        ("sb_geyser", "geyser.csv", f"{sb_base}/geyser.csv", "utf-8", ",", 0, "Seaborn Datasets", "BSD-3-Clause", "Old Faithful eruption wait times and durations"),
        ("sb_mpg", "mpg.csv", f"{sb_base}/mpg.csv", "utf-8", ",", 0, "Seaborn Datasets", "BSD-3-Clause", "Auto MPG dataset with vehicle names, cylinders, origins"),
        ("sb_taxis", "taxis.csv", f"{sb_base}/taxis.csv", "utf-8", ",", 0, "Seaborn Datasets", "BSD-3-Clause", "NYC taxi rides with pickup/dropoff datetimes, fares, tips"),
        ("sb_healthexp", "healthexp.csv", f"{sb_base}/healthexp.csv", "utf-8", ",", 0, "Seaborn Datasets", "BSD-3-Clause", "Cross-national health expenditure and life expectancy over time"),
    ]
    for fid, fn, url, enc, sep, hdr, src, lic, note in sb_items:
        manifest.append({
            "file_id": fid, "filename": fn, "url": url, "encoding": enc,
            "delimiter": sep, "header_row_index": hdr, "source": src,
            "license": lic, "license_notes": note, "type": "direct"
        })

    # 4. UCI Machine Learning Repository Datasets (CC BY 4.0)
    uci_items = [
        {
            "file_id": "uci_wine_red",
            "filename": "winequality-red.csv",
            "url": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
            "zip_file": "winequality-red.csv",
            "encoding": "utf-8", "delimiter": ";", "header_row_index": 0,
            "source": "UCI ML Repository", "license": "CC BY 4.0",
            "license_notes": "Semicolon-delimited physicochemical wine tests",
            "type": "zip"
        },
        {
            "file_id": "uci_wine_white",
            "filename": "winequality-white.csv",
            "url": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
            "zip_file": "winequality-white.csv",
            "encoding": "utf-8", "delimiter": ";", "header_row_index": 0,
            "source": "UCI ML Repository", "license": "CC BY 4.0",
            "license_notes": "Semicolon-delimited white wine quality dataset",
            "type": "zip"
        },
        {
            "file_id": "uci_bank_marketing",
            "filename": "bank.csv",
            "url": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
            "zip_file": "bank.zip",
            "sub_zip_file": "bank.csv",
            "encoding": "utf-8", "delimiter": ";", "header_row_index": 0,
            "source": "UCI ML Repository", "license": "CC BY 4.0",
            "license_notes": "Portuguese bank marketing direct calls with semicolon delimiters",
            "type": "nested_zip"
        },
        {
            "file_id": "uci_bank_marketing_full",
            "filename": "bank-full.csv",
            "url": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
            "zip_file": "bank.zip",
            "sub_zip_file": "bank-full.csv",
            "encoding": "utf-8", "delimiter": ";", "header_row_index": 0,
            "source": "UCI ML Repository", "license": "CC BY 4.0",
            "license_notes": "Full Portuguese bank marketing dataset with quotes and semicolons",
            "type": "nested_zip"
        },
        {
            "file_id": "uci_iris",
            "filename": "iris.data",
            "url": "https://archive.ics.uci.edu/static/public/53/iris.zip",
            "zip_file": "iris.data",
            "encoding": "utf-8", "delimiter": ",", "header_row_index": None,
            "column_names": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
            "source": "UCI ML Repository", "license": "CC BY 4.0",
            "license_notes": "Classic Fisher iris dataset, headerless comma-separated",
            "type": "zip"
        },
        {
            "file_id": "uci_adult",
            "filename": "adult.data",
            "url": "https://archive.ics.uci.edu/static/public/2/adult.zip",
            "zip_file": "adult.data",
            "encoding": "utf-8", "delimiter": ",", "header_row_index": None,
            "column_names": [
                "age", "workclass", "fnlwgt", "education", "education_num",
                "marital_status", "occupation", "relationship", "race", "sex",
                "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
            ],
            "source": "UCI ML Repository", "license": "CC BY 4.0",
            "license_notes": "Census income dataset with space-after-comma delimiter and '?' missing values",
            "type": "zip"
        },
        {
            "file_id": "uci_automobile",
            "filename": "imports-85.data",
            "url": "https://archive.ics.uci.edu/static/public/10/automobile.zip",
            "zip_file": "automobile/imports-85.data",
            "encoding": "utf-8", "delimiter": ",", "header_row_index": None,
            "column_names": [
                "symboling", "normalized_losses", "make", "fuel_type", "aspiration",
                "num_of_doors", "body_style", "drive_wheels", "engine_location",
                "wheel_base", "length", "width", "height", "curb_weight",
                "engine_type", "num_of_cylinders", "engine_size", "fuel_system",
                "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
                "city_mpg", "highway_mpg", "price"
            ],
            "source": "UCI ML Repository", "license": "CC BY 4.0",
            "license_notes": "1985 Auto imports with '?' missing values and headerless format",
            "type": "zip"
        },
    ]
    manifest.extend(uci_items)

    # 5. FiveThirtyEight Datasets (MIT License / CC BY 4.0)
    fte_base = "https://raw.githubusercontent.com/fivethirtyeight/data/master"
    fte_files = [
        ("airline-safety", "airline-safety.csv", "Airline incidents, fatal accidents and fatalities across eras"),
        ("alcohol-consumption", "drinks.csv", "Global beer, spirit, wine servings and pure alcohol liters"),
        ("bad-drivers", "bad-drivers.csv", "State driver fatality rates, insurance premiums, alcohol involvement"),
        ("bechdel", "movies.csv", "Bechdel test pass/fail results, IMDB IDs, budget, domestic/int gross with units"),
        ("candy-power-ranking", "candy-data.csv", "Halloween candy attributes with win percentages and sugar percentiles"),
        ("congress-demographics", "data_aging_congress.csv", "Historical age and demographics of US Congress members with bioguide IDs"),
        ("hate-crimes", "hate_crimes.csv", "Hate crime rates by state, median household income, Gini index, metro share"),
        ("bachelorette", "bachelorette.csv", "Bachelorette contestant records with elimination dates, dates, roses"),
        ("classic-rock", "classic-rock-raw-data.csv", "Classic rock radio song plays with artist, song title, release year"),
        ("police-killings", "police_killings.csv", "Demographics of individuals killed by police with race, age, county FIPS"),
        ("births", "US_births_1994-2003_CDC_NCHS.csv", "US births by date, month, day of week from CDC records"),
        ("births", "US_births_2000-2014_SSA.csv", "US births by date, year, month from SSA records"),
        ("star-wars-survey", "StarWars.csv", "Star wars fan survey responses with complex header and rating strings"),
        ("weather-check", "weather-check.csv", "Survey answers on weather-checking habits with demographic breakdowns"),
        ("biopics", "biopics.csv", "Biographical films dataset with box office gross, director, subject gender"),
        ("bob-ross", "elements-by-episode.csv", "Bob Ross paintings with binary visual elements present"),
        ("college-majors", "recent-grads.csv", "College majors earnings, employment rates, major codes, ranks"),
        ("college-majors", "grad-students.csv", "Graduate student salaries, unemployment rates, major categories"),
        ("college-majors", "majors-list.csv", "FOD1P major codes and major descriptions"),
        ("comic-characters", "marvel-wikia-data.csv", "Marvel comic character appearances, alignment, eye/hair color, dates"),
        ("comic-characters", "dc-wikia-data.csv", "DC comic character records, year introduced, alive status"),
        ("drug-use-by-age", "drug-use-by-age.csv", "Substance use frequency and percent by age bracket with disguised missing '-'"),
        ("food-world-cup", "food-world-cup-data.csv", "Global cuisine survey ratings and respondent demographics"),
        ("librarians", "librarians-by-msa.csv", "Librarian employment and salaries by metropolitan statistical area"),
        ("masculinity-survey", "raw-responses.csv", "Survey questions on modern masculinity with likert responses"),
        ("murder_2016", "murder_2015_final.csv", "City murder totals and percentage changes across US metropolitan areas"),
        ("murder_2016", "murder_2016_prelim.csv", "Preliminary city homicide counts and year-over-year comparisons"),
        ("nba-draft-2015", "historical_projections.csv", "Historical NBA draft player WAR projections and draft positions"),
        ("nba-elo", "nbaallelo.csv", "Historical NBA game outcomes from 1946 to 2015 with Elo ratings"),
        ("partisan-lean", "fivethirtyeight_partisan_lean_DISTRICTS.csv", "US Congressional district partisan lean indices"),
        ("partisan-lean", "fivethirtyeight_partisan_lean_STATES.csv", "US state partisan lean indices"),
        ("pulitzer", "pulitzer-circulation-data.csv", "Newspaper Pulitzer prizes and weekday/Sunday circulation changes"),
        ("riddler-castles", "castle-solutions.csv", "Crowdsourced game theory strategies for the Riddler castle problem"),
        ("trump-lawsuits", "trump-lawsuits.csv", "Civil litigation records involving Trump organizations"),
        ("unisex-names", "unisex_names_table.csv", "Gender-neutral given names with male/female share ratios"),
        ("us-weather-history", "KCLT.csv", "Daily weather observations in Charlotte, NC with precipitation and temps"),
        ("avengers", "avengers.csv", "Marvel Avengers comic book appearances, deaths, and returns"),
        ("fandango", "fandango_score_comparison.csv", "Movie review rating distributions from Fandango, Rotten Tomatoes, Metacritic"),
    ]
    for sub, fn, note in fte_files:
        fid = f"fte_{sub.replace('-', '_')}_{fn.replace('.csv', '').replace('-', '_').lower()}"
        url = f"{fte_base}/{sub}/{fn}"
        manifest.append({
            "file_id": fid, "filename": fn, "url": url, "encoding": "utf-8",
            "delimiter": ",", "header_row_index": 0, "source": "FiveThirtyEight",
            "license": "MIT / CC BY 4.0", "license_notes": note, "type": "direct"
        })

    # 6. TidyTuesday Datasets (MIT / CC0)
    tt_base = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data"
    tt_files = [
        ("2018/2018-04-23", "week4_australian_salary.csv", "Australian ATO taxable income and gender disparities"),
        ("2018/2018-05-14", "week7_starwars.csv", "Star wars fan demographics and movie rankings"),
        ("2018/2018-05-29", "week9_comic_characters.csv", "Comic book character records with alive status, hair, eyes"),
        ("2018/2018-06-12", "week11_fifa_audience.csv", "FIFA World Cup broadcast TV audience shares by territory"),
        ("2018/2018-10-30", "r-downloads.csv", "CRAN R daily binary download log events"),
        ("2018/2018-11-20", "thanksgiving_meals.csv", "Thanksgiving dinner food choices, pie types, US regions"),
        ("2019/2019-02-19", "phd_by_field.csv", "Doctorate degrees awarded in the US by academic field and year"),
        ("2019/2019-02-26", "full_trains.csv", "SNCF French high-speed railway delays and trip counts"),
        ("2019/2019-03-05", "jobs_gender.csv", "US census occupations, male/female wage ratios, total workers"),
        ("2019/2019-04-09", "grand_slams.csv", "Tennis Grand Slam championship winners, tournament dates, seeds"),
        ("2019/2019-04-30", "bird_call.csv", "Bird call audio acoustic feature measurements"),
        ("2019/2019-05-14", "nobel_winners.csv", "Nobel prize laureates, prize categories, birth dates, countries"),
        ("2019/2019-06-04", "ramen_ratings.csv", "Instant noodle reviews with star ratings, packaging styles, brands"),
        ("2019/2019-06-11", "meteorites.csv", "Meteorite landing sites, mass in grams, fall classification, lat/long"),
        ("2019/2019-06-18", "bird_counts.csv", "Christmas bird count census tallies across North American habitats"),
        ("2019/2019-06-25", "ufo_sightings.csv", "UFO sighting reports, encounter duration seconds, shapes, dates"),
        ("2019/2019-07-09", "wwc_outcomes.csv", "Women's World Cup soccer match outcomes and score margins"),
        ("2019/2019-07-16", "r4ds_members.csv", "R for Data Science online learning community member growth"),
        ("2019/2019-09-03", "cpu.csv", "Microprocessor transistor counts, process node nanometers, release dates"),
        ("2019/2019-09-17", "national_parks.csv", "US National Park Service visitor numbers over century of operation"),
        ("2019/2019-10-08", "ipf_lifts.csv", "International Powerlifting Federation competition lifts in kg"),
        ("2019/2019-10-15", "big_epa_cars.csv", "EPA fuel economy vehicle testing with fuel types, cylinders, transmission"),
        ("2019/2019-10-22", "horror_movies.csv", "Horror film release dates, budget, review scores, language, runtime"),
        ("2019/2019-10-29", "nyc_squirrels.csv", "Central Park squirrel census sightings, fur color, vocalizations"),
        ("2020/2020-01-07", "rainfall.csv", "Australian Bureau of Meteorology daily rainfall millimeters and stations"),
        ("2020/2020-01-14", "passwords.csv", "Common leaked password characteristics, crack time, strength ranks"),
        ("2020/2020-01-21", "spotify_songs.csv", "Spotify audio features: danceability, energy, tempo, loudness, key"),
        ("2020/2020-02-11", "hotels.csv", "Hotel booking cancellations, lead time, ADR currency, meal plans"),
        ("2020/2020-02-18", "food_consumption.csv", "Per-capita food consumption kg/person/year and CO2 emissions"),
        ("2020/2020-03-03", "season_goals.csv", "NHL hockey player career goals and assist statistics"),
        ("2020/2020-03-10", "tuition_cost.csv", "US college tuition costs, in-state/out-of-state rates, degree types"),
        ("2020/2020-04-07", "tdf_winners.csv", "Tour de France winners, stages, distances, finishing margins"),
        ("2020/2020-05-05", "villagers.csv", "Animal Crossing villagers personality, gender, species, birthdays"),
        ("2020/2020-05-12", "volcano.csv", "Volcano eruptions history, tectonic settings, evidence methods"),
    ]
    for sub, fn, note in tt_files:
        fid = f"tt_{sub.replace('/', '_').replace('-', '_')}_{fn.replace('.csv', '').lower()}"
        url = f"{tt_base}/{sub}/{fn}"
        manifest.append({
            "file_id": fid, "filename": fn, "url": url, "encoding": "utf-8",
            "delimiter": ",", "header_row_index": 0, "source": "TidyTuesday",
            "license": "MIT / CC0", "license_notes": note, "type": "direct"
        })

    # 7. Rdatasets CRAN Packages (GPL-2 / GPL-3 / CC0 / MIT)
    # Fetch datasets.csv index from Rdatasets to obtain verified active URLs
    r_index_url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/datasets.csv"
    try:
        r_index_bytes = fetch_url(r_index_url, timeout=15)
        reader = csv.DictReader(r_index_bytes.decode("utf-8").splitlines())
        r_all = list(reader)
    except Exception as e:
        print(f"Warning: Could not fetch Rdatasets index: {e}")
        r_all = []

    target_packages = [
        "carData", "MASS", "AER", "openintro", "Stat2Data", "Ecdat",
        "datasets", "wooldridge", "COUNT", "DAAG", "ISLR", "survival", "psych"
    ]
    seen_r_ids = set()

    # Select candidates from target packages
    for r in r_all:
        pkg = r.get("Package", "")
        item = r.get("Item", "")
        csv_url = r.get("CSV", "")
        rows_str = r.get("Rows", "0")
        cols_str = r.get("Cols", "0")
        title = r.get("Title", "CRAN dataset")

        if pkg in target_packages and csv_url and rows_str.isdigit() and cols_str.isdigit():
            rows = int(rows_str)
            cols = int(cols_str)
            if 15 <= rows <= 5000 and 3 <= cols <= 30:
                fid = f"r_{pkg.lower()}_{item.lower()}"
                if fid not in seen_r_ids:
                    seen_r_ids.add(fid)
                    manifest.append({
                        "file_id": fid,
                        "filename": f"{pkg}_{item}.csv",
                        "url": csv_url,
                        "encoding": "utf-8",
                        "delimiter": ",",
                        "header_row_index": 0,
                        "source": f"Rdatasets ({pkg})",
                        "license": "CRAN Open Source (GPL-2 / GPL-3 / MIT / CC0)",
                        "license_notes": title,
                        "type": "direct"
                    })
                    # Limit to ~150 total Rdatasets to keep corpus well-balanced
                    if len(seen_r_ids) >= 150:
                        break

    return manifest


# -------------------------------------------------------------
# Classification & Anomaly Detection Logic
# -------------------------------------------------------------

DISGUISED_MISSING_MARKERS = {
    "?", "NA", "N/A", "null", "NULL", "None", "none", "-", "--", ".",
    "unknown", "UNKNOWN", "missing", "MISSING", "999", "-999", "9999", "#N/A", "NaN"
}

UNIT_PATTERNS = [
    (r"^\$[\d,]+(?:\.\d+)?$", "$"),
    (r"^€[\d,]+(?:\.\d+)?$", "€"),
    (r"^£[\d,]+(?:\.\d+)?$", "£"),
    (r"^[\d,]+(?:\.\d+)?%$", "%"),
    (r"^[\d,]+(?:\.\d+)?\s*(?:kg|g|mg)$", "kg"),
    (r"^[\d,]+(?:\.\d+)?\s*(?:km|m|cm|mm)$", "km"),
    (r"^[\d,]+(?:\.\d+)?\s*(?:mph|kph)$", "mph"),
    (r"^[\d,]+(?:\.\d+)?\s*(?:years|yrs)$", "years"),
]

DATE_PATTERNS = [
    r"^\d{4}-\d{2}-\d{2}$",
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}",
    r"^\d{1,2}/\d{1,2}/\d{4}$",
    r"^\d{1,2}-\d{1,2}-\d{4}$",
    r"^\d{4}/\d{1,2}/\d{1,2}$",
    r"^[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}$",
    r"^\d{4}-\d{2}$",
]

ID_NAME_PATTERNS = ["id", "_id", "id_", "pk", "key", "index", "code", "fips", "zip", "postal", "rownames"]


def analyze_column(
    col_name: str,
    col_idx: int,
    raw_values: List[str],
    total_rows: int
) -> Tuple[str, List[str]]:
    """Inspect raw column values and determine ground truth semantic_type and anomalies."""
    anomalies: List[str] = []
    name_lower = col_name.strip().lower()

    # Detect disguised missing values
    non_null_values = []
    disguised_found = set()
    for v in raw_values:
        v_stripped = v.strip()
        if not v_stripped:
            continue
        if v_stripped in DISGUISED_MISSING_MARKERS:
            disguised_found.add(v_stripped)
        else:
            non_null_values.append(v_stripped)

    for marker in sorted(disguised_found):
        anomalies.append(f"disguised_missing: {marker}")

    # If completely empty or single unique non-null value -> constant
    unique_vals = set(non_null_values)
    if len(unique_vals) <= 1:
        return "constant", anomalies

    # Check for embedded units
    unit_counts = {}
    for v in non_null_values[:200]:
        for pat, unit_sym in UNIT_PATTERNS:
            if re.match(pat, v, re.IGNORECASE):
                unit_counts[unit_sym] = unit_counts.get(unit_sym, 0) + 1
    for u, count in unit_counts.items():
        if count >= max(2, int(0.3 * len(non_null_values[:200]))):
            anomalies.append(f"embedded_unit: {u}")

    # Check for leading zero numeric strings (e.g. "0123", "00123")
    leading_zero_count = sum(1 for v in non_null_values if len(v) > 1 and v.isdigit() and v.startswith("0"))
    if leading_zero_count >= max(2, int(0.05 * len(non_null_values))):
        anomalies.append("leading_zero")

    # Check for boolean
    unique_lower = {v.lower() for v in unique_vals}
    if len(unique_lower) == 2:
        if unique_lower in [{"true", "false"}, {"t", "f"}, {"yes", "no"}, {"y", "n"}, {"1", "0"}]:
            return "boolean", anomalies

    # Check for ID
    is_id_name = any(re.search(rf"\b{p}\b", name_lower) or name_lower == p or name_lower.endswith(p) for p in ID_NAME_PATTERNS)
    is_unique = (len(unique_vals) / len(non_null_values)) >= 0.95 if non_null_values else False
    if is_id_name and (is_unique or "leading_zero" in anomalies or "code" in name_lower or "fips" in name_lower or "zip" in name_lower):
        return "ID", anomalies

    if is_unique and len(non_null_values) >= 10:
        try:
            ints = [int(v) for v in non_null_values[:100]]
            if sorted(ints) == ints and len(set(ints)) == len(ints):
                return "ID", anomalies
        except ValueError:
            pass

    # Check for Datetime
    date_matches = 0
    sample_size = min(100, len(non_null_values))
    sample = non_null_values[:sample_size]
    for v in sample:
        if any(re.match(p, v) for p in DATE_PATTERNS):
            date_matches += 1
    if date_matches / sample_size >= 0.8:
        has_slash_dates = any(re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", v) for v in sample)
        if has_slash_dates:
            anomalies.append("ambiguous_date_format")
        return "datetime", anomalies

    # Check for Numeric
    clean_numeric_count = 0
    numeric_count = 0
    for v in sample:
        cleaned = re.sub(r"[^\d.-]", "", v)
        try:
            float(cleaned)
            numeric_count += 1
            float(v)
            clean_numeric_count += 1
        except ValueError:
            pass

    if clean_numeric_count / sample_size >= 0.85:
        if any(pat in name_lower for pat in ["zip", "postal", "fips", "bioguide"]):
            return "ID", anomalies
        return "numeric", anomalies

    if numeric_count / sample_size >= 0.85 and any(a.startswith("embedded_unit") for a in anomalies):
        return "numeric", anomalies

    # Check for Text vs Categorical
    avg_len = sum(len(v) for v in sample) / sample_size if sample else 0
    avg_words = sum(len(v.split()) for v in sample) / sample_size if sample else 0

    if avg_len > 40 or (avg_len > 25 and avg_words >= 5) or any(term in name_lower for term in ["description", "comment", "review", "summary", "text", "abstract", "question"]):
        return "text", anomalies

    return "categorical", anomalies


# -------------------------------------------------------------
# Dataset Ingestion & Processing
# -------------------------------------------------------------

def process_dataset(spec: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Downloads or uses cached dataset, parses rows and produces label record."""
    fid = spec["file_id"]
    fn = spec["filename"]
    raw_path = RAW_DIR / fn
    encoding = spec["encoding"]
    delimiter = spec["delimiter"]
    header_idx = spec["header_row_index"]

    print(f"Processing {fid} ({fn})...", end=" ", flush=True)

    # Use cached raw file if already exists and non-empty
    raw_bytes = None
    if raw_path.exists() and raw_path.stat().st_size > 0:
        with open(raw_path, "rb") as f:
            raw_bytes = f.read()
    else:
        try:
            if spec["type"] == "direct":
                raw_bytes = fetch_url(spec["url"])
            elif spec["type"] == "zip":
                zip_bytes = fetch_url(spec["url"])
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                    raw_bytes = z.read(spec["zip_file"])
            elif spec["type"] == "nested_zip":
                outer_zip_bytes = fetch_url(spec["url"])
                with zipfile.ZipFile(io.BytesIO(outer_zip_bytes)) as outer_z:
                    inner_zip_bytes = outer_z.read(spec["zip_file"])
                with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_z:
                    raw_bytes = inner_z.read(spec["sub_zip_file"])
            else:
                raise ValueError(f"Unknown type {spec['type']}")
        except Exception as e:
            print(f"FAILED to fetch: {e}")
            return False, f"Download error: {e}", None

        with open(raw_path, "wb") as f:
            f.write(raw_bytes)

    # Decode text
    try:
        text = raw_bytes.decode(encoding)
    except UnicodeDecodeError:
        encoding = "iso-8859-1"
        text = raw_bytes.decode(encoding)

    lines = text.splitlines()
    if not lines:
        print("FAILED: empty file")
        return False, "File is empty", None

    content_lines = lines
    if header_idx is not None and header_idx > 0:
        content_lines = lines[header_idx:]

    try:
        reader = csv.reader(content_lines, delimiter=delimiter)
        parsed_rows = list(reader)
    except Exception as e:
        print(f"FAILED to parse CSV: {e}")
        return False, f"CSV parsing error: {e}", None

    if not parsed_rows:
        print("FAILED: no rows parsed")
        return False, "No rows parsed", None

    if header_idx is not None:
        header_row = parsed_rows[0]
        data_rows = parsed_rows[1:]
        col_names = [h.strip() if h.strip() else f"unnamed_{i}" for i, h in enumerate(header_row)]
    else:
        data_rows = parsed_rows
        first_row = parsed_rows[0] if parsed_rows else []
        col_count = len(first_row)
        if "column_names" in spec and len(spec["column_names"]) == col_count:
            col_names = spec["column_names"]
        else:
            col_names = [f"col_{i}" for i in range(col_count)]

    col_values: Dict[int, List[str]] = {i: [] for i in range(len(col_names))}
    for row in data_rows:
        for i in range(len(col_names)):
            if i < len(row):
                col_values[i].append(row[i])
            else:
                col_values[i].append("")

    columns_label = []
    total_data_rows = len(data_rows)
    for i, name in enumerate(col_names):
        vals = col_values[i]
        sem_type, anomalies = analyze_column(name, i, vals, total_data_rows)
        columns_label.append({
            "column_name": name,
            "column_index": i,
            "semantic_type": sem_type,
            "anomalies": anomalies
        })

    label_record = {
        "file_id": fid,
        "filename": fn,
        "dialect": {
            "delimiter": delimiter,
            "encoding": encoding,
            "header_row_index": header_idx
        },
        "columns": columns_label
    }

    print(f"OK ({len(columns_label)} cols, {total_data_rows} rows)")
    return True, None, label_record


# -------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------

def main():
    manifest = build_dataset_manifest()
    print(f"Total candidates in manifest: {len(manifest)}")

    sources_rows = []
    successful_labels = []
    failed_datasets = []
    total_columns_labeled = 0

    for spec in manifest:
        ok, err, label_rec = process_dataset(spec)
        if ok and label_rec:
            fid = spec["file_id"]
            label_file = LABELS_DIR / f"{fid}.json"
            with open(label_file, "w", encoding="utf-8") as f:
                json.dump(label_rec, f, indent=2)

            successful_labels.append(label_rec)
            total_columns_labeled += len(label_rec["columns"])

            sources_rows.append({
                "file_id": spec["file_id"],
                "filename": spec["filename"],
                "source_name": spec["source"],
                "source_url": spec["url"],
                "license": spec["license"],
                "license_notes": spec["license_notes"]
            })
        else:
            failed_datasets.append((spec["file_id"], err))

    # Write sources.csv
    with open(SOURCES_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["file_id", "filename", "source_name", "source_url", "license", "license_notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sources_rows:
            writer.writerow(row)

    print("\n" + "=" * 60)
    print("CORPUS ACQUISITION & LABELING SUMMARY")
    print("=" * 60)
    print(f"Successfully processed and labeled files: {len(successful_labels)}")
    print(f"Total labeled columns: {total_columns_labeled}")
    print(f"Failed datasets: {len(failed_datasets)}")
    print(f"Manifest written to: {SOURCES_FILE}")
    print("=" * 60)

    if len(successful_labels) < 200:
        print(f"WARNING: Short of 200 files target by {200 - len(successful_labels)} files!")
        sys.exit(1)
    if total_columns_labeled < 1000:
        print(f"WARNING: Short of 1000 columns target by {1000 - total_columns_labeled} columns!")
        sys.exit(1)

    print("SUCCESS: Target requirements met (>=200 files, >=1000 columns)!")


if __name__ == "__main__":
    main()
