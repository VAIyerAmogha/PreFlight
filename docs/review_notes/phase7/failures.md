# Failures (tracebacks tails, max 15 lines)

## accuracy-harness fte_nba_elo_nbaallelo [tuning] TIMEOUT: Timeout: TIMEOUT after 120s
```
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 420, in infer_column_type
    p_d = _parse_datetime(s_clean)
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 224, in _parse_datetime
    return any(re.match(p, s) for p in DATE_PATTERNS)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 224, in <genexpr>
    return any(re.match(p, s) for p in DATE_PATTERNS)
               ^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/re/__init__.py", line 167, in match
    return _compile(pattern, flags).match(string)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/scripts/accuracy_shard.py", line 32, in _alarm
    raise Timeout(f"TIMEOUT after {timeout_s}s")
Timeout: TIMEOUT after 120s
```
## accuracy-harness tt_2019_2019_10_08_ipf_lifts [tuning] TIMEOUT: Timeout: TIMEOUT after 120s
```
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 375, in infer_column_type
    if _parse_datetime(v):
       ^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 224, in _parse_datetime
    return any(re.match(p, s) for p in DATE_PATTERNS)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 224, in <genexpr>
    return any(re.match(p, s) for p in DATE_PATTERNS)
               ^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/re/__init__.py", line 167, in match
    return _compile(pattern, flags).match(string)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/scripts/accuracy_shard.py", line 32, in _alarm
    raise Timeout(f"TIMEOUT after {timeout_s}s")
Timeout: TIMEOUT after 120s
```
## accuracy-harness tt_2020_2020_02_11_hotels [tuning] TIMEOUT: Timeout: TIMEOUT after 120s
```
  File "/home/amg/Desktop/PreFlight/src/preflight/plan.py", line 401, in plan
    inferred_schema = infer_table_types(df, context.schema)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 653, in infer_table_types
    res = infer_column_type(col_series, column_name=str(col_name))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 346, in infer_column_type
    p_flt, is_dec_comma, unit_sym, flt_lz = _parse_float(v)
                                            ^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 166, in _parse_float
    if re.match(pat, s, re.IGNORECASE):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/scripts/accuracy_shard.py", line 32, in _alarm
    raise Timeout(f"TIMEOUT after {timeout_s}s")
Timeout: TIMEOUT after 120s
```
## accuracy-harness tt_2019_2019_10_15_big_epa_cars [tuning] TIMEOUT: Timeout: TIMEOUT after 120s
```
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 653, in infer_table_types
    res = infer_column_type(col_series, column_name=str(col_name))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 418, in infer_column_type
    p_f, _, _, _ = _parse_float(s_clean)
                   ^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 166, in _parse_float
    if re.match(pat, s, re.IGNORECASE):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/re/__init__.py", line 164, in match
    def match(pattern, string, flags=0):
    
  File "/home/amg/Desktop/PreFlight/scripts/accuracy_shard.py", line 32, in _alarm
    raise Timeout(f"TIMEOUT after {timeout_s}s")
Timeout: TIMEOUT after 120s
```
## accuracy-harness tt_2020_2020_01_21_spotify_songs [held_out] TIMEOUT: Timeout: TIMEOUT after 120s
```
  File "/home/amg/Desktop/PreFlight/src/preflight/plan.py", line 401, in plan
    inferred_schema = infer_table_types(df, context.schema)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 653, in infer_table_types
    res = infer_column_type(col_series, column_name=str(col_name))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 375, in infer_column_type
    if _parse_datetime(v):
       ^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 224, in _parse_datetime
    return any(re.match(p, s) for p in DATE_PATTERNS)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/scripts/accuracy_shard.py", line 32, in _alarm
    raise Timeout(f"TIMEOUT after {timeout_s}s")
Timeout: TIMEOUT after 120s
```
## accuracy-harness tt_2019_2019_06_25_ufo_sightings [tuning] TIMEOUT: Timeout: TIMEOUT after 120s
```
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 653, in infer_table_types
    res = infer_column_type(col_series, column_name=str(col_name))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 375, in infer_column_type
    if _parse_datetime(v):
       ^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 224, in _parse_datetime
    return any(re.match(p, s) for p in DATE_PATTERNS)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 224, in <genexpr>
    return any(re.match(p, s) for p in DATE_PATTERNS)
               ^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/scripts/accuracy_shard.py", line 32, in _alarm
    raise Timeout(f"TIMEOUT after {timeout_s}s")
Timeout: TIMEOUT after 120s
```
## accuracy-harness tt_2020_2020_01_07_rainfall [tuning] TIMEOUT: Timeout: TIMEOUT after 120s
```
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 653, in infer_table_types
    res = infer_column_type(col_series, column_name=str(col_name))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 346, in infer_column_type
    p_flt, is_dec_comma, unit_sym, flt_lz = _parse_float(v)
                                            ^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/src/preflight/infer/types.py", line 166, in _parse_float
    if re.match(pat, s, re.IGNORECASE):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/re/__init__.py", line 167, in match
    return _compile(pattern, flags).match(string)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/amg/Desktop/PreFlight/scripts/accuracy_shard.py", line 32, in _alarm
    raise Timeout(f"TIMEOUT after {timeout_s}s")
Timeout: TIMEOUT after 120s
```
## shard 2 pytest shard 2 produced no junit (exit 124; 124 = hit wall-clock cap): NO RESULT
```
```
## shard 3 pytest tests.corpus.test_interact_corpus::test_interact_over_corpus: AssertionError: files hung: ['big_epa_cars.csv', 'hotels.csv', 'nbaallelo.csv']
assert not ['big_epa_cars.csv', 'hotels.csv', 'nbaallelo.csv']
```
        }
        for r in done:
            for k, v in r["categories"].items():
                summary["categories"][k] = summary["categories"].get(k, 0) + v
    
        out = ROOT / ".pytest_cache" / "interact_corpus_results.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "files": results}, indent=1), encoding="utf-8")
        print("\nINTERACT CORPUS SUMMARY\n" + json.dumps(summary, indent=1))
    
>       assert not hung, f"files hung: {hung}"
E       AssertionError: files hung: ['big_epa_cars.csv', 'hotels.csv', 'nbaallelo.csv']
E       assert not ['big_epa_cars.csv', 'hotels.csv', 'nbaallelo.csv']

tests/corpus/test_interact_corpus.py:78: AssertionError
```
## shard 2b pytest shard 2b produced no junit (exit 124; 124 = hit wall-clock cap): NO RESULT
```
```
## fast-suite tests.regression.test_mvp_acceptance.TestRegression_StringAndMultiClassTargets::test_multiclass_string_target_does_not_crash: assert 149 == 150
```

    def test_multiclass_string_target_does_not_crash(self):
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame.copy()
        df["species"] = iris.target_names[iris.target]
        df = df.drop(columns=["target"])
    
        result = pf.prepare(df, target="species", task="classification")
        assert result.df is not None
        assert result.pipeline is not None
>       assert result.df.shape[0] == 150
E       assert 149 == 150

tests/regression/test_mvp_acceptance.py:80: AssertionError
```
## fast-suite tests.regression.test_mvp_acceptance.TestRegression_IdLikeColumnsNotCoercedToNumeric::test_phone_number_like_column_not_coerced: AssertionError: assert <SemanticType.NUMERIC_FEATURE: 'NUMERIC_FEATURE'> != <SemanticType.NUMERIC_FEATURE: 'NUMERIC_FEATURE'>
 +  where <SemanticType.NUMERIC_FEATURE: 'NUMERIC_FEATURE'> = ColumnProfil
```
    def test_phone_number_like_column_not_coerced(self):
        n = 200
        df = pd.DataFrame({
            "phone": [f"{np.random.randint(1000000000, 9999999999)}" for _ in range(n)],
            "age": np.random.randint(18, 80, size=n),
            "target": np.random.randint(0, 2, size=n),
        })
        profiles, _entries = pf.run_profiler(df, target="target", task="classification")
        by_name = {p.name: p for p in profiles}
>       assert by_name["phone"].semantic_type != SemanticType.NUMERIC_FEATURE
E       AssertionError: assert <SemanticType.NUMERIC_FEATURE: 'NUMERIC_FEATURE'> != <SemanticType.NUMERIC_FEATURE: 'NUMERIC_FEATURE'>
E        +  where <SemanticType.NUMERIC_FEATURE: 'NUMERIC_FEATURE'> = ColumnProfile(name='phone', semantic_type=<SemanticType.NUMERIC_FEATURE: 'NUMERIC_FEATURE'>, missing_rate=0.0, dtype=d...info_with_target=0.0, is_leakage_suspect=False, text_avg_length=None, text_avg_word_count=None, text_missing_rate=None).semantic_type
E        +  and   <SemanticType.NUMERIC_FEATURE: 'NUMERIC_FEATURE'> = SemanticType.NUMERIC_FEATURE

tests/regression/test_mvp_acceptance.py:146: AssertionError
```
## fast-suite tests.regression.test_mvp_acceptance.TestRegression_IdLikeColumnsNotCoercedToNumeric::test_genuinely_messy_numeric_string_still_coerces_correctly: AssertionError: assert <SemanticType.CATEGORICAL_HIGH: 'CATEGORICAL_HIGH'> == <SemanticType.NUMERIC_FEATURE: 'NUMERIC_FEATURE'>
 +  where <SemanticType.CATEGORICAL_HIGH: 'CATEGORICAL_HIGH'> = ColumnPr
```
        values = [str(round(np.random.uniform(20, 120), 2)) for _ in range(n - 5)]
        values += ["", " ", "N/A", "20.5 ", " 45.3"]  # a few messy/blank entries
        df = pd.DataFrame({
            "total_charges": values,
            "tenure": np.random.randint(1, 72, size=n),
            "target": np.random.randint(0, 2, size=n),
        })
        profiles, _entries = pf.run_profiler(df, target="target", task="classification")
        by_name = {p.name: p for p in profiles}
>       assert by_name["total_charges"].semantic_type == SemanticType.NUMERIC_FEATURE
E       AssertionError: assert <SemanticType.CATEGORICAL_HIGH: 'CATEGORICAL_HIGH'> == <SemanticType.NUMERIC_FEATURE: 'NUMERIC_FEATURE'>
E        +  where <SemanticType.CATEGORICAL_HIGH: 'CATEGORICAL_HIGH'> = ColumnProfile(name='total_charges', semantic_type=<SemanticType.CATEGORICAL_HIGH: 'CATEGORICAL_HIGH'>, missing_rate=0....0.002138401105921117, is_leakage_suspect=False, text_avg_length=None, text_avg_word_count=None, text_missing_rate=None).semantic_type
E        +  and   <SemanticType.NUMERIC_FEATURE: 'NUMERIC_FEATURE'> = SemanticType.NUMERIC_FEATURE

tests/regression/test_mvp_acceptance.py:162: AssertionError
```
## fast-suite tests.regression.test_mvp_acceptance.TestRegression_UnitSuffixNumericParsing::test_simple_unit_suffix_column_parses_as_numeric: AssertionError: assert <SemanticType.CATEGORICAL_HIGH: 'CATEGORICAL_HIGH'> == <SemanticType.NUMERIC_FEATURE: 'NUMERIC_FEATURE'>
 +  where <SemanticType.CATEGORICAL_HIGH: 'CATEGORICAL_HIGH'> = ColumnPr
```

    def test_simple_unit_suffix_column_parses_as_numeric(self):
        n = 200
        df = pd.DataFrame({
            "engine_cc": [f"{np.random.randint(800, 2500)} CC" for _ in range(n)],
            "target": np.random.randint(0, 2, size=n),
        })
        profiles, _entries = pf.run_profiler(df, target="target", task="classification")
        by_name = {p.name: p for p in profiles}
>       assert by_name["engine_cc"].semantic_type == SemanticType.NUMERIC_FEATURE
E       AssertionError: assert <SemanticType.CATEGORICAL_HIGH: 'CATEGORICAL_HIGH'> == <SemanticType.NUMERIC_FEATURE: 'NUMERIC_FEATURE'>
E        +  where <SemanticType.CATEGORICAL_HIGH: 'CATEGORICAL_HIGH'> = ColumnProfile(name='engine_cc', semantic_type=<SemanticType.CATEGORICAL_HIGH: 'CATEGORICAL_HIGH'>, missing_rate=0.0, d...info_with_target=0.0, is_leakage_suspect=False, text_avg_length=None, text_avg_word_count=None, text_missing_rate=None).semantic_type
E        +  and   <SemanticType.NUMERIC_FEATURE: 'NUMERIC_FEATURE'> = SemanticType.NUMERIC_FEATURE

tests/regression/test_mvp_acceptance.py:258: AssertionError
```
