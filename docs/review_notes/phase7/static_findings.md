# Static findings

## Dependencies (name [group] ver | licence | spec | flags)
- charset-normalizer [runtime] 3.5.1 | MIT | charset-normalizer | unpinned
- hypothesis [dev] 6.170.8 | MPL-2.0 | hypothesis | unpinned
- ipywidgets [notebook] 8.1.9 | BSD 3-Clause License | ipywidgets>=8 | lower-bound-only
- joblib [runtime] 1.6.0 | BSD-3-Clause | joblib | unpinned
- matplotlib [runtime] 3.11.2 | Python Software Foundation License | matplotlib | unpinned
- mypy [dev] 2.4.0 | MIT | mypy | unpinned
- nox [dev] 2026.10.20 | Apache-2.0 | nox | unpinned
- numpy [runtime] 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | numpy | unpinned
- pandas [runtime] 3.0.6 | BSD License | pandas | unpinned
- pre-commit [dev] 4.7.0 | MIT | pre-commit | unpinned
- pypdf [dev] 6.21.1 | BSD-3-Clause | pypdf>=5.0.0 | lower-bound-only
- pytest [dev] 9.1.1 | MIT | pytest | unpinned
- pytest-cov [dev] 7.1.0 | MIT | pytest-cov | unpinned
- python-dateutil [runtime] 2.9.0.post0 | Dual License | python-dateutil | NON-PERMISSIVE/UNKNOWN,unpinned
- reportlab [runtime] 5.0.1 | BSD License | reportlab>=4.0.0 | lower-bound-only
- rich [interactive] 15.0.0 | MIT | rich>=13 | lower-bound-only
- ruff [dev] 0.17.2 | MIT | ruff | unpinned
- scikit-learn [runtime] 1.9.1 | BSD-3-Clause | scikit-learn | unpinned
- scipy [runtime] 1.18.1 | BSD License | scipy | unpinned
- typer [runtime] 0.27.2 | MIT | typer | unpinned

## broad_except_no_reraise (7)
- src/preflight/engineer/features.py:49
- src/preflight/ingest/dialect.py:95
- src/preflight/interact/channels/terminal.py:31
- src/preflight/interact/modes.py:50
- src/preflight/interact/preview.py:75
- src/preflight/interact/resolve.py:46
- src/preflight/plan.py:627

## dep_declared_unused_in_src (1)
- typer

## dep_nonpermissive (1)
- python-dateutil: Dual License

## dep_undeclared_import (1)
- IPython used in src/preflight/interact/channels/notebook.py

## dep_unpinned_runtime (9)
- charset-normalizer
- joblib
- matplotlib
- numpy
- pandas
- python-dateutil
- scikit-learn
- scipy
- typer

## except_Exception_pass (3)
- src/preflight/ingest/encoding.py:165
- src/preflight/ingest/loader.py:149
- src/preflight/report/report.py:382

## export_undocumented (9)
- preflight.Decision not in README/GUIDE
- preflight.Plan not in README/GUIDE
- preflight.TableSchema not in README/GUIDE
- preflight.apply not in README/GUIDE
- preflight.dry_run not in README/GUIDE
- preflight.plan not in README/GUIDE
- preflight.prepare not in README/GUIDE
- preflight.resolve not in README/GUIDE
- preflight.__version__ not in README/GUIDE

## fixed_temp_paths (1)
- tests/corpus/test_plan_corpus.py:134: with open("/tmp/preflight_corpus_plan_metrics.json", "w", encoding="utf-8") as f:

## label_modified_after_creation (1)
- corpus/labels/GUIDE.md <- @182903e 2026-10-23 feat: report pdf and test hardning

## magic_number_compare (37)
- src/preflight/infer/missing.py:153: len(parsed_nums) >= 3
- src/preflight/infer/missing.py:198: numeric_ratio >= 0.8
- src/preflight/infer/missing.py:198: len(remaining_vals) >= 10
- src/preflight/infer/missing.py:161: len(other_nums) >= 3
- src/preflight/infer/missing.py:202: len(v_str) <= 10
- src/preflight/infer/missing.py:176: max_other <= 150
- src/preflight/infer/types.py:383: dt_score >= 0.7
- src/preflight/infer/types.py:394: int_score >= 0.95
- src/preflight/infer/types.py:394: uniqueness_ratio >= 0.95
- src/preflight/infer/types.py:394: total_non_missing >= 5
- src/preflight/infer/types.py:210: p1 > 12
- src/preflight/infer/types.py:211: p2 > 12
- src/preflight/infer/types.py:332: len(failed_int_samples) < 5
- src/preflight/infer/types.py:352: len(failed_float_samples) < 5
- src/preflight/infer/types.py:378: len(failed_dt_samples) < 5
- src/preflight/infer/types.py:404: uniqueness_ratio >= 0.8
- src/preflight/infer/types.py:434: avg_len > 40
- src/preflight/infer/types.py:405: uniqueness_ratio >= 0.85
- src/preflight/infer/types.py:434: avg_len > 25
- src/preflight/infer/types.py:434: avg_words >= 5
- src/preflight/infer/types.py:434: avg_words >= 3
- src/preflight/infer/types.py:532: top_score < 0.6
- src/preflight/infer/types.py:608: margin < 0.3
- src/preflight/infer/types.py:549: confidence < 0.9
- src/preflight/infer/types.py:549: margin < 0.25
- src/preflight/clean/cleaner.py:100: success_count / len(non_empty) >= 0.95
- src/preflight/clean/cleaner.py:104: most_common_ratio < 0.9
- src/preflight/clean/cleaner.py:104: len(suffix_counts) > 5
- src/preflight/engineer/datetime.py:36: series.dt.dayofweek >= 5
- src/preflight/engineer/datetime.py:76: len(dt_columns) >= 2
- src/preflight/engineer/datetime.py:67: series.dt.dayofweek >= 5
- src/preflight/engineer/engineer.py:290: len(top) >= 2
- src/preflight/engineer/engineer.py:317: len(work) >= 2
- src/preflight/engineer/features.py:135: len(X) < 2
- src/preflight/engineer/vif.py:35: len(numeric_columns) > 50
- src/preflight/engineer/vif.py:41: len(df_clean) < 2
- src/preflight/engineer/vif.py:41: len(cols_to_compute) < 2

## noqa (5)
- src/preflight/clean/impute.py:12: from preflight.clean.markers import apply_missing_markers_to_series  # noqa: F401  (re-export)
- src/preflight/interact/channels/terminal.py:30: import rich  # noqa: F401
- src/preflight/interact/resolve.py:60: import ipywidgets  # noqa: F401
- tests/corpus/test_corruption_handling.py:27: from corruption import all_kinds, generate_corruption  # noqa: E402
- tests/corpus/test_degenerate_inputs.py:21: import pytest_timeout  # noqa: F401

## nosec (1)
- scripts/static_triage.py:49: line_grep("nosec", r"nosec", all_py)

## old_names (14)
- src/preflight/clean/cleaner.py:139: still hold text with units or stray spaces (v1: Profiler/Cleaner
- src/preflight/engineer/engineer.py:5: kinds come from the column signals (schema type plus the numbers v1's Profiler
- src/preflight/engineer/engineer.py:207: Profiler output the same way); when omitted they are computed from `df`.
- src/preflight/report/report.py:74: valid_stages = {"profiler", "cleaner", "engineer"}
- src/preflight/report/report.py:107: stages = ["profiler", "cleaner", "engineer"]
- src/preflight/report/report.py:281: stages = ["profiler", "cleaner", "engineer"]
- src/preflight/report/report.py:331: html_content = self.to_html(include_plots=include_plots)
- src/preflight/report/report.py:366: stages = ["profiler", "cleaner", "engineer"]
- src/preflight/report/report.py:368: "profiler": ["missingness", "class_distribution"],
- src/preflight/signals.py:5: need numbers v1's Profiler computed (missing rate, cardinality, rare values,
- src/preflight/signals.py:7: `profiler.py`; nothing else from the Profiler is ported. The column kind comes
- src/preflight/signals.py:68: # Ported verbatim from v1 profiler.py
- src/preflight/types.py:59: valid_stages = {"profiler", "cleaner", "clean", "engineer"}
- src/preflight/validation.py:143: ``ReportEntry`` with ``stage="profiler", severity="warning"``.

## skip_xfail (7)
- tests/corpus/test_interact_corpus.py:41: @pytest.mark.skipif(not RAW.is_dir(), reason="corpus/raw not present")
- tests/corpus/test_pipeline_corpus.py:147: @pytest.mark.skipif(not RAW_DIR.is_dir(), reason="corpus/raw not present")
- tests/regression/test_mvp_acceptance.py:503: @pytest.mark.skip(reason="v2 has no CLI in this phase (no CLI commands are in scope for M5); see docs/port_reg
- tests/regression/test_mvp_acceptance.py:526: @pytest.mark.skip(reason="v2 has no CLI in this phase (no CLI commands are in scope for M5); see docs/port_reg
- tests/unit/test_channels.py:94: pytest.importorskip("rich")
- tests/unit/test_channels.py:103: pytest.importorskip("rich")
- tests/unit/test_channels.py:339: pytest.importorskip("ipywidgets")

## src_changed_after_infer_report (27)
- src/preflight/clean/_common.py <- @ad04702 2026-10-23 feat: porting cleaner, engineer and assembly from v1.0.1
- src/preflight/clean/cleaner.py <- @ad04702 2026-10-23 feat: porting cleaner, engineer and assembly from v1.0.1
- src/preflight/clean/config.py <- @ad04702 2026-10-23 feat: porting cleaner, engineer and assembly from v1.0.1
- src/preflight/clean/dedupe.py <- @ad04702 2026-10-23 feat: porting cleaner, engineer and assembly from v1.0.1
- src/preflight/clean/drops.py <- @ad04702 2026-10-23 feat: porting cleaner, engineer and assembly from v1.0.1
- src/preflight/clean/impute.py <- @ad04702 2026-10-23 feat: porting cleaner, engineer and assembly from v1.0.1
- src/preflight/clean/markers.py <- @ad04702 2026-10-23 feat: porting cleaner, engineer and assembly from v1.0.1
- src/preflight/clean/normalize.py <- @ad04702 2026-10-23 feat: porting cleaner, engineer and assembly from v1.0.1
- src/preflight/clean/outliers.py <- @ad04702 2026-10-23 feat: porting cleaner, engineer and assembly from v1.0.1
- src/preflight/engineer/config.py <- @ad04702 2026-10-23 feat: porting cleaner, engineer and assembly from v1.0.1
- src/preflight/engineer/engineer.py <- @ad04702 2026-10-23 feat: porting cleaner, engineer and assembly from v1.0.1
- src/preflight/engineer/features.py <- @ad04702 2026-10-23 feat: porting cleaner, engineer and assembly from v1.0.1
- src/preflight/engineer/vif.py <- @ad04702 2026-10-23 feat: porting cleaner, engineer and assembly from v1.0.1
- src/preflight/clean/cleaner.py <- @0b51162 2026-10-22 feat: interaction layer implemented
- src/preflight/clean/config.py <- @0b51162 2026-10-22 feat: interaction layer implemented
- src/preflight/clean/dedupe.py <- @0b51162 2026-10-22 feat: interaction layer implemented
- src/preflight/clean/drops.py <- @0b51162 2026-10-22 feat: interaction layer implemented
- src/preflight/clean/impute.py <- @0b51162 2026-10-22 feat: interaction layer implemented
- src/preflight/clean/normalize.py <- @0b51162 2026-10-22 feat: interaction layer implemented
- src/preflight/clean/outliers.py <- @0b51162 2026-10-22 feat: interaction layer implemented
- src/preflight/engineer/config.py <- @0b51162 2026-10-22 feat: interaction layer implemented
- src/preflight/engineer/datetime.py <- @0b51162 2026-10-22 feat: interaction layer implemented
- src/preflight/engineer/encoding.py <- @0b51162 2026-10-22 feat: interaction layer implemented
- src/preflight/engineer/engineer.py <- @0b51162 2026-10-22 feat: interaction layer implemented
- src/preflight/engineer/features.py <- @0b51162 2026-10-22 feat: interaction layer implemented
- src/preflight/engineer/numeric.py <- @0b51162 2026-10-22 feat: interaction layer implemented
- src/preflight/engineer/vif.py <- @0b51162 2026-10-22 feat: interaction layer implemented

## swallowed_exception (5)
- src/preflight/engineer/vif.py:50: except np.linalg.LinAlgError
- src/preflight/infer/types.py:399: except ValueError
- src/preflight/ingest/encoding.py:139: except UnicodeDecodeError
- src/preflight/interact/resolve.py:63: except ImportError
- src/preflight/signals.py:168: except (ValueError, TypeError, OverflowError, ParserError)

## todo_fixme_xxx (1)
- scripts/static_triage.py:50: line_grep("todo_fixme_xxx", r"\b(TODO|FIXME|XXX)\b", all_py)

## unseeded_random (1)
- src/preflight/engineer/encoding.py:58: kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

## warnings_warn (1)
- src/preflight/interact/store.py:64: warnings.warn(

## writes_next_to_source (19)
- src/preflight/interact/channels/review_file.py:61: tmp = p.with_name(p.name + ".tmp")
- src/preflight/interact/channels/review_file.py:62: tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
- src/preflight/interact/resolve.py:51: return source.parent / "preflight_plan"
- src/preflight/interact/store.py:98: os.replace(self.path, self.path.with_name(STORE_FILENAME + ".bak"))
- src/preflight/interact/store.py:105: tmp = self.path.with_name(self.path.name + ".tmp")
- src/preflight/interact/store.py:106: tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
- src/preflight/plan.py:138: p.write_text(self.to_json(), encoding="utf-8")
- src/preflight/plan.py:1123: out_d = Path(out_dir).resolve() if out_dir is not None else p.parent / "preflight_plan"
- src/preflight/prepare.py:99: out_d = Path(out_dir).resolve() if out_dir is not None else p.parent / "preflight_out"
- src/preflight/report/report.py:317: fig.savefig(buf, format="png", bbox_inches="tight")
- src/preflight/report/report.py:332: with open(path, "w", encoding="utf-8") as f:
- src/preflight/report/report.py:362: pdf.savefig(fig)
- src/preflight/report/report.py:390: pdf.savefig(fig)
- src/preflight/report/report.py:395: pdf.savefig(s_fig)
- src/preflight/report/report.py:401: pdf.savefig(fig)
- src/preflight/report/report.py:442: pdf.savefig(fig)
- src/preflight/report/report_pdf.py:549: cleaned_df.to_csv(cleaned_path, index=False)
- src/preflight/report/report_pdf.py:554: schema_path.write_text(schema_json, encoding="utf-8")
- src/preflight/report/report_pdf.py:558: decisions_path.write_text(
