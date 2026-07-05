# Changelog

## [1.0.0] - 2026-07-05
### Added
- TEXT semantic type + detection + stats (Phase 1).
- Opt-in text feature generation via FeatureConfig.text_features/text_tfidf (Phase 2).
- Manual SemanticType override via `column_types` (Phase 3).
- Config presets via `preset="fast"|"thorough"` (Phase 4).
- Dry run preview mode on `prepare()` (Phase 5).
- Graphic-first PDF export via `Report.save_pdf()` (Phase 6).
- Visual comparison PDF via `pf.save_compare_pdf()` (Phase 7).

### Changed
- CLI updated with flags for all new features above, consolidated and verified (Phase 8).
- Library-wide error messages simplified to plain language (Phase 8.1).

### Fixed
- Fixed outlier winsorization collapsing low-variance/zero-inflated columns by checking for zero IQR or standard deviation, and logged skips explicitly in the report.
- Fixed string-labeled classification targets (e.g. "Yes"/"No") causing target encoding crashes by converting targets internally to numeric, keeping it invisible in public reports.
- Added numeric coercion to automatically parse string/object columns containing numeric data and common unit suffixes (like CC, kmpl, Nm, bhp) into numeric features.
- Fixed overcorrection of numeric-string coercion for high-cardinality identifier-like columns (such as zip codes and customer IDs) by adding cardinality/integer checks, ensuring coerced columns retain their original name metadata for proper semantic type inference, and preventing skip-coercion on columns with physical unit suffixes.
- Fixed `save_compare_pdf` and `compare` raising `AttributeError` on missing datasets, dry-run results, or invalid/non-PrepResult inputs by introducing proper isinstance and structural validations.
- Resolved virtual environment subprocess path resolution and infinite recursion issues in the test suite.

## [0.2.0] - 2026-07-05
### Added
- Task/target mismatch validation: prepare()/profile()/clean()/engineer()
  now raise a clear error when task="classification" is used on a
  continuous-looking target, and warn (without raising) when
  task="regression" is used on a low-cardinality target.
- FeatureConfig: new opt-in feature engineering, off by default.
  - Numeric interaction/ratio/product/difference features between
    top-K target-correlated numeric columns.
  - Cyclical datetime features (sin/cos month, day-of-week), is_weekend,
    cross-column date deltas, days-since-reference.
  - Cluster-based features (cluster_label, distance-to-centroid) via KMeans,
    with automatic k selection.
- add_features(): apply FeatureConfig to an already-prepared PrepResult
  without rerunning Profiler/Cleaner from scratch.
- New CLI flags on `preflight prepare`: --interactions, --interaction-top-k,
  --interaction-types, --datetime-cyclical, --datetime-deltas,
  --datetime-reference-col, --clustering, --cluster-k, --cluster-features.
- Report readability improvements: consistent chart color palette,
  figure sizing that scales with column count, rotated/truncated axis
  labels, bar value annotations, and grouped/severity-coded .show() output
  with a new verbose parameter.

### Changed
- None (all additions are backward compatible; default behavior identical
  to v0.1.0 when FeatureConfig is not used).

### Fixed
- Confusing internal errors when task didn't match the target's actual
  data shape now surface as clear, actionable messages.

## [0.1.0] - 2026-07-04
- Initial release.
