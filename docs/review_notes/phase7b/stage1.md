# Stage 1: speed (plan(), one fresh process per file, sequential, nothing else running)

| file | before s | after s | fingerprint |
|---|---|---|---|
| ipf_lifts.csv (tt_2019_2019_10_08_ipf_lifts) | 16.13 | 2.48 | changed (sample confidences on large columns) |
| nbaallelo.csv (fte_nba_elo_nbaallelo) | 72.77 | 16.03 | changed (same reason) |
| ufo_sightings.csv (tt_2019_2019_06_25_ufo_sightings) | 24.34 | 6.11 | changed (same reason) |
| AER_Affairs.csv (small) | 0.21 | 0.05 | identical |

After-fingerprints were identical across two separate after-runs (determinism).
Per-column inference over all 254 corpus files (scripts/infer_fingerprints.py):
2841 columns identical, 0 differences among columns <= TYPE_SAMPLE_SIZE non-missing values,
16 large columns differ (0 type changes; confidence from the sample; 1 new anomaly:
spotify_songs.csv track_album_release_date `verification_failed: 1855 values are not datetime`).
Raw: .review/7b/timing_*.json, .review/7b/infer_fp_*.json, .review/7b/infer_fp_diff.txt

_ensure_prep_plan: already cached on PlanContext.prep_decisions; the profile's "7 calls"
were 1 computation + 6 cache hits (5.5 s cumulative is the one computation). Now asserted
by tests/unit/test_plan.py::test_cleaning_and_engineering_are_planned_once_per_run.
