"""
Corpus-wide validation suite for PreFlight v2.0 Plan, Dry-Run, and Plan PDF (Phase 4 / M3).

Executes dry_run across all 254 messy CSV files in corpus/raw/:
1. Validates 0 unhandled exceptions across the corpus.
2. Records and computes:
   - Total files evaluated
   - Files completed cleanly without unhandled exceptions
   - Total Plan PDF pages generated
   - Average and max PDF page count per file
   - needs_attention count per file
   - Median and max needs_attention per file
   - Number of files exceeding 10 needs_attention items
"""

import json
from pathlib import Path
import statistics
import tempfile
from typing import Any, Dict, List, Tuple
import pypdf
import pytest

from preflight.plan import dry_run

CORPUS_ROOT = Path(__file__).resolve().parent.parent.parent / "corpus"
RAW_DIR = CORPUS_ROOT / "raw"
LABELS_DIR = CORPUS_ROOT / "labels"


def get_all_raw_files() -> List[Path]:
    """Retrieve all raw files in corpus/raw/."""
    return sorted([f for f in RAW_DIR.iterdir() if f.is_file()])


class TestPlanCorpusValidation:
    """Corpus evaluation for dry_run and Plan PDF generation."""

    def test_corpus_dry_run_and_pdf_metrics(self):
        """
        Run dry_run on every corpus file and record all required metrics.
        """
        raw_files = get_all_raw_files()
        assert len(raw_files) == 254, f"Expected 254 corpus files, found {len(raw_files)}"

        total_files = len(raw_files)
        clean_completions = 0
        exceptions: List[Tuple[str, str, str]] = []

        page_counts: List[int] = []
        needs_attention_counts: List[int] = []
        files_exceeding_10_questions: List[Tuple[str, int]] = []

        file_metrics: List[Dict[str, Any]] = []

        with tempfile.TemporaryDirectory() as td:
            out_root = Path(td)

            for rf in raw_files:
                fn = rf.name
                file_out = out_root / fn
                file_out.mkdir(parents=True, exist_ok=True)

                try:
                    plan_obj, pdf_path = dry_run(rf, out_dir=file_out)
                    clean_completions += 1

                    # Count pages using pypdf
                    reader = pypdf.PdfReader(str(pdf_path))
                    num_pages = len(reader.pages)
                    page_counts.append(num_pages)

                    # Count needs_attention items
                    att_count = sum(1 for d in plan_obj.decisions if d.needs_attention)
                    needs_attention_counts.append(att_count)

                    if att_count > 10:
                        files_exceeding_10_questions.append((fn, att_count))

                    file_metrics.append(
                        {
                            "filename": fn,
                            "pages": num_pages,
                            "needs_attention": att_count,
                            "columns": plan_obj.column_count,
                            "rows": plan_obj.row_count,
                        }
                    )

                except Exception as exc:
                    exceptions.append((fn, type(exc).__name__, str(exc)))

        # Assert zero unhandled crashes
        assert len(exceptions) == 0, f"Encountered {len(exceptions)} unhandled exceptions: {exceptions}"
        assert clean_completions == 254, f"Expected 254 completions, got {clean_completions}"

        # Summary statistics
        median_att = statistics.median(needs_attention_counts)
        max_att = max(needs_attention_counts)
        mean_att = statistics.mean(needs_attention_counts)

        median_pages = statistics.median(page_counts)
        max_pages = max(page_counts)
        total_pages = sum(page_counts)

        print("\n================ Corpus Plan & PDF Validation Results ================")
        print(f"Total Corpus Files:                       {total_files}")
        print(f"Files Completed Without Exceptions:      {clean_completions} (100.0%)")
        print(f"Total Plan PDF Pages Generated:           {total_pages}")
        print(f"PDF Pages per File (median / max):        {median_pages:.1f} / {max_pages}")
        print(f"needs_attention per File (median / mean): {median_att:.1f} / {mean_att:.2f}")
        print(f"needs_attention per File (max):           {max_att}")
        print(f"Files Exceeding 10 Needs Attention Items: {len(files_exceeding_10_questions)}")
        if files_exceeding_10_questions:
            print("Files exceeding 10 questions:")
            for fn, count in files_exceeding_10_questions:
                print(f"  - {fn}: {count} questions")
        print("=======================================================================")

        # Save metrics to temporary json for documentation reporting
        metrics_summary = {
            "total_files": total_files,
            "clean_completions": clean_completions,
            "total_pages": total_pages,
            "median_pages": median_pages,
            "max_pages": max_pages,
            "median_needs_attention": median_att,
            "max_needs_attention": max_att,
            "mean_needs_attention": mean_att,
            "files_exceeding_10_count": len(files_exceeding_10_questions),
            "files_exceeding_10_list": files_exceeding_10_questions,
            "file_metrics": file_metrics,
        }
        with open("/tmp/preflight_corpus_plan_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
