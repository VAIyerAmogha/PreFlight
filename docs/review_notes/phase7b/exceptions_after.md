| category | location | except | justification |
|---|---|---|---|
| broad_except_no_reraise | src/preflight/engineer/features.py:49 | Exception | mutual information is an optional signal; None means "not available" and callers treat it as such |
| swallowed_exception | src/preflight/engineer/vif.py:50 | np.linalg.LinAlgError | a singular correlation matrix leaves every score None ("not computed"), as in v1 |
| swallowed_exception | src/preflight/infer/types.py:496 | ValueError | a value that is not a plain integer means the column is not a sequence |
| broad_except_no_reraise | src/preflight/ingest/dialect.py:95 | Exception | a candidate dialect that cannot parse the sample scores zero, which is the evidence against it |
| swallowed_exception | src/preflight/ingest/encoding.py:150 | UnicodeDecodeError | not valid UTF-8, so the next detector below decides |
| broad_except_no_reraise | src/preflight/interact/modes.py:53 | Exception | if IPython cannot report its shell, this is not a usable notebook kernel |
| broad_except_no_reraise | src/preflight/interact/resolve.py:54 | Exception | previews are optional; with None every preview says the values are unavailable |
| swallowed_exception | src/preflight/interact/resolve.py:69 | ImportError | no widgets installed, so fall back to the terminal prompt |
| broad_except_no_reraise | src/preflight/report/report.py:287 | Exception | reported as a warning; the PDF still renders without this plot |
| swallowed_exception | src/preflight/signals.py:168 | (ValueError, TypeError, OverflowError, ParserError) | counting parse failures is the measurement |
