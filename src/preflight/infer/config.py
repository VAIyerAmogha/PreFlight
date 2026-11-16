"""
Named settings for type inference. Change them here, not in the code that uses them.
"""

# Columns with more non-missing values than this are scored on a seeded sample of this
# many values; the winning type is then checked against every value. Columns at or below
# this size are scored on every value, exactly as without sampling.
TYPE_SAMPLE_SIZE = 10_000

# Seed for the sample above, so the same file always gives the same result.
TYPE_SAMPLE_SEED = 20_241_005

# Embedded units are found generically: a number followed by a short alphabetic suffix
# (any letters up to UNIT_MAX_LETTERS long), or a currency symbol before / a percent sign after it.
# No unit names are listed anywhere.
UNIT_MAX_LETTERS = 5
# A unit must appear at least this many times in the column ("repeated")...
UNIT_MIN_REPEATS = 2
# ...and on at least this share of non-missing values to be reported as an embedded unit.
UNIT_FLAG_SHARE = 0.15
# When the most common unit is on at least this share of non-missing values, and EVERY
# non-missing value is a number with that unit (or a bare number), the column is read as
# numbers with the unit kept in the Decision (flagged as a lossy conversion). Otherwise a
# unit column is kept as text and flagged.
UNIT_COLUMN_SHARE = 0.80

# A value written as '+' followed by at least this many digits (for example an
# international phone number) is a code, not a number: the '+' would be lost, so the
# column is never cast to a number (handled like leading zeros).
PLUS_CODE_MIN_DIGITS = 7

# A column whose name suggests prose ('description', 'comment', 'title', 'name', ...) is
# read as free text when its values average at least this many words. Phase 7c: was 3,
# which made short multi-word labels such as person, place or event names free text;
# raised to 5 (tuning set: every column this rule decided at 3-4 words was labelled
# categorical). Long values are still read as text by the length rule.
TEXT_NAME_HINT_MIN_WORDS = 5
