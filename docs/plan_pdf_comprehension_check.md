# Plan PDF Comprehension Check — Study Protocol

## Purpose

PreFlight generates a "Plan PDF" (built by `src/preflight/report/plan_pdf.py`, using
plain-language sentence templates from `src/preflight/report/templates.py`) before it
touches a user's file. The PDF has five fixed sections, in this order:

1. **Summary box** — file name, row/column counts, target column, objective
   (classification/regression/none), a "Review Needed" count, and a fixed line
   "Source File Protected."
2. **Needs your decision** — zero or more boxed items, each a plain-language question
   (e.g. "Should '?' in column 'occupation' be treated as missing data?"), a list of
   checkbox-style options, and a highlighted "Recommended safe default" line. If there
   are none, the PDF instead says: "All automated checks completed with high certainty.
   No manual decisions are required. All operations will proceed automatically using
   safe defaults."
3. **What we will do** — five subsections, always all five, in this order: Reading,
   Fixing, Filling gaps, Removing, Creating. Each lists bullet summaries of what will
   happen (e.g. "Remove the repeated rows.", "Fill blanks with the most common value.")
   or, if nothing applies, the literal sentence "Nothing to do at this stage."
4. **What we won't touch** — two fixed sentences: a guarantee that the original source
   file is never changed/modified/overwritten (all outputs are new files), and a note
   that identifier/constant/preserved-text columns keep their original values.
5. **Sign-off checklist** — three checkbox lines ("I have reviewed the file reading
   settings...", "...the proposed column types...", "I confirm the selections above...")
   plus signature/date lines.

The document deliberately bans technical jargon (no "schema", "impute", "dtype",
"outlier", "median", etc. — see `BANNED_JARGON` in `templates.py`).

This protocol tests whether a non-technical reader, after looking at a real generated
Plan PDF, can correctly answer three questions a user actually needs answered before
signing off: *what will happen to my file*, *what do I need to decide*, and *what won't
change*. It does **not** test whether the PDF is well-designed or attractive — only
comprehension of its content.

This is a documentation-only protocol. No session is to be run as part of this task;
this file only describes how the study would be conducted if and when the team chooses
to run it.

---

## 1. Materials: which 3 Plan PDFs to generate and show

Generate one Plan PDF per file by running PreFlight's normal plan step against each of
these three corpus files, chosen for contrast in messiness:

| # | File | Why it's a good/varied example |
|---|------|--------------------------------|
| 1 | `corpus/raw/AER_CASchools.csv` | **Clean/simple baseline.** Already has a header row, comma-separated, consistent numeric columns, no obvious missing-data markers. Expect a short "Needs your decision" section (ideally the "no manual decisions required" message) and mostly "Nothing to do at this stage" entries. This is the control case: if readers can't understand even the simplest PDF, the language itself is the problem. |
| 2 | `corpus/raw/adult.data` | **Messy: no header, mixed types, missing-value marker.** No column-name row (PreFlight must infer/assign names), a leading-space-padded CSV-like format, and the classic `"?"` placeholder used for missing values in several columns (`workclass`, `occupation`, `native-country`). Expect "Needs your decision" items about the header row and about treating `"?"` as missing, plus Fixing/Filling-gaps entries. |
| 3 | `corpus/raw/AER_HMDA.csv` | **Messy: categorical-heavy, many yes/no and text columns, a target-like column (`deny`).** Columns such as `chist`, `mhist`, `phist` are encoded as small integers/yes-no codes that could plausibly be read multiple ways, and `deny` looks like a natural classification target. Expect a "Needs your decision" item about the prediction objective (classification vs. regression) and possibly a column-type ambiguity. |

If running the plan step on any of these three produces an unexpectedly empty "Needs
your decision" section for file 2 or 3, swap in another corpus file with a similar
messiness profile (e.g. another AER_* file with missing markers or a non-numeric target)
so that at least one of the three PDFs has 2+ decision items — the study needs variety
in both PDF content and reader experience.

Print or export each PDF; readers should view it the way a real user would (on screen or
on paper), not read the generation code or underlying CSV.

---

## 2. Recruiting / selecting readers

**Target: 5 readers** (5 is enough to surface a clear pattern of confusion; this is a
qualitative comprehension check, not a statistically powered study).

**"Non-technical" for this purpose means:** no professional or academic background in
data science, statistics, machine learning, or software engineering, and no prior
exposure to PreFlight or similar data-cleaning tools. A reader is non-technical for this
study even if they are a frequent spreadsheet user (e.g. uses Excel for budgets or
inventory) — the point is that they must have never worked with concepts like
"imputation," "encoding," "outliers," or "train/test splits" as part of their job or
studies. Good candidate pools: friends/family members, colleagues from non-technical
departments (sales, support, operations, admin), or a convenience sample via a generic
recruiting panel.

Exclude anyone who works in software engineering, data analysis/science, statistics, or
has a STEM degree with a quantitative research component — they are likely to
over-interpret jargon-adjacent phrasing the way an engineer would, which defeats the
purpose of testing plain language on a lay audience.

Aim for some variety in age/background across the 5, but this is not a hard requirement
given the small sample.

---

## 3. Session logistics

- **Length:** ~20–25 minutes total per reader (about 5–7 minutes per PDF: 2–3 minutes
  reading, 3–4 minutes answering the question sheet, plus a few minutes of intro/wrap-up).
  Run readers individually, not in a group, so answers aren't influenced by each other.
- **Setting:** in person or on a video call, screen-sharing or handing over a printed
  copy of the PDF. The facilitator should stay silent while the reader reads and
  answers — no coaching, no pointing at sections.
- **What to say beforehand (neutral framing — read or paraphrase closely):**

  > "I'm going to show you a document that a piece of software generated automatically
  > before it processes a data file someone owns. I'd like you to read it the way you
  > normally would if you'd received it about your own file, then answer some
  > questions about what it says. There are no wrong answers about *you* — I'm
  > testing whether the document itself is clear, not testing you. Please don't
  > guess based on outside knowledge of spreadsheets or software; just go by what's
  > written on the page. Take as long as you need to read it, and you can look back
  > at it while answering the questions."

  Do **not** say anything like "this tool cleans missing data" or "pay attention to
  the decisions section" beforehand — that would prime the answers and defeat the
  comprehension test.
- Repeat the process for all 3 PDFs with each reader, ideally varying the order the 3
  PDFs are shown across readers (e.g. reader 1 sees them in order 1-2-3, reader 2 in
  2-3-1, etc.) to avoid an order/fatigue effect.
- **Write-up:** after all 5 readers have gone through all 3 PDFs, the facilitator
  compiles results into a short results doc containing: (a) the scoring table (below),
  (b) per-PDF aggregate comprehension scores, (c) a list of the specific phrases or
  sections that caused repeated wrong answers (quote the reader's words), and (d) a
  recommendation of which PDF(s), if any, need language changes before the next design
  pass. Keep raw answer sheets attached as an appendix; don't discard them.

---

## 4. Question sheet (answered after each PDF, same sheet for all 3)

Give the reader a fresh copy of this sheet after each PDF, with the PDF still available
for them to refer back to. Record their written/spoken answers verbatim.

### Dimension A — "What will happen to my file?"

A1. (free text) *In your own words, what is this software going to do to your file?*

A2. (multiple choice) *Which of these best describes what happens to your original
file?*
   - [ ] It gets permanently edited in place
   - [ ] A new, separate file or output is created, and the original is left alone
   - [ ] It gets deleted after processing
   - [ ] The document doesn't say

A3. (multiple choice) *According to the document, will any rows or columns be removed
from the data entirely?*
   - [ ] Yes, at least one column or set of rows will be removed
   - [ ] No, nothing will be removed
   - [ ] The document doesn't say

### Dimension B — "What do I need to decide?"

B1. (free text) *Does this document ask you to make any choices or decisions? If yes,
list what they are about, in your own words.*

B2. (multiple choice) *How many separate things does the document want you to decide
or confirm?*
   - [ ] None — everything happens automatically
   - [ ] One
   - [ ] More than one
   - [ ] Not sure

B3. (for readers who said "one" or "more than one" in B2, free text, repeat per
decision item) *For [this decision], what does the document recommend you choose if
you're not sure, and why does it make that recommendation?*

B4. (multiple choice) *If you did nothing at all — didn't check any boxes or make any
choice — what would happen?*
   - [ ] The software would still proceed, using its recommended defaults
   - [ ] The software would refuse to continue
   - [ ] The document doesn't say

### Dimension C — "What will not change?"

C1. (free text) *Does the document promise anything will stay exactly the same / not be
touched? If yes, what?*

C2. (multiple choice) *According to the document, is your original source file ever
modified or overwritten?*
   - [ ] Yes
   - [ ] No
   - [ ] The document doesn't say

C3. (short free text) *Name one specific column, value, or kind of data (if any) that
the document says will be kept exactly as-is.*

---

## 5. Scoring rubric

For each of the 3 generated PDFs, before running sessions, the facilitator should write
down the **correct answer key** derived directly from that PDF's actual rendered content
(not from the source code) — e.g., for the `AER_CASchools.csv` PDF, A3's correct answer
is "No, nothing will be removed" only if the actual rendered "Removing" subsection says
"Nothing to do at this stage"; for `adult.data`, it may be "Yes" if a `drop_high_missingness`
or `drop_duplicates` decision appears. Build this answer key once per PDF, then reuse it
to score every reader who saw that PDF.

### Per-question scoring (apply to every question A1–C3)

- **Correct (2 points):** answer matches the PDF's actual content and captures the
  substance (for free-text questions, the key facts are present even if wording
  differs).
- **Partially correct (1 point):** answer is in the right direction but misses a key
  detail, over-generalizes, or includes some technical/incorrect embellishment not
  supported by the PDF (e.g., says "it deletes bad rows" when the PDF actually says
  the *default* is to remove them but it's still an open decision for the reader).
- **Incorrect (0 points):** answer contradicts the PDF, says "not sure"/"doesn't say"
  when the PDF does say, or invents content not in the PDF.

Multiple-choice questions are binary: 2 points for the correct option, 0 for any other
(no partial credit on MC items, since there's no room for nuance there).

### Aggregate comprehension score per PDF

For each PDF, per reader:
- Sum points across all 9 questions (A1–C3) → max 18 points per reader per PDF.
- Convert to a percentage: `score / 18 * 100`.

Then across the 5 readers for that PDF:
- **Aggregate comprehension score for the PDF** = mean percentage across the 5 readers.
- Also report the **per-dimension score** (A, B, C each out of 3 questions / 6 points)
  separately, since a PDF could be clear about "what will happen" but confusing about
  "what do I need to decide" — the aggregate alone would hide that.

### Interpreting the score — when is a PDF NOT clear enough?

- **Aggregate ≥ 85%:** comprehension is strong; no changes needed based on this round.
- **Aggregate 65–84%:** borderline; look at which specific questions or dimension
  dragged the score down and revise the language in that section only.
- **Aggregate < 65%, OR any single dimension (A, B, or C) scoring < 60% on its own even
  if the overall aggregate looks fine:** the PDF is **not clear enough** and needs a
  language revision before being used with real users. A dimension-specific failure
  (e.g., strong A and C but weak B) is just as disqualifying as a low overall score,
  since it points at a specific section (here, "Needs your decision") actively
  confusing readers rather than the document being evenly mediocre.
- Also flag as a failure, regardless of the numeric score, any case where **2 or more
  of the 5 readers** independently got the *same* question wrong in the *same* way —
  that's a signal of a specific confusing phrase or section, not reader variance, and
  should be called out by name/quote in the write-up even if the aggregate score is
  passing.
