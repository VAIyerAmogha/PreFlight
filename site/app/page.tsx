import fixtureUnits from "@/data/fixtures/disguised_missing_units.json";
import fixtureSemicolon from "@/data/fixtures/messy_noheader_semicolon.json";
import CopyButton from "@/components/CopyButton";
import AccuracyFigure from "@/components/figures/AccuracyFigure";
import HeroTerminal from "@/components/figures/HeroTerminal";
import HygieneFigure from "@/components/figures/HygieneFigure";
import PipelineStages from "@/components/figures/PipelineStages";
import QuestionsFigure from "@/components/figures/QuestionsFigure";
import ReadTwoWays from "@/components/figures/ReadTwoWays";
import DecisionExplorer from "@/components/DecisionExplorer";
import Link from "next/link";
import PageEffects, { EngineEyebrow } from "@/components/PageEffects";
import Topbar from "@/components/Topbar";
import TryTeaser from "@/components/figures/TryTeaser";
import { CodeBlock } from "@/components/ui";
import { PYPI, REPO } from "@/lib/config";
import type { Fixture } from "@/lib/types";

// Fixtures are real runs of public/py/runner.py (scripts/build_fixtures.py).
const units = fixtureUnits as unknown as Fixture;
const semicolon = fixtureSemicolon as unknown as Fixture;

const TOC = [
  ["why", "Why PreFlight"],
  ["how", "How it works"],
  ["decisions", "Everything is a Decision"],
  ["abstain", "Abstain, don’t guess"],
  ["hygiene", "Fit hygiene"],
  ["try", "Try it live"],
  ["quickstart", "Quickstart"],
  ["api", "API reference"],
  ["accuracy", "Accuracy"],
  ["limits", "Limitations"],
] as const;

const CODE = {
  install: `pip install --pre pypreflight
pip install --pre "pypreflight[interactive]"   # rich-formatted terminal questions
pip install --pre "pypreflight[notebook]"      # ipywidgets questions inside Jupyter`,
  one: `import preflight

out = preflight.prepare("patients.csv", target="outcome")`,
  plan: `p = preflight.plan("examples/disguised_missing_units.csv", target="outcome")

for d in p.decisions:
    flag = "ASK" if d.needs_attention else ""
    print(f"{d.stage:<9}{d.kind:<22}{str(d.columns):<15}{d.action:<26}{d.confidence:.2f} {flag}")`,
  callback: `def answer(questions):
    choices = {}
    for q in questions:
        print(q.column, q.text, q.options)
        choices[q.question_id] = q.recommended_default
    return choices

out = preflight.prepare("patients.csv", target="outcome", mode="ask", channel=answer)`,
  cv: `from sklearn.base import clone
from sklearn.model_selection import KFold

df = out["apply_result"].prepared
X, y = df.drop(columns=["outcome"]), df["outcome"]

for train, test in KFold(5, shuffle=True, random_state=0).split(X):
    pipe = clone(out["pipeline"])
    X_train = pipe.fit_transform(X.iloc[train], y.iloc[train])
    X_test = pipe.transform(X.iloc[test])
    # fit and score your model here`,
};

function Sig({ fn, params, ret }: { fn: string; params: [string, string?][]; ret?: string }) {
  return (
    <div className="sig">
      <span className="fn">{fn}</span>(
      {params.map(([p, d], i) => (
        <span key={p}>
          {i > 0 && ", "}
          <span className="p">{p}</span>
          {d !== undefined && <>=<span className="d">{d}</span></>}
        </span>
      ))}
      ){ret && <> → {ret}</>}
    </div>
  );
}

export default function Page() {
  const v = units.meta.preflight;
  return (
    <>
      <PageEffects />
      <Topbar page="docs" />

      <div className="layout" id="top">
        <aside className="toc" aria-label="On this page">
          <div className="toc-label">On this page</div>
          <ol>
            {TOC.map(([id, label], i) => (
              <li key={id}><a href={`#${id}`}><span>{String(i + 1).padStart(2, "0")}</span>{label}</a></li>
            ))}
          </ol>
          <div className="toc-meta">
            Every figure on this page is real <code>pypreflight</code> output.
            <br />
            Figures built with {v}.
          </div>
        </aside>

        <main className="prose">
          {/* ===================================================== HERO */}
          <section className="hero full" aria-labelledby="hero-title">
            <EngineEyebrow />
            <h1 id="hero-title">Model-ready data, with <em>every choice</em> written down.</h1>
            <p className="lede">
              PreFlight reads a messy CSV, works out what it means, and hands back a cleaned table, an unfitted scikit-learn pipeline and a report anyone can read.
              When the evidence is too close to call, it asks you instead of guessing.
            </p>
            <div className="byline">
              <span><b>pypreflight</b> · Python 3.10–3.12</span>
              <span>MIT licensed</span>
              <span>0 / 254 corpus files crash</span>
              <span>0.46% silent type errors</span>
            </div>
            <div className="cta-row">
              <Link className="btn primary" href="/try">
                <svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
                Try it on your CSV
              </Link>
              <a className="btn" href="#quickstart">Read the docs</a>
              <span className="install"><span className="prompt">$</span> pip install --pre pypreflight <CopyButton text="pip install --pre pypreflight" label="" /></span>
            </div>
            <figure className="hero-figure wide">
              <HeroTerminal decisions={units.analysis.decisions} file={units.sample} version={v} />
              <figcaption>
                <b>Figure 0.</b> <code>preflight.plan()</code> on a 60-row file with four traps: units in a numeric column, <code>-999</code> meaning “no reading”,{" "}
                <code>?</code> meaning unknown, and an ID that looks like a category. Each line is one <code>Decision</code>;{" "}
                <span className="badge ask">ASK</span> marks the ones that need a person. This is the package’s real output.
              </figcaption>
            </figure>
          </section>

          {/* ===================================================== 01 */}
          <section className="chapter" id="why" aria-labelledby="why-h">
            <span className="chapter-num">01</span>
            <h2 id="why-h">Why PreFlight</h2>
            <p className="lead-in"><code>pandas.read_csv(path)</code> assumes UTF-8, commas and a header on the first line. Real files break those assumptions all the time.</p>
            <p>
              European exports separated by semicolons. Latin-1 text. Headerless dumps. Title rows above the table. <code>-999</code> or <code>?</code> standing in for
              “missing”. <code>69 kg</code> in a numeric column. Most preprocessing tools either fail on these files or, worse, handle them silently in a way nobody checks.
            </p>
            <ReadTwoWays fx={semicolon} />
            <p>PreFlight is built on five rules. The rest of this page shows each one working on real files.</p>
            <div className="rules wide reveal">
              {[
                ["i", "Everything is a Decision", "Encoding, separator, header, every type, fill value and encoder is a record with evidence and confidence."],
                ["ii", "Plan before apply", "plan() only reads and proposes. Nothing changes until apply()."],
                ["iii", "Abstain, don’t guess", "If two readings score too closely, or a conversion would lose information, the column stays text and becomes a question."],
                ["iv", "Ask once, never hang", "Questions are batched into one review. With nobody to ask, the run stops with a clear list."],
                ["v", "Fit hygiene", "Steps that learn from data go into an unfitted pipeline you fit per fold, so CV scores stay honest."],
              ].map(([n, h, p]) => (
                <div className="rule" key={n}><span className="n">{n}</span><h5>{h}</h5><p>{p}</p></div>
              ))}
            </div>
          </section>

          {/* ===================================================== 02 */}
          <section className="chapter" id="how" aria-labelledby="how-h">
            <span className="chapter-num">02</span>
            <h2 id="how-h">How it works</h2>
            <p>A run moves through five stages. The first four only read and propose: your file is never modified, and nothing is written until the last stage.</p>
            <PipelineStages fx={units} />
          </section>

          {/* ===================================================== 03 */}
          <section className="chapter" id="decisions" aria-labelledby="dec-h">
            <span className="chapter-num">03</span>
            <h2 id="dec-h">Everything is a Decision</h2>
            <p>
              Every step produces <code>Decision</code> objects. Each carries the <b>evidence</b> behind it, a <b>confidence</b> between 0 and 1, a <b>safe default</b> for
              when nobody answers, and a <b>one-sentence explanation</b> a non-specialist can read. The same records end up in <code>plan.json</code>,{" "}
              <code>decisions.json</code> and both PDFs.
            </p>
            <figure className="wide reveal">
              <div className="fig">
                <div className="fig-head">
                  <span className="label">Figure 3</span><span className="title">Decision explorer</span>
                  <div className="tools"><span className="source">{units.sample} · {units.analysis.decisions.length} decisions</span></div>
                </div>
                <DecisionExplorer decisions={units.analysis.decisions} />
              </div>
              <figcaption>
                <b>Figure 3.</b> Every Decision <code>plan()</code> made for <code>{units.sample}</code>. Select one to see its plain-language reason and evidence. For type
                decisions, the bars are the score each candidate type received; a column is typed only when one clearly wins.
              </figcaption>
            </figure>
            <div className="ref-wrap">
              <table className="ref">
                <thead><tr><th>Field</th><th>Meaning</th></tr></thead>
                <tbody>
                  <tr><td><code>stage</code>, <code>kind</code></td><td>Where it came from (<code>reading</code>, <code>infer</code>, <code>clean</code>, <code>engineer</code>) and what it is</td></tr>
                  <tr><td><code>columns</code></td><td>Columns it applies to</td></tr>
                  <tr><td><code>action</code></td><td>What will be done, e.g. <code>cast_to_float</code>, <code>fill_median</code></td></tr>
                  <tr><td><code>evidence</code></td><td>The measurements behind it: scores, counts, learned values</td></tr>
                  <tr><td><code>confidence</code></td><td>0 to 1</td></tr>
                  <tr><td><code>default</code></td><td>The safe choice used when nobody answers</td></tr>
                  <tr><td><code>needs_attention</code></td><td><code>True</code> if a person should confirm it</td></tr>
                  <tr><td><code>plain_language</code></td><td>One sentence a non-specialist can read</td></tr>
                  <tr><td><code>status</code>, <code>history</code></td><td>Who decided it (automatic, default or user) and how it changed</td></tr>
                </tbody>
              </table>
            </div>
          </section>

          {/* ===================================================== 04 */}
          <section className="chapter" id="abstain" aria-labelledby="abs-h">
            <span className="chapter-num">04</span>
            <h2 id="abs-h">Abstain, don’t guess</h2>
            <p>
              When two readings of a column score too closely, or a conversion would throw information away, PreFlight does not quietly pick one. It keeps the column
              as text and turns the choice into a question, with an explanation, a summary of the evidence, two to four options, a recommended default, and a preview of
              real values from your file under each option.
            </p>
            <QuestionsFigure questions={units.analysis.questions} file={units.sample} />
            <p>
              All open questions are batched into one review. How they reach a person is up to you: a terminal prompt, a notebook form, a review file someone fills in
              later, or a function of your own. The playground below uses the last. With nobody available, <code>&quot;strict&quot;</code> mode stops with a list
              instead of hanging, and <code>&quot;defaults&quot;</code> takes the safe choice for every item. Answers are stored by fingerprint, so you answer once per
              dataset.
            </p>
          </section>

          {/* ===================================================== 05 */}
          <section className="chapter" id="hygiene" aria-labelledby="hyg-h">
            <span className="chapter-num">05</span>
            <h2 id="hyg-h">Fit hygiene</h2>
            <p>
              A median fill computed on the whole table has seen your test rows. So has a target encoding, a clipping limit or a rare-category list. Cross-validation
              scores built on top of them are quietly optimistic.
            </p>
            <p>
              PreFlight splits the work. Steps that learn nothing from data run on the table straight away. Every step that learns a statistic goes into an{" "}
              <b>unfitted</b> scikit-learn <code>Pipeline</code>, which you clone and fit inside each fold.
            </p>
            <HygieneFigure rows={units.result.prepared.shape[0]} />
            <CodeBlock code={CODE.cv} />
          </section>

          {/* ===================================================== 06 */}
          <section className="chapter full" id="try" aria-labelledby="try-h">
            <span className="chapter-num">06</span>
            <h2 id="try-h">Try it live</h2>
            <p style={{ maxWidth: "var(--prose)" }}>
              Everything on this page so far was computed ahead of time. The playground runs the real <code>pypreflight</code> live, through{" "}
              <a href="https://pyodide.org">Pyodide</a>, on any CSV you give it. <b>Your file never leaves your machine</b>: there is no server.
            </p>
            <TryTeaser fx={units} />
          </section>

          {/* ===================================================== 07 */}
          <section className="chapter" id="quickstart" aria-labelledby="qs-h">
            <span className="chapter-num">07</span>
            <h2 id="qs-h">Quickstart</h2>
            <h3>Install</h3>
            <CodeBlock code={CODE.install} lang="bash" />
            <p>
              The distribution is named <code>pypreflight</code>; the import name is <code>preflight</code>. It supports Python 3.10, 3.11 and 3.12, and builds on
              pandas, NumPy, SciPy, scikit-learn, charset-normalizer and ReportLab.
            </p>
            <h3>One call</h3>
            <CodeBlock code={CODE.one} />
            <p>
              In a terminal or notebook, PreFlight shows any open questions on one screen and waits for your answers. In a script or CI job it stops and lists them
              instead of hanging. <code>out</code> is a dictionary:
            </p>
            <div className="ref-wrap">
              <table className="ref">
                <thead><tr><th>Key</th><th>Contents</th></tr></thead>
                <tbody>
                  <tr><td><code>apply_result</code></td><td><code>ApplyResult</code> with <code>.data</code> (typed table), <code>.prepared</code> (typed and cleaned), <code>.pipeline</code>, <code>.decisions</code>, <code>.anomalies</code></td></tr>
                  <tr><td><code>pipeline</code></td><td><b>Unfitted</b> scikit-learn <code>Pipeline</code>. Fit it on training rows only.</td></tr>
                  <tr><td><code>pipeline_fitted_on_all_rows</code></td><td>The same pipeline fitted on every row, for new data once model selection is done. Not for evaluation.</td></tr>
                  <tr><td><code>plan</code></td><td>The final <code>Plan</code></td></tr>
                  <tr><td><code>paths</code></td><td>Paths of every file written</td></tr>
                </tbody>
              </table>
            </div>
            <p>Everything is written to <code>./preflight_out/</code> unless you pass <code>out_dir</code>. Nothing is ever written next to the source file.</p>
            <div className="tree">
              {[
                ["preflight_out/", ""],
                ["├── plan.json", "the full plan, machine-readable"],
                ["├── plan.pdf", "what PreFlight intends to do, and what it asked"],
                ["├── cleaned_data.csv", "the prepared table"],
                ["├── schema.json", "encoding, dialect, header, per-column types"],
                ["├── decisions.json", "every final Decision, and who decided it"],
                ["├── pipeline.joblib", "unfitted pipeline"],
                ["├── pipeline_fitted_on_all_rows.joblib", ""],
                ["└── report.pdf", "what was done, by whom, what looked unusual"],
              ].map(([f, d]) => <div className="f" key={f}><span>{f}</span><span>{d}</span></div>)}
            </div>
            <h3>Look before you leap</h3>
            <p><code>plan()</code> is pure: it reads the file, proposes, and writes nothing.</p>
            <CodeBlock code={CODE.plan} />
            <p>
              That prints the lines in Figure 0. <code>preflight.dry_run()</code> does the same and also resolves the questions and writes <code>plan.json</code> and{" "}
              <code>plan.pdf</code>, without producing cleaned data.
            </p>
            <h3>Answering questions from your own code</h3>
            <CodeBlock code={CODE.callback} />
            <p>The <Link href="/try">playground</Link> does exactly this: its callback returns the options you picked on the page.</p>
          </section>

          {/* ===================================================== 08 */}
          <section className="chapter" id="api" aria-labelledby="api-h">
            <span className="chapter-num">08</span>
            <h2 id="api-h">API reference</h2>
            <p>Everything below is importable from the top-level <code>preflight</code> package.</p>
            <h3 id="api-prepare">prepare</h3>
            <Sig fn="prepare" params={[["path"], ["target"], ["task", "None"], ["mode", '"auto"'], ["channel", "None"], ["out_dir", "None"], ["model_hint", '"tree"'], ["feature_config", "None"]]} ret="dict" />
            <p>Everything in one call: plan → resolve → apply → save → Report PDF. Writes all outputs.</p>
            <h3 id="api-plan">plan</h3>
            <Sig fn="plan" params={[["path"], ["target", "None"], ["task", "None"], ["model_hint", '"tree"'], ["feature_config", "None"]]} ret="Plan" />
            <p>Read, infer and propose. Writes nothing.</p>
            <h3 id="api-dry">dry_run</h3>
            <Sig fn="dry_run" params={[["path"], ["target", "None"], ["task", "None"], ["out_dir", "None"], ["mode", '"auto"'], ["channel", "None"], ["…"]]} ret="(Plan, Path)" />
            <p><code>plan</code> plus question resolution. Writes <code>plan.json</code>, <code>plan.pdf</code> and <code>answers.json</code>.</p>
            <h3 id="api-resolve">resolve</h3>
            <Sig fn="resolve" params={[["plan_or_path"], ["mode", "None"], ["channel", "None"], ["out_dir", "None"], ["target", "None"], ["task", "None"]]} />
            <p>Answer every open question. Accepts a <code>Plan</code>, a data file, or a filled-in <code>review.json</code>.</p>
            <h3 id="api-apply">apply</h3>
            <Sig fn="apply" params={[["plan"], ["decisions", "None"], ["mode", "None"], ["channel", "None"], ["out_dir", "None"]]} ret="ApplyResult" />
            <p>Run the plan. Writes files only when <code>mode</code> is given.</p>
            <p>Classes: <code>Plan</code> (<code>to_json</code>, <code>save_json</code>, <code>from_json</code>, <code>from_dict</code>), <code>Decision</code>, <code>TableSchema</code>.</p>

            <h3>Common arguments</h3>
            <div className="ref-wrap">
              <table className="ref">
                <tbody>
                  <tr><td><code>task</code></td><td><code>&quot;classification&quot;</code>, <code>&quot;regression&quot;</code>, or <code>None</code> to infer it from the target.</td></tr>
                  <tr><td><code>model_hint</code></td><td><code>&quot;tree&quot;</code> ordinal-encodes low-variety categories; <code>&quot;linear&quot;</code> one-hot encodes them and standard-scales numeric features.</td></tr>
                  <tr><td><code>out_dir</code></td><td>Defaults to <code>./preflight_out/</code>. PreFlight never writes next to the source file.</td></tr>
                </tbody>
              </table>
            </div>

            <h3 id="api-modes">Resolution modes</h3>
            <div className="ref-wrap">
              <table className="ref">
                <thead><tr><th>Mode</th><th>Behaviour</th></tr></thead>
                <tbody>
                  <tr><td><code>&quot;auto&quot;</code> <span className="badge">default</span></td><td><code>&quot;ask&quot;</code> when the session is interactive (terminal or notebook), <code>&quot;strict&quot;</code> otherwise</td></tr>
                  <tr><td><code>&quot;ask&quot;</code></td><td>Show all open questions through a channel and use the answers</td></tr>
                  <tr><td><code>&quot;defaults&quot;</code></td><td>Accept the safe default for every question. Uncertain types stay as text.</td></tr>
                  <tr><td><code>&quot;strict&quot;</code></td><td>Raise <code>StrictResolutionError</code> listing every open question. Never waits.</td></tr>
                </tbody>
              </table>
            </div>
            <p className="aside">
              <b>Tip.</b> <code>&quot;defaults&quot;</code> is deliberately cautious: the safe default for an uncertain column type is to keep the original text,
              because keeping text never destroys information. For the best result on a new dataset, answer the questions once; the answers are stored and reused.
            </p>

            <h3 id="api-channels">Channels</h3>
            <div className="ref-wrap">
              <table className="ref">
                <thead><tr><th>Channel</th><th>Use it for</th></tr></thead>
                <tbody>
                  <tr><td><code>&quot;terminal&quot;</code></td><td>Interactive prompt, formatted with <code>rich</code> if the <code>interactive</code> extra is installed</td></tr>
                  <tr><td><code>&quot;notebook&quot;</code></td><td>ipywidgets form inside Jupyter (<code>notebook</code> extra)</td></tr>
                  <tr><td><code>&quot;review_file&quot;</code></td><td>Asynchronous review: writes <code>review.json</code>, raises <code>ReviewPending</code>, continues when you call <code>preflight.resolve(&quot;…/review.json&quot;)</code></td></tr>
                  <tr><td>a callable</td><td>Your own UI or automation. Receives the list of <code>Question</code> objects and returns <code>{"{question_id: option}"}</code>.</td></tr>
                </tbody>
              </table>
            </div>
          </section>

          {/* ===================================================== 09 */}
          <section className="chapter" id="accuracy" aria-labelledby="acc-h">
            <span className="chapter-num">09</span>
            <h2 id="acc-h">Accuracy</h2>
            <p>
              PreFlight is measured against a corpus of <b>254 real public CSV files</b> (Rdatasets, TidyTuesday, FiveThirtyEight, UCI, seaborn, pandas test fixtures and
              the Frictionless messy-CSV tests) with <b>2,850 hand-labelled columns</b>.
            </p>
            <AccuracyFigure />
          </section>

          {/* ===================================================== 10 */}
          <section className="chapter" id="limits" aria-labelledby="lim-h">
            <span className="chapter-num">10</span>
            <h2 id="lim-h">Known limitations</h2>
            <div className="note warn">
              <b>PreFlight can make mistakes.</b> Every type, format and cleaning choice is inferred from the data, and some will be wrong for your file. On the
              benchmark corpus about 1 column in 11 gets the wrong type (0.46% with no flag), and about 1 file in 12 has its format misread. A high confidence score
              means the evidence was clear, not that the answer is right, and a recommended answer is a safe default, not knowledge of what your column means.
              Read the Plan PDF before <code>prepare()</code>, check the report and the cleaned data afterwards, and override anything you disagree with.
            </div>
            <p>Specific known gaps:</p>
            <ul>
              <li><b>CSV-like files only.</b> No Excel, JSON or Parquet, and no DataFrame input in 2.0.</li>
              <li><b>No command-line interface yet.</b> Use the Python API.</li>
              <li><b>The leakage check is numeric-only.</b> A text or yes/no copy of the target is not detected.</li>
              <li><b>Text identifier columns</b> with many unique values are target-encoded rather than dropped.</li>
              <li><b>Dialect detection</b> tries four separators and looks at the first 128 KB of the file.</li>
              <li><b>Type conversion</b> runs per value in Python, so very large files are slower than they need to be.</li>
              <li><b>Numeric columns that are 30–60% blank</b> keep their blanks. Add an imputer if your model cannot handle missing values.</li>
            </ul>
            <p>Bug reports and pull requests for any of these are welcome on <a href={`${REPO}/issues`}>GitHub</a>.</p>
          </section>

          <footer className="full">
            <span>
              PreFlight · MIT License · <a href={PYPI}>PyPI</a> · <a href={REPO}>GitHub</a> · <a href={`${REPO}/tree/main/site`}>Site source</a>
            </span>
            <span>
              Sample <code>autos.csv</code>: UCI Machine Learning Repository, <a href="https://archive.ics.uci.edu/dataset/10/automobile">Automobile</a> (Schlimmer, 1985), CC BY 4.0.
            </span>
          </footer>
        </main>
      </div>
    </>
  );
}
