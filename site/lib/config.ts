// Site configuration. The only file to edit when the package version changes.

export const PYODIDE_VERSION = "0.27.2";

// Where the browser gets pypreflight from:
//   * a path ending in .whl is served from this site (public/wheels/). Use it until
//     2.0.0 is on PyPI: PyPI has only 1.x, which lacks the v2 API.
//   * anything else is a pip requirement that micropip resolves on PyPI,
//     e.g. "pypreflight==2.0.0". Switch to that after the release (DEMO_DEV.md Q-03).
export const PREFLIGHT_SPEC = "pypreflight==2.0.0";

export interface Sample {
  file: string;
  target: string;
  tag: string;
  desc: string;
}

// Bundled samples in public/samples/. Keep in sync with scripts/build_fixtures.py.
export const SAMPLES: Sample[] = [
  { file: "disguised_missing_units.csv", target: "outcome", tag: "UNI", desc: "60 patients · units, -999, ? and an ID column" },
  { file: "messy_noheader_semicolon.csv", target: "col_3", tag: "SEP", desc: "Iris-like · semicolons, no header row" },
  { file: "autos.csv", target: "col_25", tag: "UCI", desc: "UCI Automobile · 205 cars, headerless, ? = missing" },
];

export const MAX_UPLOAD_MB = 50;
export const REPO = "https://github.com/VAIyerAmogha/PreFlight";
export const PYPI = "https://pypi.org/project/pypreflight/";
