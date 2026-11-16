Minimal local type stubs for third-party packages that ship no type information.
They declare only what PreFlight subclasses or calls in typed code, so that
`mypy --strict` on src/preflight can check those classes; every other module of
these packages stays untyped (`ignore_missing_imports`). Used via `mypy_path`.
