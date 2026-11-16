# CHANGELOG

## Unreleased

### Changed
- A column whose type is clear but is flagged only because some values are placeholders for
  missing data (`?`, `N/A`, `-999`) now recommends that type instead of "Keep as plain text",
  and `mode="defaults"` converts it. Close calls, units, leading zeros and ambiguous dates
  still default to plain text. Stored answers for affected columns are asked again once,
  because the decision's fingerprint includes its default.
- Type questions always offer "Keep as plain text", and explain when a column is flagged only
  for missing-value placeholders.
