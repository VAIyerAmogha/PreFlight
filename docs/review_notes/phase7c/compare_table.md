| File | Model | v1 mean ± std | v2 mean ± std | Verdict | Note |
|---|---|---|---|---|---|
| adult.data | linear (acc) | failed | 0.8466 ± 0.0074 | v1 FAILED | subsampled to 4,000 rows; v1: KeyError: "target 'income' not in columns after plain read_csv: ['32', ' Private', ' 205019', ' Assoc-acdm', ' |
| adult.data | tree (acc) | failed | 0.8502 ± 0.0056 | v1 FAILED | subsampled to 4,000 rows; v1: KeyError: "target 'income' not in columns after plain read_csv: ['32', ' Private', ' 205019', ' Assoc-acdm', ' |
| iris.data | linear (acc) | failed | 0.9546 ± 0.0245 | v1 FAILED | v1: KeyError: "target 'species' not in columns after plain read_csv: ['5.1', '3.5', '1.4', '0.2', 'Iris-setosa']" |
| iris.data | tree (acc) | failed | 0.9569 ± 0.0215 | v1 FAILED | v1: KeyError: "target 'species' not in columns after plain read_csv: ['5.1', '3.5', '1.4', '0.2', 'Iris-setosa']" |
| winequality-red.csv | linear (R²) | failed | 0.3597 ± 0.0382 | v1 FAILED | v1: KeyError: 'target \'quality\' not in columns after plain read_csv: [\'fixed acidity;"volatile acidity";"citric |
| winequality-red.csv | tree (R²) | failed | 0.4557 ± 0.0271 | v1 FAILED | v1: KeyError: 'target \'quality\' not in columns after plain read_csv: [\'fixed acidity;"volatile acidity";"citric |
| winequality-white.csv | linear (R²) | failed | 0.2794 ± 0.0242 | v1 FAILED | subsampled to 4,000 rows; v1: KeyError: 'target \'quality\' not in columns after plain read_csv: [\'fixed acidity;"volatile acidity";"citric |
| winequality-white.csv | tree (R²) | failed | 0.4702 ± 0.0194 | v1 FAILED | subsampled to 4,000 rows; v1: KeyError: 'target \'quality\' not in columns after plain read_csv: [\'fixed acidity;"volatile acidity";"citric |
| titanic.csv | linear (acc) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | TIE |  |
| titanic.csv | tree (acc) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | TIE |  |
| tips.csv | linear (R²) | 0.3825 ± 0.0689 | 0.3964 ± 0.0631 | TIE |  |
| tips.csv | tree (R²) | 0.1956 ± 0.1719 | 0.1968 ± 0.1187 | TIE |  |
| AER_Affairs.csv | linear (R²) | 0.0869 ± 0.0460 | 0.0932 ± 0.0422 | TIE |  |
| AER_Affairs.csv | tree (R²) | 0.0459 ± 0.0635 | 0.0660 ± 0.0651 | TIE |  |
| diamonds.csv | linear (R²) | 0.9202 ± 0.0053 | 0.9183 ± 0.0097 | TIE | subsampled to 4,000 rows |
| diamonds.csv | tree (R²) | 0.9616 ± 0.0036 | 0.9635 ± 0.0062 | TIE | subsampled to 4,000 rows |
| carData_TitanicSurvival.csv | linear (acc) | 0.6175 ± 0.0022 | 0.7841 ± 0.0093 | v2 better |  |
| carData_TitanicSurvival.csv | tree (acc) | 0.6180 ± 0.0010 | 0.7545 ± 0.0207 | v2 better |  |
| COUNT_titanic.csv | linear (acc) | 0.7773 ± 0.0119 | 0.7773 ± 0.0119 | TIE |  |
| COUNT_titanic.csv | tree (acc) | 0.7923 ± 0.0116 | 0.7844 ± 0.0132 | TIE |  |
