import sys
sys.path.insert(0, "./src")
from preflight import prepare
from preflight.types import FeatureConfig
import pandas as pd
import numpy as np

n = 300
rng = np.random.default_rng(42)
df = pd.DataFrame({
    "sqft": rng.normal(1500, 300, n),
    "rooms": rng.integers(1, 6, n),
    "neighborhood": rng.choice(["A", "B", "C", "D"], n),
    "built_date": pd.date_range("2000-01-01", periods=n, freq="7D"),
    "price": rng.normal(300000, 50000, n),
})
config = FeatureConfig(datetime_cyclical=True)
result = prepare(df, target="price", task="regression", feature_config=config)
print([c for c in result.df.columns])
