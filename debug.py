from preflight.cli import app
from typer.testing import CliRunner
import tempfile
import pandas as pd
import numpy as np

runner = CliRunner()
n = 50
df = pd.DataFrame({
    "age": np.random.normal(40, 10, n),
    "city": np.random.choice(["NYC", "LA", "SF"], n),
    "target": np.random.choice([0, 1], n),
})
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
    df.to_csv(f.name, index=False)
    path = f.name

result = runner.invoke(app, ["prepare", path, "--target", "target"])
print("Exit code:", result.exit_code)
print("Output:", result.output)
if result.exception:
    print("Exception:", result.exception)
