import typer
from typer.testing import CliRunner

app = typer.Typer()
@app.command()
def test(
    interactions: bool = typer.Option(False, "--interactions/--no-interactions"),
):
    print(f"interactions={interactions}")

runner = CliRunner()
result = runner.invoke(app, ["--interactions"])
print(result.output)
