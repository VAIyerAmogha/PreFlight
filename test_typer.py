import typer
from typer.testing import CliRunner

app = typer.Typer()

@app.callback()
def main():
    pass

@app.command()
def prepare(a: str):
    print(f"Prepared {a}")

runner = CliRunner()
print(runner.invoke(app, ['prepare', 'foo']).output)
