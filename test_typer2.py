import typer
from typer.testing import CliRunner
app = typer.Typer(rich_markup_mode=None)
@app.command()
def main(datetime_cyclical: bool = typer.Option(False, '--datetime-cyclical/--no-datetime-cyclical')): pass
print(CliRunner().invoke(app, ['--help']).output)
