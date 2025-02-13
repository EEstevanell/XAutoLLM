import typer
from pathlib import Path
from rich.console import Console
import subprocess

app = typer.Typer(name="streamlit")
console = Console()


@app.command()
def demo():
    """
    🌟 Launch streamlit demo.
    """

    try:
        import os.path
        import autogoal_streamlit

        subprocess.call(
            [
                "streamlit",
                "run",
                Path(os.path.abspath(autogoal_streamlit.__file__)).parent / "demo.py",
            ]
        )
    except ImportError:
        console.print("(!) Too run the demo you need streamlit installed.")
        console.print("(!) Fix it by running `pip install autogoal[streamlit]`.")


global typer_app
typer_app = app
