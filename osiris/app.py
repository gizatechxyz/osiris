import typer
import polars as pl
import numpy as np

from osiris.dtypes.input_output_formats import InputFormat, OutputFormat
from osiris.cairo.data_converter.data_converter import convert_to_cairo
from osiris.dtypes.cairo_dtypes import Dtype

app = typer.Typer()


@app.command()
def convert(
    input_file: str,
    output_file: str,
    input_format: InputFormat = InputFormat.CSV,
    output_format: OutputFormat = OutputFormat.NUMPY,
    dtype: Dtype = Dtype.I32.value,
):
    typer.echo("🚀 Starting the conversion process...")
    typer.echo(f"📂 Loading data from {input_file}...")

    # Load data
    if input_format == InputFormat.CSV:
        df = pl.read_csv(input_file)
    elif input_format == InputFormat.PARQUET:
        df = pl.read_parquet(input_file)
    elif input_format == InputFormat.NUMPY:
        df = np.load(input_file)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    typer.echo("✅ Data loaded successfully!")

    # Convert to numpy
    if not isinstance(df, np.ndarray):
        typer.echo("🔄 Converting data to numpy...")
        np_array = df.to_numpy()
    else:
        np_array = df

    typer.echo("✅ Conversion to numpy completed!")

    # Convert to specified output format
    typer.echo(f"🔄 Converting data to {output_format.value}...")
    match output_format:
        case OutputFormat.CAIRO:
            convert_to_cairo(np_array, output_file, dtype)
        case OutputFormat.NUMPY:
            np.save(output_file, np_array)
        case _:
            raise ValueError(f"Unsupported output format: {output_format}")

    typer.echo("✅ Conversion process completed! 🎉")


if __name__ == "__main__":
    app()
