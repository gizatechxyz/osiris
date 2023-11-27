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
    dtype: Dtype = Dtype.I32,
):
    # Load data
    if input_format == InputFormat.CSV:
        df = pl.read_csv(input_file)
    elif input_format == InputFormat.PARQUET:
        df = pl.read_parquet(input_file)
    elif input_format == InputFormat.NUMPY:
        df = pl.from_pandas(np.load(input_file))
    else:
        raise ValueError(f"Unsupported input format: {input_format}")
    # Convert to numpy
    np_array = df.to_numpy()

    # Convert to specified output format
    match output_format:
        case "cairo":
            convert_to_cairo(np_array, output_file, dtype)
        case "numpy":
            np.save(output_file, np_array)
        case _:
            raise ValueError(f"Unsupported output format: {output_format}")


if __name__ == "__main__":
    app()
