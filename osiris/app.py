import numpy as np
import polars as pl
import typer

from osiris.cairo.data_converter.data_converter import convert_to_cairo
from osiris.dtypes.cairo_dtypes import Dtype
from osiris.dtypes.input_output_formats import InputFormat, OutputFormat
from osiris.cairo.serde.serialize import serializer
from osiris.cairo.serde.data_structures import create_tensor_from_array

app = typer.Typer()


def load_data(input_file: str, input_format: InputFormat):
    typer.echo(f"📂 Loading data from {input_file}...")
    match input_format:
        case InputFormat.CSV:
            return pl.read_csv(input_file)
        case InputFormat.PARQUET:
            return pl.read_parquet(input_file)
        case InputFormat.NUMPY:
            return np.load(input_file)
        case _:
            raise ValueError(f"Unsupported input format: {input_format}")


def convert_to_numpy(data):
    if not isinstance(data, np.ndarray):
        typer.echo("🔄 Converting data to numpy...")
        return data.to_numpy()
    return data


@app.command()
def serialize(input_file: str, input_format: InputFormat = InputFormat.CSV, fp_impl: str = 'FP16x16'):
    typer.echo("🚀 Starting the conversion process...")
    data = load_data(input_file, input_format)
    typer.echo("✅ Data loaded successfully!")

    numpy_array = convert_to_numpy(data)
    typer.echo("✅ Conversion to numpy completed!")

    tensor = create_tensor_from_array(numpy_array, fp_impl)
    typer.echo("✅ Conversion to tensor completed!")

    serialized = serializer(tensor)
    typer.echo("✅ Serialized tensor successfully! 🎉")

    return serialized


@app.command()
def convert(input_file: str, output_file: str, input_format: InputFormat = InputFormat.CSV, output_format: OutputFormat = OutputFormat.NUMPY, dtype: Dtype = Dtype.I32):
    typer.echo("🚀 Starting the conversion process...")
    data = load_data(input_file, input_format)
    typer.echo("✅ Data loaded successfully!")

    numpy_array = convert_to_numpy(data)
    typer.echo("✅ Conversion to numpy completed!")

    typer.echo(f"🔄 Converting data to {output_format.value}...")
    match output_format:
        case OutputFormat.CAIRO:
            convert_to_cairo(numpy_array, output_file, dtype)
        case OutputFormat.NUMPY:
            np.save(output_file, numpy_array)
        case _:
            raise ValueError(f"Unsupported output format: {output_format}")

    typer.echo("✅ Conversion process completed! 🎉")


if __name__ == "__main__":
    app()
