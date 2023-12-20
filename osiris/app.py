import json
import os

import numpy as np
import polars as pl
import typer

from osiris.cairo.data_converter.data_converter import convert_to_cairo
from osiris.cairo.serde.data_structures import create_tensor_from_array
from osiris.cairo.serde.deserialize import deserializer
from osiris.cairo.serde.serialize import serializer
from osiris.dtypes.cairo_dtypes import Dtype
from osiris.dtypes.input_output_formats import InputFormat, OutputFormat

app = typer.Typer()


def check_file_format(file_path):
    _, file_extension = os.path.splitext(file_path)

    if file_extension in ['.csv']:
        return 'CSV'
    elif file_extension in ['.parquet']:
        return 'Parquet'
    elif file_extension in ['.npy']:
        return 'NumPy'
    else:
        return 'Unknown'


def load_data(input_file: str):
    """
    Load data from a file into a DataFrame or numpy array.

    Args:
    input_file (str): The path to the input file.

    Returns:
    DataFrame or numpy array: The loaded data.
    """
    typer.echo(f"📂 Loading data from {input_file}...")

    input_format = check_file_format(input_file)
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
    """
    Convert the given data to a numpy array.

    Args:
    data: Data to be converted. Can be a DataFrame or numpy array.

    Returns:
    numpy array: The converted data.
    """
    if not isinstance(data, np.ndarray):
        typer.echo("🔄 Converting data to numpy...")
        return data.to_numpy()
    return data


@app.command()
def serialize(input_file: str, fp_impl: str = 'FP16x16'):
    """
    Serialize data from a file to a tensor representation.

    Args:
    input_file (str): The path to the input file.
    fp_impl (str): Fixed-point implementation detail.

    Returns:
    Serialized tensor.
    """

    typer.echo("🚀 Starting the conversion process...")
    data = load_data(input_file)
    typer.echo("✅ Data loaded successfully!")

    numpy_array = convert_to_numpy(data)
    typer.echo("✅ Conversion to numpy completed!")

    tensor = create_tensor_from_array(numpy_array, fp_impl)
    typer.echo("✅ Conversion to tensor completed!")

    serialized = serializer(tensor)
    typer.echo("✅ Serialized tensor successfully! 🎉")

    return serialized


@app.command()
def deserialize(serialized: str, data_type: str, fp_impl: str = 'FP16x16'):
    """
    Deserialize a serialized string into a specific data type.

    Args:
    serialized (str): Serialized data in string format.
    data_type (str): The type of data to deserialize into.
    fp_impl (str): Fixed-point implementation detail.

    Returns:
    Deserialized data.
    """
    typer.echo("🚀 Starting deserialization process...")

    try:
        serialized = json.loads(serialized)
    except json.JSONDecodeError as e:
        typer.echo(f"Error: Invalid JSON - {e}")
        raise typer.Exit(code=1) from e

    deserialized = deserializer(serialized, data_type, fp_impl)
    typer.echo("✅ Deserialization completed! 🎉")

    return deserialized


@app.command()
def convert(input_file: str, output_file: str, output_format: OutputFormat = OutputFormat.NUMPY, dtype: Dtype = Dtype.I32):
    """
    Convert data from one format to another.

    Args:
    input_file (str): The path to the input file.
    output_file (str): The path for the output file.
    output_format (OutputFormat): The format for the output file.
    dtype (Dtype): Data type for Cairo conversion.

    """
    typer.echo("🚀 Starting the conversion process...")
    data = load_data(input_file)
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
