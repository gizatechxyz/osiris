# This module convert Data source to Cairo code.

from enum import Enum
import os
from typing import List
import numpy as np

from osiris.cairo.file_manager.file import ModFile
from osiris.cairo.file_manager.cairo_data import CairoData
from osiris.dtypes.cairo_dtypes import Dtype
from osiris.dtypes.cairo_impls import FixedImpl
from osiris.cairo.data_converter.tensor_creator import create_tensor, Tensor, Sequence
from osiris.cairo.data_converter.data_statement_generator import (
    get_data_refs,
    get_data_statement,
    get_data_statement_for_sequences,
)


def create_cairo_data(output_file: str, tensor: Tensor) -> CairoData:
    """
    Create a CairoData object.

    Args:
        output_file (str): The path to the output file.
        tensor (Tensor): The Tensor object.

    Returns:
        CairoData: The created CairoData object.
    """
    cairo_data = CairoData(output_file)
    cairo_data.buffer = CairoData.base_template(
        func="main",
        dtype=tensor.dtype.value,
        refs=get_data_refs(tensor.dtype),
        data=get_data_statement(tensor.data, tensor.dtype),
        shape=tensor.shape,
    )
    return cairo_data


def convert_to_cairo(np_array: np.ndarray, output_file: str, dtype: Dtype):
    """
    Convert numpy array to Cairo format and save to output file.

    Args:
        np_array (np.ndarray): The numpy array to convert.
        output_file (str): The path to the output file.
        dtype (Dtype): The data type of the tensor.
    """
    tensor = create_tensor(dtype, np_array.shape, np_array)
    cairo_data = create_cairo_data(output_file, tensor)
    cairo_data.dump()


def inputs_gen(inputs: list[Tensor | Sequence]):
    """
    Generate and write Cairo file based on the provided inputs .

    Args:
        inputs (list[Tensor | list[Tensor]]): A list of input tensors or tensor sequences.
        name (str): The name of the inputs file.
    """
    inputs_name = "inputs"

    ModFile().update(inputs_name)

    for i, input in enumerate(inputs):
        input_data = CairoData(os.path.join(inputs_name, f"input_{i}.cairo"))
        match input:
            case list():
                input_data.buffer = CairoData.sequence_template(
                    func=f"input_{i}",
                    dtype=input[0].dtype.value,
                    refs=get_data_refs(input[0].dtype),
                    data=get_data_statement_for_sequences(input, input[0].dtype),
                    shape=[x.shape for x in input],
                )
            case Tensor():
                input_data.buffer = CairoData.base_template(
                    func=f"input_{i}",
                    dtype=input.dtype.value,
                    refs=get_data_refs(input.dtype),
                    data=get_data_statement(input.data, input.dtype),
                    shape=input.shape,
                )

        input_data.dump()
