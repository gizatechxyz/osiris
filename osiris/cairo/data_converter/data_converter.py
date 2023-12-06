# This module convert Data source to Cairo code.


import numpy as np

from osiris.cairo.data_converter.data_statement_generator import (
    get_data_refs,
    get_data_statement,
)
from osiris.cairo.data_converter.tensor_creator import Tensor, create_tensor
from osiris.cairo.file_manager.cairo_data import CairoData
from osiris.dtypes.cairo_dtypes import Dtype


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
