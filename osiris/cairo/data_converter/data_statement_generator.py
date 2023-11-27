from enum import Enum
import numpy as np
from typing import List, Sequence
from osiris.dtypes.cairo_dtypes import Dtype


class Trait(Enum):
    TENSOR = "TENSOR"
    NN = "NN"


def get_data_refs(dtype: Dtype) -> list[str]:
    """
    Get the data references based on the data type.

    Args:
        dtype (Dtype): The data type of the tensor.

    Returns:
        list[str]: The data references.
    """
    refs = []
    refs.extend(trait_to_ref[Trait.TENSOR])
    refs.extend(dtype_to_tensor[dtype])
    refs.extend(dtype_to_numbers[dtype])
    return refs


class DataStatement(Enum):
    U32 = "u32"
    I32 = "i32 { mag: {magnitude}, sign: {sign} }"
    I8 = "i8 { mag: {magnitude}, sign: {sign} }"
    FP8x23 = "FP8x23 { mag: {magnitude}, sign: {sign} }"
    FP16x16 = "FP16x16 { mag: {magnitude}, sign: {sign} }"


def get_data_statement(data: np.ndarray, dtype: Dtype) -> list[str]:
    """
    Generate data statements based on the data type.

    Args:
        data (np.ndarray): The numpy array to convert.
        dtype (Dtype): The data type of the tensor.

    Returns:
        list[str]: The generated data statements.
    """
    statement_template = DataStatement[dtype.name].value
    return [
        statement_template.format(magnitude=int(x), sign=str(x < 0).lower())
        for x in data.flatten()
    ]


def get_data_statement_for_sequences(data: Sequence, dtype: Dtype) -> list[list[str]]:
    """
    Generate data statements for sequences.

    Args:
        data (Sequence): The sequence data.
        dtype (Dtype): The data type of the tensor.

    Returns:
        list[list[str]]: The generated data statements for sequences.
    """
    return [get_data_statement(tensor.data, dtype) for tensor in data]


trait_to_ref = {
    Trait.TENSOR: [
        "array::{ArrayTrait, SpanTrait}",
        "orion::operators::tensor::{TensorTrait, Tensor}",
    ],
    Trait.NN: [
        "orion::numbers::FixedTrait",
        "orion::operators::nn::NNTrait",
    ],
}

dtype_to_tensor = {
    Dtype.U32: [
        "orion::operators::tensor::U32Tensor",
    ],
    Dtype.I32: [
        "orion::operators::tensor::I32Tensor",
    ],
    Dtype.I8: [
        "orion::operators::tensor::I8Tensor",
    ],
    Dtype.FP8x23: [
        "orion::operators::tensor::FP8x23Tensor",
    ],
    Dtype.FP16x16: [
        "orion::operators::tensor::FP16x16Tensor",
    ],
}

dtype_to_numbers = {
    Dtype.U32: [],
    Dtype.I32: [
        "orion::numbers::{IntegerTrait, i32}",
    ],
    Dtype.I8: [
        "orion::numbers::{IntegerTrait, i8}",
    ],
    Dtype.FP8x23: [
        "orion::numbers::{FixedTrait, FP8x23}",
    ],
    Dtype.FP16x16: [
        "orion::numbers::{FixedTrait, FP16x16}",
    ],
}
