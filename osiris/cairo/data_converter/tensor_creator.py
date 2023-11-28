from typing import List

import numpy as np
from numpy import ndarray

from osiris.dtypes.cairo_dtypes import Dtype
from osiris.dtypes.cairo_impls import FixedImpl

# Define constants
FP8x23_FACTOR = 2**23
FP16x16_FACTOR = 2**16
FP32x32_FACTOR = 2**32


def to_fp(x: np.ndarray, fp_impl: FixedImpl):
    """
    Convert numpy array to fixed point representation.

    Args:
        x (np.ndarray): The numpy array to convert.
        fp_impl (FixedImpl): The fixed point implementation to use.

    Returns:
        np.ndarray: The converted numpy array.
    """
    match fp_impl:
        case FixedImpl.FP8x23:
            return (x * FP8x23_FACTOR).astype(np.int64)
        case FixedImpl.FP16x16:
            return (x * FP16x16_FACTOR).astype(np.int64)
        case FixedImpl.FP32x32:
            return (x * FP32x32_FACTOR).astype(np.int64)


class Tensor:
    def __init__(self, dtype: Dtype, shape: tuple, data: np.ndarray):
        self.dtype = dtype
        self.shape = shape
        match dtype:
            case Dtype.FP8x23:
                self.data = to_fp(data.flatten(), FixedImpl.FP8x23)
            case Dtype.FP16x16:
                self.data = to_fp(data.flatten(), FixedImpl.FP16x16)
            case Dtype.FP32x32:
                self.data = to_fp(data.flatten(), FixedImpl.FP32x32)
            case _:
                self.data = data.flatten()


Sequence = List[Tensor]


def create_tensor(dtype: Dtype, shape: tuple, data: ndarray) -> Tensor:
    """
    Create a Tensor object.

    Args:
        dtype (Dtype): The data type of the tensor.
        shape (tuple): The shape of the tensor.
        data (np.ndarray): The data of the tensor.

    Returns:
        Tensor: The created Tensor object.
    """
    return Tensor(dtype, shape, data)
