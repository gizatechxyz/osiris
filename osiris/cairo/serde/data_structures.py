import numpy as np

from .utils import to_fp


class Tensor:
    def __init__(self, shape: tuple, data):
        self.shape = shape
        self.data = data


class SignedInt:
    def __init__(self, mag, sign):
        self.mag = mag
        self.sign = sign


class FixedPoint:
    def __init__(self, mag, sign):
        self.mag = mag
        self.sign = sign


def create_tensor_from_array(arr: np.ndarray, fp_impl='FP16x16'):
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    flat_array = arr.flatten()
    tensor_data = []

    for value in flat_array:
        if isinstance(value, (np.unsignedinteger)):
            tensor_data.append(value)
        elif isinstance(value, (int, np.integer, np.signedinteger)):
            sign = 0 if value >= 0 else 1
            tensor_data.append(SignedInt(abs(value), sign))
        elif isinstance(value, (float, np.floating)):
            (mag, sign) = to_fp(value, fp_impl)
            tensor_data.append(FixedPoint(mag, sign))
        else:
            raise TypeError(f"Unsupported type in tensor data: {type(value)}")

    return Tensor(arr.shape, tensor_data)
