import numpy as np
import pytest
from osiris.cairo.serde.data_structures import create_tensor_from_array, Tensor, SignedInt, FixedPoint
from osiris.cairo.serde.serialize import serializer


def test_create_tensor_from_array_with_integers():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    tensor = create_tensor_from_array(arr)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == arr.shape
    assert all(isinstance(x, SignedInt) for x in tensor.data)


def test_create_tensor_from_array_with_floats():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    tensor = create_tensor_from_array(arr)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == arr.shape
    assert all(isinstance(x, FixedPoint) for x in tensor.data)


def test_create_tensor_from_array_with_invalid_input():
    with pytest.raises(TypeError):
        create_tensor_from_array("not a numpy array")


def test_serializer_for_tensor_signedint():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    tensor = create_tensor_from_array(arr)
    serialized_data = serializer(tensor)
    assert isinstance(serialized_data, list)


def test_serializer_for_tensor_uint():
    arr = np.array([[1, 2], [3, 4]], dtype=np.uint64)
    tensor = create_tensor_from_array(arr)
    serialized_data = serializer(tensor)
    assert isinstance(serialized_data, list)
