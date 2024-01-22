import numpy as np
import pytest
from osiris.cairo.serde.data_structures import create_tensor_from_array, Tensor, FixedPoint, Int
from osiris.cairo.serde.serialize import serializer


def test_create_tensor_from_array_with_integers():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    tensor = create_tensor_from_array(arr)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == arr.shape
    assert all(isinstance(x, Int) for x in tensor.data)


def test_create_tensor_from_array_with_floats():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    tensor = create_tensor_from_array(arr)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == arr.shape
    assert all(isinstance(x, FixedPoint) for x in tensor.data)


def test_create_tensor_from_array_with_invalid_input():
    with pytest.raises(TypeError):
        create_tensor_from_array("not a numpy array")


def test_serializer_for_boolean():
    data = False
    serialized_data = serializer(data)
    assert serialized_data == "0"


def test_serializer_for_int():
    data = 42
    serialized_data = serializer(data)
    assert serialized_data == "42"


def test_serializer_for_list():
    data = [1, 2, 3]
    serialized_data = serializer(data)
    assert serialized_data == "[1 2 3]"


def test_serializer_for_tuple():
    data = (1, 2, 3)
    serialized_data = serializer(data)
    assert serialized_data == "[1 2 3]"


def test_serializer_for_fixedpoint():
    data = FixedPoint(42, True)
    serialized_data = serializer(data)
    assert serialized_data == "42 1"


def test_serializer_for_tensor_int():
    arr = np.array([[1, 2], [3, 4]], dtype=np.uint64)
    tensor = create_tensor_from_array(arr)
    serialized_data = serializer(tensor)
    assert serialized_data == "[2 2] [1 2 3 4]"


def test_serializer_for_tensor_fixedpoint():
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    tensor = create_tensor_from_array(arr)
    serialized_data = serializer(tensor)
    assert serialized_data == "[2 2] [65536 0 131072 0 196608 0 262144 0]"
