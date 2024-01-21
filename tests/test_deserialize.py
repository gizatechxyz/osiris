import numpy as np
import pytest
from math import isclose

from osiris.cairo.serde.deserialize import *


def test_deserialize_signed_int():
    serialized = '[{"Int":"2A"}, {"Int":"0"}]'
    deserialized = deserializer(serialized, 'signed_int')
    assert deserialized == 42

    serialized = '[{"Int":"2A"}, {"Int":"0x1"}]'
    deserialized = deserializer(serialized, 'signed_int')
    assert deserialized == -42


def test_deserialize_fp():
    serialized = '[{"Int":"2A6B85"}, {"Int":"0"}]'
    deserialized = deserializer(serialized, 'fixed_point', 'FP16x16')
    assert isclose(deserialized, 42.42, rel_tol=1e-7)

    serialized = '[{"Int":"2A6B85"}, {"Int":"1"}]'
    deserialized = deserializer(serialized, 'fixed_point', 'FP16x16')
    assert isclose(deserialized, -42.42, rel_tol=1e-7)


def test_deserialize_array_uint():
    serialized = '[{"Array": [{"Int": "0x1"}, {"Int": "0x2"}]}]'
    deserialized = deserializer(serialized, 'arr_uint')
    assert np.array_equal(deserialized, np.array([1, 2], dtype=np.int64))


def test_deserialize_array_signed_int():
    serialized = '[{"Array": [{"Int": "2A"}, {"Int": "0"}, {"Int": "2A"}, {"Int": "0x1"}]}]'
    deserialized = deserializer(serialized, 'arr_signed_int')
    assert np.array_equal(deserialized, np.array([42, -42], dtype=np.int64))


def test_deserialize_arr_fixed_point():
    serialized = '[{"Array": [{"Int": "2A6B85"}, {"Int": "0"}, {"Int": "2A6B85"}, {"Int": "0x1"}]}]'
    deserialized = deserializer(serialized, 'arr_fixed_point')
    expected = np.array([42.42, -42.42], dtype=np.float64)
    assert np.all(np.isclose(deserialized, expected, atol=1e-7))


def test_deserialize_tensor_uint():
    serialized = '[{"Array": [{"Int": "0x2"}, {"Int": "0x2"}]}, {"Array": [{"Int": "0x1"}, {"Int": "0x2"}, {"Int": "0x3"}, {"Int": "0x4"}]}]'
    deserialized = deserializer(serialized, 'tensor_uint')
    assert np.array_equal(deserialized, np.array(
        ([1, 2], [3, 4]), dtype=np.int64))


def test_deserialize_tensor_signed_int():
    serialized = '[{"Array": [{"Int": "0x2"}, {"Int": "0x2"}]}, {"Array": [{"Int": "2A"}, {"Int": "0x0"}, {"Int": "2A"}, {"Int": "0x0"}, {"Int": "2A"}, {"Int": "0x1"}, {"Int": "2A"}, {"Int": "0x1"}]}]'
    deserialized = deserializer(serialized, 'tensor_signed_int')
    assert np.array_equal(deserialized, np.array([[42, 42], [-42, -42]]))


def test_deserialize_tensor_fixed_point():
    serialized = '[{"Array": [{"Int": "0x2"}, {"Int": "0x2"}]}, {"Array": [{"Int": "2A6B85"}, {"Int": "0x0"}, {"Int": "2A6B85"}, {"Int": "0x0"}, {"Int": "2A6B85"}, {"Int": "0x1"}, {"Int": "2A6B85"}, {"Int": "0x1"}]}]'
    expected_array = np.array([[42.42, 42.42], [-42.42, -42.42]])
    deserialized = deserializer(serialized, 'tensor_fixed_point')
    assert np.allclose(deserialized, expected_array, atol=1e-7)


# def test_deserialize_tuple_uint():
#     serialized = [1, 2]
#     deserialized = deserialize_tuple_uint(serialized)
#     assert np.array_equal(deserialized, np.array([1, 2], dtype=np.int64))


# def test_deserialize_tuple_signed_int():
#     serialized = [42, 0, 42, 1, 42, 0]
#     deserialized = deserialize_tuple_signed_int(serialized)
#     assert np.array_equal(deserialized, np.array(
#         [42, -42, 42], dtype=np.int64))


# def test_deserialize_tuple_fixed_point():
#     serialized = [2780037, 0, 2780037, 1, 2780037, 0]
#     deserialized = deserialize_tuple_fixed_point(serialized)
#     expected = np.array([42.42, -42.42, 42.42], dtype=np.float64)
#     assert np.all(np.isclose(deserialized, expected, atol=1e-7))

# def test_deserialize_tensor_tuple_tensor_uint():
#     serialized = [2, 2, 2, 4, 1, 2, 3, 4, 2, 2, 2, 4, 5, 6, 7, 8]
#     deserialized = deserialize_tuple_tensor_uint(serialized)

#     assert np.array_equal(deserialized[0], np.array(
#         [[1, 2], [3, 4]], dtype=np.int64))
#     assert np.array_equal(deserialized[1], np.array(
#         [[5, 6], [7, 8]], dtype=np.int64))


# def test_deserialize_tensor_tuple_tensor_signed_int():
#     serialized = [2, 2, 2, 8, 42,
#                   0, 42, 0, 42, 1, 42, 1, 2, 2, 2, 8, 42,
#                   0, 42, 0, 42, 1, 42, 1]
#     deserialized = deserialize_tuple_tensor_signed_int(serialized)

#     expected_array = np.array([[42, 42], [-42, -42]])
#     assert np.allclose(deserialized[0], expected_array, atol=1e-7)
#     assert np.allclose(deserialized[1], expected_array, atol=1e-7)


# def test_deserialize_tensor_tuple_tensor_fixed_point():
#     serialized = [2, 2, 2, 8, 2780037,
#                   0, 2780037, 0, 2780037, 1, 2780037, 1, 2, 2, 2, 8, 2780037,
#                   0, 2780037, 0, 2780037, 1, 2780037, 1]
#     deserialized = deserialize_tuple_tensor_fixed_point(serialized)

#     expected_array = np.array([[42.42, 42.42], [-42.42, -42.42]])
#     assert np.allclose(deserialized[0], expected_array, atol=1e-7)
#     assert np.allclose(deserialized[1], expected_array, atol=1e-7)
