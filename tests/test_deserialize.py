import numpy as np
import pytest
from math import isclose

from osiris.cairo.serde.deserialize import *


def test_deserialize_signed_int():
    serialized = [42, 0]
    deserialized = deserialize_signed_int(serialized)
    assert deserialized == 42

    serialized = [42, 1]
    deserialized = deserialize_signed_int(serialized)
    assert deserialized == -42


def test_deserialize_signed_int():
    serialized = [2780037, 0]
    deserialized = deserialize_fixed_point(serialized, 'FP16x16')
    assert isclose(deserialized, 42.42, rel_tol=1e-7)

    serialized = [2780037, 1]
    deserialized = deserialize_fixed_point(serialized, 'FP16x16')
    assert isclose(deserialized, -42.42, rel_tol=1e-7)


def test_deserialize_array_uint():
    serialized = [2, 1, 2]
    deserialized = deserialize_arr_uint(serialized)
    assert np.array_equal(deserialized, np.array([1, 2], dtype=np.int64))


def test_deserialize_array_signed_int():
    serialized = [2, 42, 0, 42, 1]
    deserialized = deserialize_arr_signed_int(serialized)
    assert np.array_equal(deserialized, np.array([42, -42], dtype=np.int64))


def test_deserialize_arr_fixed_point():
    serialized = [2, 2780037, 0, 2780037, 1]
    deserialized = deserialize_arr_fixed_point(serialized)
    expected = np.array([42.42, -42.42], dtype=np.float64)
    assert np.all(np.isclose(deserialized, expected, atol=1e-7))


def test_deserialize_tuple_uint():
    serialized = [1, 2]
    deserialized = deserialize_tuple_uint(serialized)
    assert np.array_equal(deserialized, np.array([1, 2], dtype=np.int64))


def test_deserialize_tuple_signed_int():
    serialized = [42, 0, 42, 1, 42, 0]
    deserialized = deserialize_tuple_signed_int(serialized)
    assert np.array_equal(deserialized, np.array(
        [42, -42, 42], dtype=np.int64))


def test_deserialize_tuple_fixed_point():
    serialized = [2780037, 0, 2780037, 1, 2780037, 0]
    deserialized = deserialize_tuple_fixed_point(serialized)
    expected = np.array([42.42, -42.42, 42.42], dtype=np.float64)
    assert np.all(np.isclose(deserialized, expected, atol=1e-7))


def test_deserialize_tensor_uint():
    serialized = [2, 2, 2, 4, 1, 2, 3, 4]
    deserialized = deserialize_tensor_uint(serialized)
    assert np.array_equal(deserialized, np.array(
        ([1, 2], [3, 4]), dtype=np.int64))


def test_deserialize_tensor_signed_int():
    serialized_tensor = [2, 2, 2, 8, 42, 0, 42, 0, 42, 1, 42, 1]
    deserialized = deserialize_tensor_signed_int(serialized_tensor)
    assert np.array_equal(deserialized, np.array([[42, 42], [-42, -42]]))


def test_deserialize_tensor_fixed_point():
    serialized_tensor = [2, 2, 2, 8, 2780037,
                         0, 2780037, 0, 2780037, 1, 2780037, 1]
    expected_array = np.array([[42.42, 42.42], [-42.42, -42.42]])
    deserialized = deserialize_tensor_fixed_point(serialized_tensor)
    assert np.allclose(deserialized, expected_array, atol=1e-7)


def test_deserialize_tensor_tuple_tensor_uint():
    serialized = [2, 2, 2, 4, 1, 2, 3, 4, 2, 2, 2, 4, 5, 6, 7, 8]
    deserialized = deserialize_tuple_tensor_uint(serialized)

    assert np.array_equal(deserialized[0], np.array(
        [[1, 2], [3, 4]], dtype=np.int64))
    assert np.array_equal(deserialized[1], np.array(
        [[5, 6], [7, 8]], dtype=np.int64))


def test_deserialize_tensor_tuple_tensor_signed_int():
    serialized = [2, 2, 2, 8, 42,
                  0, 42, 0, 42, 1, 42, 1, 2, 2, 2, 8, 42,
                  0, 42, 0, 42, 1, 42, 1]
    deserialized = deserialize_tuple_tensor_signed_int(serialized)

    expected_array = np.array([[42, 42], [-42, -42]])
    assert np.allclose(deserialized[0], expected_array, atol=1e-7)
    assert np.allclose(deserialized[1], expected_array, atol=1e-7)


def test_deserialize_tensor_tuple_tensor_fixed_point():
    serialized = [2, 2, 2, 8, 2780037,
                  0, 2780037, 0, 2780037, 1, 2780037, 1, 2, 2, 2, 8, 2780037,
                  0, 2780037, 0, 2780037, 1, 2780037, 1]
    deserialized = deserialize_tuple_tensor_fixed_point(serialized)

    expected_array = np.array([[42.42, 42.42], [-42.42, -42.42]])
    assert np.allclose(deserialized[0], expected_array, atol=1e-7)
    assert np.allclose(deserialized[1], expected_array, atol=1e-7)
