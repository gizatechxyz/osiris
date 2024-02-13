import numpy as np
import numpy.testing as npt
import pytest
from math import isclose

from osiris.cairo.serde.deserialize import *


def test_deserialize_int():
    serialized = '[{"Int":"2A"}]'
    deserialized = deserializer(serialized, 'u32')
    assert deserialized == 42

    serialized = '[{"Int":"800000000000010FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFD7"}]'
    deserialized = deserializer(serialized, 'i32')
    assert deserialized == -42


def test_deserialize_fp():
    serialized = '[{"Int":"2A6B85"}, {"Int":"0"}]'
    deserialized = deserializer(serialized, 'FP16x16')
    assert isclose(deserialized, 42.42, rel_tol=1e-7)

    serialized = '[{"Int":"2A6B85"}, {"Int":"1"}]'
    deserialized = deserializer(serialized, 'FP16x16')
    assert isclose(deserialized, -42.42, rel_tol=1e-7)


def test_deserialize_array_int():
    serialized = '[{"Array": [{"Int": "0x1"}, {"Int": "0x2"}]}]'
    deserialized = deserializer(serialized, 'Span<u32>')
    assert np.array_equal(deserialized, np.array([1, 2], dtype=np.int64))

    serialized = '[{"Array": [{"Int": "2A"}, {"Int": "800000000000010FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFD7"}]}]'
    deserialized = deserializer(serialized, 'Span<i32>')
    assert np.array_equal(deserialized, np.array([42, -42], dtype=np.int64))


def test_deserialize_arr_fixed_point():
    serialized = '[{"Array": [{"Int": "2A6B85"}, {"Int": "0"}, {"Int": "2A6B85"}, {"Int": "0x1"}]}]'
    deserialized = deserializer(serialized, 'Span<FP16x16>')
    expected = np.array([42.42, -42.42], dtype=np.float64)
    assert np.all(np.isclose(deserialized, expected, atol=1e-7))


def test_deserialize_tensor_int():
    serialized = '[{"Array": [{"Int": "0x2"}, {"Int": "0x2"}]}, {"Array": [{"Int": "0x1"}, {"Int": "0x2"}, {"Int": "0x3"}, {"Int": "0x4"}]}]'
    deserialized = deserializer(serialized, 'Tensor<i32>')
    assert np.array_equal(deserialized, np.array(
        ([1, 2], [3, 4]), dtype=np.int64))

    serialized = '[{"Array": [{"Int": "0x2"}, {"Int": "0x2"}]}, {"Array": [{"Int": "2A"}, {"Int": "2A"},{"Int": "800000000000010FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFD7"}, {"Int": "800000000000010FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFD7"}]}]'
    deserialized = deserializer(serialized, 'Tensor<i32>')
    assert np.array_equal(deserialized, np.array([[42, 42], [-42, -42]]))


def test_deserialize_tensor_fixed_point():
    serialized = '[{"Array": [{"Int": "0x2"}, {"Int": "0x2"}]}, {"Array": [{"Int": "2A6B85"}, {"Int": "0x0"}, {"Int": "2A6B85"}, {"Int": "0x0"}, {"Int": "2A6B85"}, {"Int": "0x1"}, {"Int": "2A6B85"}, {"Int": "0x1"}]}]'
    expected_array = np.array([[42.42, 42.42], [-42.42, -42.42]])
    deserialized = deserializer(serialized, 'Tensor<FP16x16>')
    assert np.allclose(deserialized, expected_array, atol=1e-7)


def test_deserialize_tuple_int():
    serialized = '[{"Int":"0x1"},{"Int":"0x3"}]'
    deserialized = deserializer(serialized, '(u32, u32)')
    assert deserialized == (1, 3)


def test_deserialize_tuple_span():
    serialized = '[{"Array":[{"Int":"0x1"},{"Int":"0x2"}]},{"Int":"0x3"}]'
    deserialized = deserializer(serialized, '(Span<u32>, u32)')
    expected = (np.array([1, 2]), 3)
    npt.assert_array_equal(deserialized[0], expected[0])
    assert deserialized[1] == expected[1]


def test_deserialize_tuple_span_tensor_fp():
    serialized = '[{"Array":[{"Int":"0x1"},{"Int":"0x2"}]},{"Array": [{"Int": "0x2"}, {"Int": "0x2"}]}, {"Array": [{"Int": "2A6B85"}, {"Int": "0x0"}, {"Int": "2A6B85"}, {"Int": "0x0"}, {"Int": "2A6B85"}, {"Int": "0x1"}, {"Int": "2A6B85"}, {"Int": "0x1"}]}]'
    deserialized = deserializer(serialized, '(Span<u32>, Tensor<FP16x16>)')
    expected = (np.array([1, 2]), np.array([[42.42, 42.42], [-42.42, -42.42]]))
    npt.assert_array_equal(deserialized[0], expected[0])
    assert np.allclose(deserialized[1], expected[1], atol=1e-7)

    serialized = '[{"Array": [{"Int": "0x2"}, {"Int": "0x2"}]}, {"Array": [{"Int": "2A6B85"}, {"Int": "0x0"}, {"Int": "2A6B85"}, {"Int": "0x0"}, {"Int": "2A6B85"}, {"Int": "0x1"}, {"Int": "2A6B85"}, {"Int": "0x1"}]}, {"Array":[{"Int":"0x1"},{"Int":"0x2"}]}]'
    deserialized = deserializer(serialized, '(Tensor<FP16x16>, Span<u32>)')
    expected = (np.array([[42.42, 42.42], [-42.42, -42.42]]), np.array([1, 2]))
    assert np.allclose(deserialized[0], expected[0], atol=1e-7)
    npt.assert_array_equal(deserialized[1], expected[1])
