import numpy as np
import numpy.testing as npt
from math import isclose

from osiris.cairo.serde.deserialize import *


def test_deserialize_int():
    serialized = '42'
    deserialized = deserializer(serialized, 'u32')
    assert deserialized == 42

    serialized = '3618502788666131213697322783095070105623107215331596699973092056135872020439'
    deserialized = deserializer(serialized, 'i32')
    assert deserialized == -42


def test_deserialize_fp():
    serialized = '2780037 false'
    deserialized = deserializer(serialized, 'FP16x16')
    assert isclose(deserialized, 42.42, rel_tol=1e-7)

    serialized = '2780037 true'
    deserialized = deserializer(serialized, 'FP16x16')
    assert isclose(deserialized, -42.42, rel_tol=1e-7)


def test_deserialize_array_int():
    serialized = '[1 2]'
    deserialized = deserializer(serialized, 'Span<u32>')
    assert np.array_equal(deserialized, np.array([1, 2], dtype=np.int64))

    serialized = '[42 3618502788666131213697322783095070105623107215331596699973092056135872020439]'
    deserialized = deserializer(serialized, 'Span<i32>')
    assert np.array_equal(deserialized, np.array([42, -42], dtype=np.int64))


def test_deserialize_arr_fixed_point():
    serialized = '[2780037 false 2780037 true]'
    deserialized = deserializer(serialized, 'Span<FP16x16>')
    expected = np.array([42.42, -42.42], dtype=np.float64)
    assert np.all(np.isclose(deserialized, expected, atol=1e-7))


def test_deserialize_tensor_int():
    serialized = '[2 2] [1 2 3 4]'
    deserialized = deserializer(serialized, 'Tensor<i32>')
    assert np.array_equal(deserialized, np.array(
        ([1, 2], [3, 4]), dtype=np.int64))

    serialized = '[2 2] [42 42 3618502788666131213697322783095070105623107215331596699973092056135872020439 3618502788666131213697322783095070105623107215331596699973092056135872020439]'
    deserialized = deserializer(serialized, 'Tensor<i32>')
    assert np.array_equal(deserialized, np.array([[42, 42], [-42, -42]]))


def test_deserialize_tensor_fixed_point():
    serialized = '[2 2] [2780037 false 2780037 false 2780037 true 2780037 true]'
    expected_array = np.array([[42.42, 42.42], [-42.42, -42.42]])
    deserialized = deserializer(serialized, 'Tensor<FP16x16>')
    assert np.allclose(deserialized, expected_array, atol=1e-7)

def test_deserialize_matrix_fixed_point():
    serialized = "{0: 2780037 false 2: 2780037 false 1: 2780037 true 3: 2780037 true} 4 2 2"
    expected_array = np.array([[42.42, 42.42], [-42.42, -42.42]])
    deserialized = deserializer(serialized, 'MutMatrix<FP16x16>')
    assert np.allclose(deserialized, expected_array, atol=1e-7)

def test_deserialize_tuple_int():
    serialized = '1 3'
    deserialized = deserializer(serialized, '(u32, u32)')
    assert deserialized == (1, 3)


def test_deserialize_tuple_span():
    serialized = '[1 2] 3'
    deserialized = deserializer(serialized, '(Span<u32>, u32)')
    expected = (np.array([1, 2]), 3)
    npt.assert_array_equal(deserialized[0], expected[0])
    assert deserialized[1] == expected[1]


def test_deserialize_tuple_span_tensor_fp():
    serialized = '[1 2] [2 2] [2780037 false 2780037 false 2780037 true 2780037 true]'
    deserialized = deserializer(serialized, '(Span<u32>, Tensor<FP16x16>)')
    expected = (np.array([1, 2]), np.array([[42.42, 42.42], [-42.42, -42.42]]))
    npt.assert_array_equal(deserialized[0], expected[0])
    assert np.allclose(deserialized[1], expected[1], atol=1e-7)

    serialized = '[2 2] [2780037 false 2780037 false 2780037 true 2780037 true] [1 2]'
    deserialized = deserializer(serialized, '(Tensor<FP16x16>, Span<u32>)')
    expected = (np.array([[42.42, 42.42], [-42.42, -42.42]]), np.array([1, 2]))
    assert np.allclose(deserialized[0], expected[0], atol=1e-7)
    npt.assert_array_equal(deserialized[1], expected[1])

def test_deserialize_tuple_matrix_fp():
    serialized = '[1 2]{0: 2780037 false 2: 2780037 false 1: 2780037 true 3: 2780037 true} 4 2 2'
    deserialized = deserializer(serialized, '(Span<u32>, MutMatrix<FP16x16>)')
    expected = (np.array([1, 2]), np.array([[42.42, 42.42], [-42.42, -42.42]]))
    npt.assert_array_equal(deserialized[0], expected[0])
    assert np.allclose(deserialized[1], expected[1], atol=1e-7)