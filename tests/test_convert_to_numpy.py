from osiris.app import convert_to_numpy, load_data
from osiris.dtypes.input_output_formats import InputFormat

import numpy as np


def test_convert_to_numpy_from_csv():
    data = load_data("tests/data/simple_tensor.csv")
    numpy_array = convert_to_numpy(data)
    assert np.array_equal(numpy_array, np.array(
        [[1, 2], [3, 4]], dtype=np.uint32))


def test_convert_to_numpy_from_parquet():
    data = load_data("tests/data/simple_tensor.parquet")
    numpy_array = convert_to_numpy(data)
    assert np.array_equal(numpy_array, np.array(
        [[1, 2], [3, 4]], dtype=np.uint32))


def test_convert_to_numpy_from_npy():
    data = load_data("tests/data/simple_tensor.npy")
    numpy_array = convert_to_numpy(data)
    assert np.array_equal(numpy_array, np.array(
        [[1, 2], [3, 4]], dtype=np.uint32))
