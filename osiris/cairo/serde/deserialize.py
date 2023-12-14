import numpy as np

from .utils import from_fp


def deserializer(serialized: list, data_type: str, fp_impl='FP16x16'):
    """
    Main deserialization function that handles various data types.

    :param serialized: The serialized list of data.
    :param data_type: The type of data to deserialize ('uint', 'signed_int', 'fixed_point', etc.).
    :param fp_impl: The implementation detail, used for fixed-point deserialization.
    :return: The deserialized data.
    """
    if data_type == 'signed_int':
        return deserialize_signed_int(serialized)
    elif data_type == 'fixed_point':
        return deserialize_fixed_point(serialized, fp_impl)
    elif data_type == 'arr_uint':
        return deserialize_arr_uint(serialized)
    elif data_type == 'arr_signed_int':
        return deserialize_arr_signed_int(serialized)
    elif data_type == 'arr_fixed_point':
        return deserialize_arr_fixed_point(serialized, fp_impl)
    elif data_type == 'tensor_uint':
        return deserialize_tensor_uint(serialized)
    elif data_type == 'tensor_signed_int':
        return deserialize_tensor_signed_int(serialized)
    elif data_type == 'tensor_fixed_point':
        return deserialize_tensor_fixed_point(serialized, fp_impl)
    elif data_type == 'tuple_uint':
        return deserialize_tuple_uint(serialized)
    elif data_type == 'tuple_signed_int':
        return deserialize_tuple_signed_int(serialized)
    elif data_type == 'tuple_fixed_point':
        return deserialize_tuple_fixed_point(serialized, fp_impl)
    elif data_type == 'tuple_tensor_uint':
        return deserialize_tuple_tensor_uint(serialized)
    elif data_type == 'tuple_tensor_signed_int':
        return deserialize_tuple_tensor_signed_int(serialized)
    elif data_type == 'tuple_tensor_fixed_point':
        return deserialize_tuple_tensor_fixed_point(serialized, fp_impl)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

# ================= SIGNED INT =================


def deserialize_signed_int(serialized: list) -> np.int64:
    serialized_mag = serialized[0]
    serialized_sign = serialized[1]

    deserialized = serialized_mag if serialized_sign == 0 else -serialized_mag
    return np.int64(deserialized)

# ================= FIXED POINT =================


def deserialize_fixed_point(serialized: list, impl='FP16x16') -> np.float64:
    serialized_mag = from_fp(serialized[0], impl)
    serialized_sign = serialized[1]

    deserialized = serialized_mag if serialized_sign == 0 else -serialized_mag
    return np.float64(deserialized)

# ================= ARRAY UINT =================


def deserialize_arr_uint(serialized: list) -> np.array:
    return np.array(serialized[1:], dtype=np.int64)

# ================= ARRAY SIGNED INT =================


def deserialize_arr_signed_int(serialized: list) -> np.array:
    num_ele = (len(serialized) - 1) // 2

    deserialized_array = np.empty(num_ele, dtype=np.int64)

    for i in range(num_ele):
        deserialized_array[i] = deserialize_signed_int(
            serialized[1 + i*2: 3 + i*2])

    return deserialized_array

# ================= ARRAY FIXED POINT =================


def deserialize_arr_fixed_point(serialized: list, impl='FP16x16') -> np.array:
    num_ele = (len(serialized) - 1) // 2

    deserialized_array = np.empty(num_ele, dtype=np.float64)

    for i in range(num_ele):
        deserialized_array[i] = deserialize_fixed_point(
            serialized[1 + i*2: 3 + i*2], impl)

    return deserialized_array


# ================= TENSOR UINT =================


def deserialize_tensor_uint(serialized: list) -> np.array:
    num_shape_elements = serialized[0]
    shape = serialized[1:1 + num_shape_elements]
    data = serialized[1 + num_shape_elements + 1:]

    return np.array(data, dtype=np.int64).reshape(shape)

# ================= TENSOR SIGNED INT =================


def deserialize_tensor_signed_int(serialized: list) -> np.array:
    num_shape_elements = serialized[0]
    shape = serialized[1:1 + num_shape_elements]
    data = deserialize_arr_signed_int(
        serialized[1 + num_shape_elements:])

    return data.reshape(shape)


# ================= TENSOR FIXED POINT =================


def deserialize_tensor_fixed_point(serialized: list, impl='FP16x16') -> np.array:
    num_shape_elements = serialized[0]
    shape = serialized[1:1 + num_shape_elements]
    data = deserialize_arr_fixed_point(
        serialized[1 + num_shape_elements:], impl)

    return data.reshape(shape)

# ================= TUPLE UINT =================


def deserialize_tuple_uint(serialized: list):
    return np.array(serialized, dtype=np.int64)


# ================= TUPLE SIGNED INT =================


def deserialize_tuple_signed_int(serialized: list):
    num_ele = (len(serialized)) // 2

    deserialized_array = np.empty(num_ele, dtype=np.int64)

    for i in range(num_ele):
        deserialized_array[i] = deserialize_signed_int(
            serialized[i*2: 3 + i*2])

    return deserialized_array

# ================= TUPLE FIXED POINT =================


def deserialize_tuple_fixed_point(serialized: list, impl='FP16x16'):
    num_ele = (len(serialized)) // 2

    deserialized_array = np.empty(num_ele, dtype=np.float64)

    for i in range(num_ele):
        deserialized_array[i] = deserialize_fixed_point(
            serialized[i*2: 3 + i*2], impl)

    return deserialized_array


# ================= TUPLE TENSOR UINT =================

def deserialize_tuple_tensor_uint(serialized: list):
    return deserialize_tuple_tensor(serialized, deserialize_arr_uint)

# ================= TUPLE TENSOR SIGNED INT =================


def deserialize_tuple_tensor_signed_int(serialized: list):
    return deserialize_tuple_tensor(serialized, deserialize_arr_signed_int)

# ================= TUPLE TENSOR FIXED POINT =================


def deserialize_tuple_tensor_fixed_point(serialized: list, impl='FP16x16'):
    return deserialize_tuple_tensor(serialized, deserialize_arr_fixed_point, impl)


# ================= HELPERS =================


def extract_shape(serialized, start_index):
    """ Extracts the shape part of a tensor from a serialized list. """
    num_shape_elements = serialized[start_index]
    shape = serialized[start_index + 1: start_index + 1 + num_shape_elements]
    return shape, start_index + 1 + num_shape_elements


def extract_data(serialized, start_index, deserialization_func, impl=None):
    """ Extracts and deserializes the data part of a tensor from a serialized list. """
    num_data_elements = serialized[start_index]
    end_index = start_index + 1 + num_data_elements
    data_serialized = serialized[start_index: end_index]
    if impl:
        data = deserialization_func(data_serialized, impl)
    else:
        data = deserialization_func(data_serialized)
    return data, end_index


def deserialize_tuple_tensor(serialized, deserialization_func, impl=None):
    """ Generic deserialization function for a tuple of tensors. """
    deserialized_tensors = []
    i = 0
    while i < len(serialized):
        shape, i = extract_shape(serialized, i)
        data, i = extract_data(serialized, i, deserialization_func, impl)
        tensor = data.reshape(shape)
        deserialized_tensors.append(tensor)
    return tuple(deserialized_tensors)
