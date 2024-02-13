import json

import numpy as np

from .utils import felt_to_int, from_fp


def deserializer(serialized: str, dtype: str):
    # Check if the serialized data is a string and needs conversion
    if isinstance(serialized, str):
        serialized = convert_data(serialized)

    # Function to deserialize individual elements within a tuple
    def deserialize_element(element, element_type):
        if element_type in ("u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32", "i64", "i128"):
            return deserialize_int(element)
        elif element_type.startswith("FP"):
            return deserialize_fixed_point(element, element_type)
        elif element_type.startswith("Span<") and element_type.endswith(">"):
            inner_type = element_type[5:-1]
            if inner_type.startswith("FP"):
                return deserialize_arr_fixed_point(element, inner_type)
            else:
                return deserialize_arr_int(element)
        elif element_type.startswith("Tensor<") and element_type.endswith(">"):
            inner_type = element_type[7:-1]
            if inner_type.startswith("FP"):
                return deserialize_tensor_fixed_point(element, inner_type)
            else:
                return deserialize_tensor_int(element)
        elif element_type.startswith("(") and element_type.endswith(")"):
            # Recursive call for nested tuples
            return deserializer(element, element_type)
        else:
            raise ValueError(f"Unsupported data type: {element_type}")

    # Handle tuple data type
    if dtype.startswith("(") and dtype.endswith(")"):
        types = dtype[1:-1].split(", ")
        deserialized_elements = []
        i = 0  # Initialize loop counter

        while i < len(serialized):
            ele_type = types[len(deserialized_elements)]

            if ele_type.startswith("Tensor<"):
                # For Tensors, take two elements from serialized (shape and data)
                ele = serialized[i:i+2]
                i += 2
            else:
                # For other types, take one element
                ele = serialized[i]
                i += 1

            if ele_type.startswith("Tensor<"):
                deserialized_elements.append(
                    deserialize_element(ele, ele_type))
            else:
                deserialized_elements.append(
                    deserialize_element([ele], ele_type))

        if len(deserialized_elements) != len(types):
            raise ValueError(
                "Serialized data length does not match tuple length")

        return tuple(deserialized_elements)

    else:
        return deserialize_element(serialized, dtype)


def parse_return_value(return_value):
    """
    Parse a ReturnValue dictionary to extract the integer value or recursively parse an array of ReturnValues (cf: OrionRunner ReturnValues).
    """
    if 'Int' in return_value:
        # Convert hexadecimal string to integer
        return int(return_value['Int'], 16)
    elif 'Array' in return_value:
        # Recursively parse each item in the array
        return [parse_return_value(item) for item in return_value['Array']]
    else:
        raise ValueError("Invalid ReturnValue format")


def convert_data(data):
    """
    Convert the given JSON-like data structure to the desired format.
    """
    parsed_data = json.loads(data)
    result = []
    for item in parsed_data:
        # Parse each item based on its keys
        if 'Array' in item:
            # Process array items
            result.append(parse_return_value(item))
        elif 'Int' in item:
            # Process single int items
            result.append(parse_return_value(item))
        else:
            raise ValueError("Invalid data format")
    return result


# ================= INT =================


def deserialize_int(serialized: list) -> np.int64:
    return np.int64(felt_to_int(serialized[0]))


# ================= FIXED POINT =================


def deserialize_fixed_point(serialized: list, impl='FP16x16') -> np.float64:
    serialized_mag = from_fp(serialized[0], impl)
    serialized_sign = serialized[1]

    deserialized = serialized_mag if serialized_sign == 0 else -serialized_mag
    return np.float64(deserialized)


# ================= ARRAY INT =================


def deserialize_arr_int(serialized):

    serialized = serialized[0]

    deserialized = []
    for ele in serialized:
        deserialized.append(felt_to_int(ele))

    return np.array(deserialized)

# ================= ARRAY FIXED POINT =================


def deserialize_arr_fixed_point(serialized: list, impl='FP16x16'):

    serialized = serialized[0]

    if len(serialized) % 2 != 0:
        raise ValueError("Array length must be even")

    deserialized = []
    for i in range(0, len(serialized), 2):
        mag = serialized[i]
        sign = serialized[i + 1]

        deserialized.append(deserialize_fixed_point([mag, sign], impl))

    return np.array(deserialized)


# ================= TENSOR INT =================


def deserialize_tensor_int(serialized: list) -> np.array:
    shape = serialized[0]
    data = deserialize_arr_int([serialized[1]])

    return np.array(data, dtype=np.int64).reshape(shape)


# ================= TENSOR FIXED POINT =================

def deserialize_tensor_fixed_point(serialized: list, impl='FP16x16') -> np.array:
    shape = serialized[0]
    data = deserialize_arr_fixed_point([serialized[1]], impl)

    return np.array(data, dtype=np.float64).reshape(shape)
