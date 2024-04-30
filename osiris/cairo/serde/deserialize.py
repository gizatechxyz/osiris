import re

import numpy as np

from osiris.cairo.serde.utils import felt_to_int, from_fp


def deserializer(serialized, dtype, framework='ONNX_ORION'):

    if dtype in ["u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32", "i64", "i128"]:

        match framework:
            case 'XGB':
                return felt_to_int(int(serialized)) / 100000
            case 'LGBM':
                return felt_to_int(int(serialized)) / 100000
            case _:
                return felt_to_int(int(serialized))

    elif dtype.startswith("FP"):
        return deserialize_fp(serialized)

    elif dtype.startswith('Span<'):
        return deserialize_span(serialized, dtype)

    elif dtype.startswith('Tensor<'):
        return deserialize_tensor(serialized, dtype)

    elif dtype.startswith('MutMatrix<'):
        return deserialize_matrix(serialized, dtype)

    elif dtype.startswith('('):  # Tuple
        return deserialize_tuple(serialized, dtype)

    else:
        raise ValueError(f"Unknown data type: {dtype}")


def deserialize_fp(serialized):
    parts = serialized.split()
    value = from_fp(int(parts[0]))
    if len(parts) > 1 and parts[1] == 'true':  # Check for negative sign
        value = -value
    return value


def deserialize_span(serialized, dtype):
    inner_type = dtype[5:-1]
    elements = serialized[1:-1].split()
    if inner_type.startswith("FP"):
        # For fixed point, elements consist of two parts (value and sign)
        deserialized_elements = [deserializer(' '.join(elements[i:i + 2]), inner_type)
                                 for i in range(0, len(elements), 2)]
        return np.array(deserialized_elements, dtype=np.float64)
    else:
        return np.array([deserializer(e, inner_type) for e in elements], dtype=np.int64)


def deserialize_tensor(serialized, dtype):
    inner_type = dtype[7:-1]
    parts = serialized.split('] [')
    dims = [int(d) for d in parts[0][1:].split()]
    values = parts[1][:-1].split()
    if inner_type.startswith("FP"):
        tensor_data = np.array([deserializer(' '.join(values[i:i + 2]), inner_type)
                                for i in range(0, len(values), 2)])
    else:
        tensor_data = np.array(
            [deserializer(v, inner_type) for v in values])
    return tensor_data.reshape(dims)


def deserialize_tuple(serialized, dtype):
    types = dtype[1:-1].split(', ')
    # Check if there is no space between span and matrix.
    is_no_space = re.search(r']\{', serialized)
    if is_no_space:
        split_index = is_no_space.start() + 1
        part1 = deserializer(serialized[:split_index].strip(), types[0])
        part2 = deserializer(serialized[split_index:].strip(), types[1])
        return part1, part2
    else:
        if 'Tensor' in types[0]:
            tensor_end = find_nth_occurrence(serialized, ']', 2)
            depth = 1
            for i in range(tensor_end, len(serialized)):
                if serialized[i] == '[':
                    depth += 1
                elif serialized[i] == ']':
                    depth -= 1
                    if depth == 0:
                        tensor_end = i + 1
                        break
            part1 = deserializer(serialized[:tensor_end].strip(), types[0])
            part2 = deserializer(serialized[tensor_end:].strip(), types[1])
        else:
            split_index = serialized.find(']') + 2
            part1 = deserializer(serialized[:split_index].strip(), types[0])
            part2 = deserializer(serialized[split_index:].strip(), types[1])
        return part1, part2


def deserialize_matrix(serialized, dtype):

    # Extract inner dtype
    pattern = r"<(.*)>"
    inner_dtype = re.search(pattern, dtype).group(1)

    # Extract the matrix content and shape from the serialized string
    content, shape_str = serialized.split("} ")
    # Last two numbers are the shape
    shape = tuple(map(int, shape_str.split()[-2:]))

    # Use regex to find all occurrences of ': ' followed by any characters until the next ' :' or end of string
    pattern = r': (.*?)(?=\s\d+: |$)'
    elements = re.findall(pattern, content)

    # Deserialize each element using the appropriate deserializer based on dtype
    deserialized_elements = [deserializer(
        element, inner_dtype) for element in elements]

    # Reshape the deserialized elements into a numpy array of the specified shape
    matrix = np.array(deserialized_elements).reshape(shape)
    return matrix


def find_nth_occurrence(string, sub_string, n):
    start_index = string.find(sub_string)
    while start_index >= 0 and n > 1:
        start_index = string.find(sub_string, start_index + 1)
        n -= 1
    return start_index
