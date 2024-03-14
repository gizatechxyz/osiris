import numpy as np
from math import isclose


def deserializer(serialized, dtype):
    if dtype in ['u32', 'i32']:
        return int(serialized)

    elif dtype == 'FP16x16':
        parts = serialized.split()
        value = int(parts[0]) / 2**16
        if len(parts) > 1 and parts[1] == '1':  # Check for negative sign
            value = -value
        return value

    elif dtype.startswith('Span<'):
        inner_type = dtype[5:-1]
        if 'FP16x16' in inner_type:
            # For FP16x16, elements consist of two parts (value and sign)
            elements = serialized[1:-1].split()
            deserialized_elements = []
            for i in range(0, len(elements), 2):
                element = ' '.join(elements[i:i+2])
                deserialized_elements.append(deserializer(element, inner_type))
            return np.array(deserialized_elements, dtype=np.float64)
        else:
            elements = serialized[1:-1].split()
            return np.array([deserializer(e, inner_type) for e in elements], dtype=np.int64)

    elif dtype.startswith('Tensor<'):
        inner_type = dtype[7:-1]
        parts = serialized.split('] [')
        dims = [int(d) for d in parts[0][1:].split()]
        if 'FP16x16' in inner_type:
            values = parts[1][:-1].split()  # Split the values normally first
            # Now, process every two items (value and sign) as one FP16x16 element
            tensor_data = np.array([deserializer(
                ' '.join(values[i:i+2]), inner_type) for i in range(0, len(values), 2)])
        else:
            values = parts[1][:-1].split()
            tensor_data = np.array(
                [deserializer(v, inner_type) for v in values])
        return tensor_data.reshape(dims)

    elif dtype.startswith('('):  # Tuple
        types = dtype[1:-1].split(', ')
        if 'Tensor' in types[0]:  # Handling Tensor as the first element in the tuple
            # Find the end of the Tensor definition
            tensor_end = find_nth_occurrence(serialized, ']', 2)

            # Handle cases where there might be nested arrays or tensors
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

    else:
        raise ValueError(f"Unknown data type: {dtype}")


def find_nth_occurrence(string, sub_string, n):
    start_index = string.find(sub_string)
    while start_index >= 0 and n > 1:
        start_index = string.find(sub_string, start_index + 1)
        n -= 1
    return start_index
