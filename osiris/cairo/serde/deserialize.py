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
            tensor_end = serialized.find(']') + 2  # Find the end of the Tensor definition
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

# import json

# import numpy as np

# from .utils import felt_to_int, from_fp


# def deserializer(serialized: str, dtype: str):
#     # Check if the serialized data is a string and needs conversion
#     if isinstance(serialized, str):
#         serialized = convert_data(serialized)

#     # Function to deserialize individual elements within a tuple
#     def deserialize_element(element, element_type):
#         if element_type in ("u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32", "i64", "i128"):
#             return deserialize_int(element)
#         elif element_type.startswith("FP"):
#             return deserialize_fixed_point(element, element_type)
#         elif element_type.startswith("Span<") and element_type.endswith(">"):
#             inner_type = element_type[5:-1]
#             if inner_type.startswith("FP"):
#                 return deserialize_arr_fixed_point(element, inner_type)
#             else:
#                 return deserialize_arr_int(element)
#         elif element_type.startswith("Tensor<") and element_type.endswith(">"):
#             inner_type = element_type[7:-1]
#             if inner_type.startswith("FP"):
#                 return deserialize_tensor_fixed_point(element, inner_type)
#             else:
#                 return deserialize_tensor_int(element)
#         elif element_type.startswith("(") and element_type.endswith(")"):
#             # Recursive call for nested tuples
#             return deserializer(element, element_type)
#         else:
#             raise ValueError(f"Unsupported data type: {element_type}")

#     # Handle tuple data type
#     if dtype.startswith("(") and dtype.endswith(")"):
#         types = dtype[1:-1].split(", ")
#         deserialized_elements = []
#         i = 0  # Initialize loop counter

#         while i < len(serialized):
#             ele_type = types[len(deserialized_elements)]

#             if ele_type.startswith("Tensor<"):
#                 # For Tensors, take two elements from serialized (shape and data)
#                 ele = serialized[i:i+2]
#                 i += 2
#             else:
#                 # For other types, take one element
#                 ele = serialized[i]
#                 i += 1

#             if ele_type.startswith("Tensor<"):
#                 deserialized_elements.append(
#                     deserialize_element(ele, ele_type))
#             else:
#                 deserialized_elements.append(
#                     deserialize_element([ele], ele_type))

#         if len(deserialized_elements) != len(types):
#             raise ValueError(
#                 "Serialized data length does not match tuple length")

#         return tuple(deserialized_elements)

#     else:
#         return deserialize_element(serialized, dtype)


# def parse_return_value(return_value):
#     """
#     Parse a ReturnValue dictionary to extract the integer value or recursively parse an array of ReturnValues (cf: OrionRunner ReturnValues).
#     """
#     if 'Int' in return_value:
#         # Convert hexadecimal string to integer
#         return int(return_value['Int'], 16)
#     elif 'Array' in return_value:
#         # Recursively parse each item in the array
#         return [parse_return_value(item) for item in return_value['Array']]
#     else:
#         raise ValueError("Invalid ReturnValue format")


# def convert_data(data):
#     """
#     Convert the given JSON-like data structure to the desired format.
#     """
#     parsed_data = json.loads(data)
#     result = []
#     for item in parsed_data:
#         # Parse each item based on its keys
#         if 'Array' in item:
#             # Process array items
#             result.append(parse_return_value(item))
#         elif 'Int' in item:
#             # Process single int items
#             result.append(parse_return_value(item))
#         else:
#             raise ValueError("Invalid data format")
#     return result


# # ================= INT =================


# def deserialize_int(serialized: list) -> np.int64:
#     return np.int64(felt_to_int(serialized[0]))


# # ================= FIXED POINT =================


# def deserialize_fixed_point(serialized: list, impl='FP16x16') -> np.float64:
#     serialized_mag = from_fp(serialized[0], impl)
#     serialized_sign = serialized[1]

#     deserialized = serialized_mag if serialized_sign == 0 else -serialized_mag
#     return np.float64(deserialized)


# # ================= ARRAY INT =================


# def deserialize_arr_int(serialized):

#     serialized = serialized[0]

#     deserialized = []
#     for ele in serialized:
#         deserialized.append(felt_to_int(ele))

#     return np.array(deserialized)

# # ================= ARRAY FIXED POINT =================


# def deserialize_arr_fixed_point(serialized: list, impl='FP16x16'):

#     serialized = serialized[0]

#     if len(serialized) % 2 != 0:
#         raise ValueError("Array length must be even")

#     deserialized = []
#     for i in range(0, len(serialized), 2):
#         mag = serialized[i]
#         sign = serialized[i + 1]

#         deserialized.append(deserialize_fixed_point([mag, sign], impl))

#     return np.array(deserialized)


# # ================= TENSOR INT =================


# def deserialize_tensor_int(serialized: list) -> np.array:
#     shape = serialized[0]
#     data = deserialize_arr_int([serialized[1]])

#     return np.array(data, dtype=np.int64).reshape(shape)


# # ================= TENSOR FIXED POINT =================

# def deserialize_tensor_fixed_point(serialized: list, impl='FP16x16') -> np.array:
#     shape = serialized[0]
#     data = deserialize_arr_fixed_point([serialized[1]], impl)

#     return np.array(data, dtype=np.float64).reshape(shape)
