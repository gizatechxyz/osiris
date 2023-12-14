from osiris.cairo.serde.data_structures import FixedPoint, SignedInt, Tensor


def serializer(data) -> list[str]:
    if isinstance(data, bool):
        return ["1"] if data else ["0"]
    elif isinstance(data, int):
        return [str(data)]
    elif isinstance(data, (list, tuple)):
        serialized_list = [str(len(data))]
        for item in data:
            serialized_list.extend(serializer(item))
        return serialized_list
    elif isinstance(data, dict):
        serialized_dict = [str(len(data))]
        for key, value in data.items():
            serialized_dict.extend(serializer(key))
            serialized_dict.extend(serializer(value))
        return serialized_dict
    elif isinstance(data, Tensor):
        serialized_tensor = serializer(data.shape)
        serialized_tensor.extend(serializer(data.data))
        return serialized_tensor
    elif isinstance(data, (SignedInt, FixedPoint)):
        return [str(data.mag), str(data.sign)]
    else:
        raise ValueError("Unsupported data type for serialization")
