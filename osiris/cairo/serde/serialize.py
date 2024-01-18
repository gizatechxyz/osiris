from osiris.cairo.serde.data_structures import (
    FixedPoint,
    SignedInt,
    Tensor,
    UnsignedInt,
)


def serializer(data):
    if isinstance(data, bool):
        return "1" if data else "0"
    elif isinstance(data, int):
        if data >= 0:
            return f"{data}"
        else:
            raise ValueError("Native signed integers are not supported yet")
            # TODO: Support native singned-int
    elif isinstance(data, (list, tuple)):
        joined_elements = ' '.join(serializer(e) for e in data)
        return f"[{joined_elements}]"
    elif isinstance(data, Tensor):
        return f"{serializer(data.shape)} {serializer(data.data)}"
    elif isinstance(data, (SignedInt, FixedPoint)):
        return f"{serializer(data.mag)} {serializer(data.sign)}"
    elif isinstance(data, UnsignedInt):
        return f"{data.mag}"

    else:
        raise ValueError("Unsupported data type for serialization")
