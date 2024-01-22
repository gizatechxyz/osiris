from osiris.cairo.serde.data_structures import (
    FixedPoint,
    Int,
    Tensor,
)
from osiris.cairo.serde.utils import int_to_felt


def serializer(data):
    if isinstance(data, bool):
        return "1" if data else "0"
    elif isinstance(data, int):
            return f"{int_to_felt(data)}"
    elif isinstance(data, Int):
            return f"{int_to_felt(data.val)}"
    elif isinstance(data, (list, tuple)):
        joined_elements = ' '.join(serializer(e) for e in data)
        return f"[{joined_elements}]"
    elif isinstance(data, Tensor):
        return f"{serializer(data.shape)} {serializer(data.data)}"
    elif isinstance(data, FixedPoint):
        return f"{serializer(data.mag)} {serializer(data.sign)}"

    else:
        raise ValueError("Unsupported data type for serialization")
