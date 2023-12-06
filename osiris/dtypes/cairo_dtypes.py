from enum import Enum


class Dtype(Enum):
    FP32x32 = "FP32x32"
    FP8x23 = "FP8x23"
    FP16x16 = "FP16x16"
    I8 = "i8"
    I32 = "i32"
    U32 = "u32"
    BOOL = 'bool'
