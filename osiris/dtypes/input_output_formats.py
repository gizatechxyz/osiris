from enum import Enum


class InputFormat(Enum):
    CSV = "csv"
    PARQUET = "parquet"
    NUMPY = "numpy"


class OutputFormat(Enum):
    CAIRO = "cairo"
    NUMPY = "numpy"
