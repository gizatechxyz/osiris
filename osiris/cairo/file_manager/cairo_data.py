from osiris.cairo.file_manager.file import File


class CairoData(File):
    def __init__(self, file: str):
        super().__init__(file)

    @classmethod
    def base_template(cls, func: str, dtype: str, refs: list[str], data: list[str], shape: tuple) -> list[str]:
        """
        Create a base template for data representation in Cairo.

        Args:
            func (str): The function name.
            dtype (str): The data type of the tensor.
            refs (list[str]): A list of module references.
            data (list[str]): The data to be included in the tensor.
            shape (tuple): The shape of the tensor.

        Returns:
            list[str]: A list of strings that together form the template of a data function in Cairo.

        This method generates a list of strings representing a function in Cairo for data handling,
        defining the shape and contents of a tensor.
        """
        template = [
            *[f"use {ref};" for ref in refs],
            *[ ""],
            *[f"fn {func}() -> Tensor<{dtype}>"+" {"],
            *[ "    let mut shape = ArrayTrait::<usize>::new();"],
            *[f"    shape.append({s});" for s in shape],
            *[ ""],
            *[ "    let mut data = ArrayTrait::new();"],
            *[f"    data.append({d});" for d in data],
            *[ "    TensorTrait::new(shape.span(), data.span())"],
            *[ "}"],
        ]

        return template

    @classmethod
    def sequence_template(cls, func: str, dtype: str, refs: list[str], data: list[list[str]], shape: list[tuple]) -> list[str]:
        """
        Create a template for handling tensor sequences in Cairo.

        Args:
            func (str): The function name.
            dtype (str): The data type of the tensor sequence.
            refs (list[str]): A list of module references.
            data (list[list[str]]): The data to be included in each tensor.
            shape (list[tuple]): The shapes of each tensor in the sequence.

        Returns:
            list[str]: A list of strings that together form the template of a sequence tensor function in Cairo.

        This method generates a list of strings representing a function in Cairo for handling a sequence
        of tensors, each with its own data and shape.
        """
        def expand_sequence_init(s: list[tuple], d: list[list[str]]) -> list[str]:
            snippet = []
            for i in range(len(s)):
                snippet += [
                    *[ "    let mut shape = ArrayTrait::<usize>::new();"],
                    *[f"    shape.append({s});" for s in s[i]],
                    *[ ""],
                    *[ "    let mut data = ArrayTrait::new();"],
                    *[f"    data.append({d});" for d in d[i]],
                    *[ ""],
                    *[ "    sequence.append(TensorTrait::new(shape.span(), data.span()));"],
                    *[ ""],
                ]

            return snippet

        template = [
            *[f"use {ref};" for ref in refs],
            *[ ""],
            *[f"fn {func}() -> Array<Tensor<{dtype}>>"+" {"],
            *[ "    let mut sequence = ArrayTrait::new();"],
            *[ ""],
            *expand_sequence_init(shape, data),
            *[ "    sequence"],
            *[ "}"],
        ]

        return template
