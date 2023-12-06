from pathlib import Path


class ModFile:
    def __init__(self, path):
        """
        Initialize a ModFile object.

        This method creates a new file with a .cairo extension in the path directory.
        If the directory doesn't exist, it's created. The contents of the file are then read
        into the buffer attribute.
        """
        self.path = Path(f"{path}.cairo")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("r") as f:
            self.buffer = f.readlines()

    def update(self, name: str):
        """
        Update the .cairo file with a new module statement.

        Args:
            name (str): The name of the module to be added.

        This method checks if a module statement for the given name already exists in the buffer.
        If it doesn't, the new module statement is appended to the file.
        """
        statement = f"mod {name};"
        if any(line.startswith(statement) for line in self.buffer):
            # Use generator expression
            return

        with self.path.open("a") as f:
            f.write(f"{statement}\n")


class File:
    def __init__(self, path: str):
        """
        Initialize a File object.

        Args:
            path (str): The file path where the File object will operate.

        This method creates a new file at the specified path. If the file already exists, its
        contents are read into the buffer attribute.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer = []

        if self.path.is_file(): # Use pathlib's is_file method
            with self.path.open("r") as f:
                self.buffer = f.readlines()

    def dump(self):
        """
        Write the contents of the buffer to the file.

        This method writes each line in the buffer to the file, ensuring each line is
        properly terminated with a newline character.
        """
        with self.path.open("w") as f:
            f.writelines([f"{line}\n" for line in self.buffer])
