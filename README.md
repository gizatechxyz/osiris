# Osiris

Osiris is a Python library for data conversion and management. It is designed to facilitate the transformation of data into Cairo programs. The library supports various input and output formats including CSV, Parquet, and Numpy, with a primary focus on generating Cairo files that represent the data. This makes the data easily consumable in Cairo programs, providing an efficient and optimized process for data conversion.

## Installation

Use the package manager [Poetry](https://python-poetry.org/docs/) to install osiris.

```bash
poetry add giza-osiris
```

## Local Development

To setup local development environment, clone the repository and install dependencies using Poetry.

```bash
git clone https://github.com/gizatechxyz/osiris.git
cd osiris
poetry install
```

## Features

- Supports various input and output formats including CSV, Parquet, and Numpy. The primary focus is on generating Cairo files that represent the data.
- Efficient data conversion and management. The conversion process is optimized for transforming data into Cairo format, which makes the data easily consumable in Cairo programs.

## Usage

### Python API

```python
from osiris import convert
convert(input_file='data.csv', output_file='data.cairo', input_format='csv', output_format='cairo')
```

### CLI

```bash
osiris examples/preprocessed_image.npy data.cairo --input-format numpy --output-format cairo
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
