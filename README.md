# ome_zarr_multiscale_writer

A fast one-shot writer for OME-Zarr multiscale image data, designed for scientific imaging applications requiring efficient storage and access to multi-resolution volumetric datasets.

## Overview

This package provides tools to write multi-scale image data in the OME-Zarr format, which is particularly useful for large biological imaging datasets that benefit from being stored in a format that supports efficient access to different resolutions of the same data.

The library supports both command-line interface and Python API usage for creating OME-Zarr multiscale pyramids from numpy arrays or existing datasets.

## Installation

```bash
pip install .
```

## Usage

### Command Line Interface

The package provides a command-line tool `ome-zarr-multiscale`:

```bash
# Generate multiscales from existing OME-Zarr store
ome-zarr-multiscale generate source.ome.zarr

# Generate multiscales with custom settings
ome-zarr-multiscale generate \
    source.ome.zarr \
    --target output.ome.zarr \
    --voxel-size 2 0.5 0.5 \
    --start-chunks 32 256 256 \
    --end-chunks 256 256 256 \
    --compressor zstd \
    --compression-level 5 \
    --ome-version 0.4
```

### Python API

#### Writing from NumPy Arrays

```python
import numpy as np
from ome_zarr_multiscale_writer.write import write_ome_zarr_multiscale

# Create sample data (Z, Y, X)
data = np.random.randint(0, 65535, size=(2000, 2000, 2000), dtype=np.uint16)

# Write to OME-Zarr multiscale pyramid
write_ome_zarr_multiscale(
    data=data,
    path="output.ome.zarr",
    voxel_size=(2, 0.5, 0.5),  # physical voxel size in microns
    generate_multiscales=True,
    start_chunks=(32, 256, 256),  # chunk shape for level 0
    end_chunks=(256, 256, 256),   # chunk shape for coarsest level
    compressor='zstd',
    compression_level=5,
    ome_version="0.4"
)
```

#### Generating Multiscales from Existing Store

```python
from ome_zarr_multiscale_writer.write import generate_multiscales_from_omezarr

# Generate multiscales for an existing dataset
generate_multiscales_from_omezarr(
    source_path="source.ome.zarr",
    target_path="output.ome.zarr",
    voxel_size=(2, 0.5, 0.5),
    start_chunks=(32, 256, 256),
    end_chunks=(256, 256, 256)
)
```

## Features

- Fast, one-shot writing of OME-Zarr multiscale pyramids
- Support for various compression algorithms (zstd, lz4, etc.)
- Configurable chunking strategies
- Multi-threaded processing for efficient writing
- Support for Zarr v3 sharding
- Compatible with OME-NGFF metadata standards (versions 0.4 and 0.5)

## Requirements

- Python 3.12+
- zarr
- typer
- numpy

## License

BSD-3-Clause

## Documentation

See the [GitHub repository](https://github.com/CBI-PITT/ome_zarr_multiscale_writer) for more information and usage examples.