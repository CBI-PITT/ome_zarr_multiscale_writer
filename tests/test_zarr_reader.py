import sys
from pathlib import Path

import shutil
import numpy as np
import zarr

# Ensure repository root on sys.path for direct invocation
ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ome_zarr_multiscale_writer.zarr_reader import OmeZarrArray
from ome_zarr_multiscale_writer.write import (
    write_ome_zarr_multiscale,
    generate_multiscales_from_omezarr,
)


def _prepare_store(base: Path, name: str = "sample.ome.zarr") -> Path:
    base.mkdir(parents=True, exist_ok=True)
    store_path = base / name
    if store_path.exists():
        shutil.rmtree(store_path, ignore_errors=True)
    return store_path


def _run_generate_and_validate(store_path: Path):
    data = np.arange(512 * 512 * 512, dtype=np.uint16).reshape((512, 512, 512))

    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=True,
        async_close=False,  # keep test deterministic
        voxel_size=(2.0, 0.5, 0.5),
        start_chunks=(256, 256, 256),
        end_chunks=(64, 64, 64),
    )


def test_open_zarr():
    # Create a small level-0-only dataset inside a repo-local, ignored folder
    store_path = _prepare_store(TEST_DATA_DIR, "reader.ome.zarr")
    _run_generate_and_validate(store_path)

    multiscale_array = OmeZarrArray(str(store_path))
    print(multiscale_array)
    assert multiscale_array.resolution_level == 0
    assert multiscale_array.shape == (512, 512, 512)
    assert multiscale_array.dtype == np.uint16
    assert multiscale_array.ndim == 3
    assert multiscale_array.size == 512 * 512 * 512
    # Test __getitem__ slicing
    slice_data = multiscale_array[0:10, 0:10, 0:10]
    assert slice_data.shape == (10, 10, 10)

    # Test iteration over z-slices
    z_slice_count = 0
    for z_slice in multiscale_array:
        assert z_slice.shape == (512, 512)
        z_slice_count += 1
    assert z_slice_count == 512
