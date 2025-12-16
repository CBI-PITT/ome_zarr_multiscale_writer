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


def test_generate_multiscales_from_level0_only():
    # Create a small level-0-only dataset inside a repo-local, ignored folder
    store_path = _prepare_store(TEST_DATA_DIR)
    _run_generate_and_validate(store_path)


def _run_generate_and_validate(store_path: Path):
    data = np.arange(512 * 512 * 512, dtype=np.uint16).reshape((512, 512, 512))

    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=False,
        async_close=False,  # keep test deterministic
        voxel_size=(2.0, 0.5, 0.5),
        start_chunks=(256, 256, 256),
        end_chunks=(64, 64, 64),
    )

    group = zarr.open(store_path, mode="r")
    assert sorted(group.array_keys()) == ["0"]
    original_plane0 = np.array(group["0"][0])

    # Generate multiscales in-place from the existing level 0 data
    generate_multiscales_from_omezarr(
        source_path=store_path,
        async_close=False,
        voxel_size=(2.0, 0.5, 0.5),
        start_chunks=(256, 256, 256),
        end_chunks=(64, 64, 64),
    )

    group = zarr.open(store_path, mode="r")
    assert sorted(group.array_keys()) == ["0", "1", "2", "3"]
    assert group["0"].shape == (512, 512, 512)
    assert group["1"].shape == (512, 256, 256)  # 2x downsampled
    assert group["2"].shape == (512, 128, 128)  # 4x downsampled
    assert group["3"].shape == (256, 64, 64)    # 8x downsampled

    # Level 0 data should remain intact when generating additional scales
    np.testing.assert_array_equal(group["0"][0], original_plane0)

    # Multiscales metadata should now describe two datasets
    ms = group.attrs.get("multiscales") or group.attrs.get("ome", {}).get("multiscales")
    assert ms and len(ms[0]["datasets"]) == 4


def main(base_path: Path | str = TEST_DATA_DIR):
    """Manual test harness: write level-0-only data to the given path and generate multiscales."""

    store_path = _prepare_store(Path(base_path))
    _run_generate_and_validate(store_path)


if __name__ == "__main__":
    main()
