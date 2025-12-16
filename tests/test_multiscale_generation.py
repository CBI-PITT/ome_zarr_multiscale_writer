import sys
from pathlib import Path

import shutil
import numpy as np
import zarr

# Ensure repository root on sys.path for direct invocation
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ome_zarr_multiscale_writer.write import (
    write_ome_zarr_multiscale,
    generate_multiscales_from_omezarr,
)


def test_generate_multiscales_from_level0_only(tmp_path):
    # Create a small level-0-only dataset
    base = Path(tmp_path)
    base.mkdir(parents=True, exist_ok=True)
    store_path = base / "sample.ome.zarr"
    if store_path.exists():
        shutil.rmtree(store_path, ignore_errors=True)
    data = np.arange(256 * 256 * 256, dtype=np.uint16).reshape((256, 256, 256))

    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=False,
        async_close=False,  # keep test deterministic
    )

    group = zarr.open(store_path, mode="r")
    assert sorted(group.array_keys()) == ["0"]
    original_plane0 = np.array(group["0"][0])

    # Generate multiscales in-place from the existing level 0 data
    generate_multiscales_from_omezarr(
        source_path=store_path,
        async_close=False,
    )

    group = zarr.open(store_path, mode="r")
    assert sorted(group.array_keys()) == ["0", "1", "2"]
    assert group["0"].shape == (256, 256, 256)
    assert group["1"].shape == (128, 128, 128)  # 2x downsampled
    assert group["2"].shape == (64, 64, 64)  # 4x downsampled

    # Level 0 data should remain intact when generating additional scales
    np.testing.assert_array_equal(group["0"][0], original_plane0)

    # Multiscales metadata should now describe two datasets
    ms = group.attrs.get("multiscales") or group.attrs.get("ome", {}).get("multiscales")
    assert ms and len(ms[0]["datasets"]) == 3


def main(base_path="z:/Acquire/MesoSPIM/alan-test/"):
    """Manual test harness: write level-0-only data to the given path and generate multiscales."""

    test_generate_multiscales_from_level0_only(base_path)


if __name__ == "__main__":
    main()
