import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import zarr
import shutil

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ome_zarr_multiscale_writer.write import write_ome_zarr_multiscale


def _prepare_store(base: Path, name: str = "sample.ome.zarr") -> Path:
    base.mkdir(parents=True, exist_ok=True)
    store_path = base / name
    if store_path.exists():
        shutil.rmtree(store_path, ignore_errors=True)
    return store_path


def test_cli_generates_multiscales():
    store_path = _prepare_store(TEST_DATA_DIR, "cli_sample.ome.zarr")
    data = np.arange(512 * 512 * 512, dtype=np.uint16).reshape((512, 512, 512))

    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=False,
        async_close=False,
        voxel_size=(2.0, 0.5, 0.5),
        start_chunks=(256, 256, 256),
        end_chunks=(64, 64, 64),
    )

    group = zarr.open(store_path, mode="r")
    assert sorted(group.array_keys()) == ["0"]
    original_plane0 = np.array(group["0"][0])

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ome_zarr_multiscale_writer.cli",
            "generate",
            str(store_path),
            "--voxel-size",
            "2.0",
            "0.5",
            "0.5",
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout

    store = zarr.open(store_path, mode="r")
    assert sorted(store.array_keys()) == ["0", "1", "2", "3"]
    assert store["0"].shape == (512, 512, 512)
    assert store["1"].shape == (512, 256, 256)  # 2x downsampled
    assert store["2"].shape == (512, 128, 128)  # 4x downsampled
    assert store["3"].shape == (256, 64, 64)  # 8x downsampled

    # Level 0 data should remain intact when generating additional scales
    np.testing.assert_array_equal(store["0"][0], original_plane0)

    # Multiscales metadata should now describe two datasets
    ms = store.attrs.get("multiscales") or store.attrs.get("ome", {}).get("multiscales")
    assert ms and len(ms[0]["datasets"]) == 4
