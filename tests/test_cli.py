import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ome_zarr_multiscale_writer.write import write_ome_zarr_multiscale


def test_cli_generates_multiscales(tmp_path: Path):
    store_path = tmp_path / "cli_store.ome.zarr"
    data = np.arange(64 * 64 * 64, dtype=np.uint16).reshape((64, 64, 64))

    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=False,
        async_close=False,
        voxel_size=(2.0, 0.5, 0.5),
    )

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

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
    assert sorted(store.array_keys()) == ["0", "1", "2"]
    assert store["0"].shape == (64, 64, 64)
    assert store["1"].shape == (64, 32, 32)
    assert store["2"].shape == (32, 16, 16)
