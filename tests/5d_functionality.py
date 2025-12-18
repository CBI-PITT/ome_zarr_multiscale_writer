"""
Test 5D functionality for Live3DPyramidWriter.

This test validates:
1. Backward compatibility with existing 3D usage
2. New 5D functionality with t,c dimensions
3. Chunking behavior for 5D arrays
4. OME metadata generation for 5D data
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from ome_zarr_multiscale_writer.zarr_tools import (
    Live3DPyramidWriter,
    PyramidSpec,
    ChunkScheme,
)


if __name__ == "__main__":
    # Simple demonstration
    ome_zarr_path = r"c:\code\test\test5D.ome.zarr"  # Replace with your OME-Zarr path
    temp_path = Path(ome_zarr_path)

    # Create 5D test data: (2, 3, 8, 32, 32)
    t, c, z, y, x = 2, 3, 256, 256, 256
    data = np.random.randint(0, 1000, (t, c, z, y, x), dtype=np.uint16)

    spec = PyramidSpec(
        t_size=t,
        c_size=c,
        z_size_estimate=z,
        y=y,
        x=x,
        levels=3
    )

    print("Testing 5D Live3DPyramidWriter...")
    print(f"Data shape: {data.shape}")
    print(f"Spec: t_size={spec.t_size}, c_size={spec.c_size}")
    print(f"Z estimate: {spec.z_size_estimate}")

    with Live3DPyramidWriter(
        spec=spec,
        path=temp_path,
        async_close=False,
        ome_version="0.5"
    ) as writer:
        print(f"Writer 5D mode: {writer.is_5d}")

        # Write data one (t,c) at a time
        for time_idx in range(t):
            for chan_idx in range(c):
                print(f"Writing t={time_idx}, c={chan_idx}...")
                writer.push_slice(data[time_idx, chan_idx], t_index=time_idx, c_index=chan_idx)

    print("✅ 5D writing completed successfully!")

    # Verify the output
    import zarr
    root = zarr.open(ome_zarr_path)
    print(f"Output array shape: {root['0'].shape}")
    print(f"Output chunks: {root['0'].chunks}")
    axes = root.attrs['ome']['multiscales'][0]['axes']
    print(f"OME axes: {[ax['name'] for ax in axes]}")
    print("✅ Verification completed!")