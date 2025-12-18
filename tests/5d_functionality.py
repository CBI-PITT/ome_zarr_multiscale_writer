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

    # Create data where each (t,c) combination has a different value evenly spread across 16 bits
    # Total combinations: t * c = 2 * 3 = 6
    # Spread values evenly across uint16 range: 0, 10922, 21845, 32768, 43690, 54613
    max_uint16 = 65535
    values_per_combination = max_uint16 // (t * c)

    data = np.zeros((t, c, z, y, x), dtype=np.uint16)
    for time_idx in range(t):
        for chan_idx in range(c):
            # Calculate unique value for this (t,c) combination
            value = (time_idx * c + chan_idx) * values_per_combination
            # Fill entire (z,y,x) volume for this (t,c) with its unique value
            data[time_idx, chan_idx, :, :, :] = value

    print(f"Created test data with values per (t,c):")
    for time_idx in range(t):
        for chan_idx in range(c):
            value = (time_idx * c + chan_idx) * values_per_combination
            print(f"  t={time_idx}, c={chan_idx}: value={value}")

    spec = PyramidSpec(t_size=t, c_size=c, z_size_estimate=z, y=y, x=x, levels=3)

    print("Testing 5D Live3DPyramidWriter...")
    print(f"Data shape: {data.shape}")
    print(f"Spec: t_size={spec.t_size}, c_size={spec.c_size}")
    print(f"Z estimate: {spec.z_size_estimate}")

    with Live3DPyramidWriter(
        spec=spec, path=temp_path, async_close=False, ome_version="0.5"
    ) as writer:
        print(f"Writer 5D mode: {writer.is_5d}")

        # Write data one (t,c) at a time
        for time_idx in range(t):
            for chan_idx in range(c):
                print(f"Writing t={time_idx}, c={chan_idx}...")
                writer.push_slice(
                    data[time_idx, chan_idx], t_index=time_idx, c_index=chan_idx
                )

    print("✅ 5D writing completed successfully!")

    # Verify the output
    import zarr

    if Path(ome_zarr_path).exists():
        root = zarr.open(ome_zarr_path)
        if "0" in root:
            arr = root["0"]
            print(f"Output array shape: {arr.shape}")
            print(f"Output chunks: {arr.chunks}")

            # Verify each (t,c) combination has the correct unique value
            print("Verifying data integrity...")
            for time_idx in range(t):
                for chan_idx in range(c):
                    expected_value = (time_idx * c + chan_idx) * values_per_combination
                    # Sample a few points to verify correctness
                    sample_slice = arr[time_idx, chan_idx, 0, 0, 0]
                    if sample_slice == expected_value:
                        print(
                            f"  ✅ t={time_idx}, c={chan_idx}: value={sample_slice} (expected)"
                        )
                    else:
                        print(
                            f"  ❌ t={time_idx}, c={chan_idx}: value={sample_slice} (expected {expected_value})"
                        )

            axes = root.attrs["ome"]["multiscales"][0]["axes"]
            print(f"OME axes: {[ax['name'] for ax in axes]}")
            print("✅ Verification completed!")
        else:
            print("❌ No array '0' found in output")
    else:
        print(f"❌ Output file not found at {ome_zarr_path}")

    from ome_zarr_multiscale_writer.zarr_reader import OmeZarrArray
    array = OmeZarrArray(str(ome_zarr_path))
