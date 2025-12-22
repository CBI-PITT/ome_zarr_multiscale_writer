#!/usr/bin/env python3
"""
Quick test of OME version consistency.
"""

import tempfile
import numpy as np
from pathlib import Path
import zarr
from ome_zarr_multiscale_writer.zarr_reader import OmeZarrArray
from ome_zarr_multiscale_writer.zarr_tools import Live3DPyramidWriter, PyramidSpec

# Create persistent temp directory
temp_dir = Path(tempfile.mkdtemp())
print(f"Using temp directory: {temp_dir}")

try:
    # Test v0.4 format creation and reading
    print("🏗️  Creating OME-Zarr v0.4 test data...")
    v04_path = temp_dir / "test_v04.ome.zarr"

    t, c, z, y, x = 1, 1, 16, 64, 64
    data = np.random.randint(0, 1000, (z, y, x), dtype=np.uint16)

    spec = PyramidSpec(z_size_estimate=z, y=y, x=x, levels=2)

    # Create v0.4 format
    with Live3DPyramidWriter(
        spec=spec, path=str(v04_path), ome_version="0.4", async_close=False
    ) as writer:
        for z_idx in range(z):
            writer.push_slice(data[z_idx])

    print(f"✅ Created v0.4: {v04_path}")

    # Test reading and version detection
    print("🔍 Testing version detection...")
    reader = OmeZarrArray(str(v04_path))
    print(f"Detected OME version: {reader._ome_version}")

    # Test create_multiscales preserves version
    print("🧪 Testing create_multiscales...")
    output_path = temp_dir / "output.ome.zarr"

    result_path = reader.create_multiscales(target_path=str(output_path), levels=2)
    print(f"✅ Output path: {result_path}")

    # Read output and check version
    output_reader = OmeZarrArray(result_path)
    print(f"Output OME version: {output_reader._ome_version}")

    # Validate
    if reader._ome_version == output_reader._ome_version:
        print("🎉 SUCCESS: OME version preserved!")
    else:
        print(f"❌ FAILURE: {reader._ome_version} -> {output_reader._ome_version}")

finally:
    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)
    print("🧹 Cleaned up temp directory")
