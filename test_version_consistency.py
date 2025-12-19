#!/usr/bin/env python3
"""
Test OME version consistency in create_multiscales.

This test ensures that create_multiscales() preserves the original
OME-Zarr version instead of always defaulting to 0.5.
"""

import tempfile
import numpy as np
from pathlib import Path
from ome_zarr_multiscale_writer.zarr_reader import OmeZarrArray
from ome_zarr_multiscale_writer.zarr_tools import Live3DPyramidWriter, PyramidSpec


def create_ome_v04_data():
    """Create OME-Zarr v0.4 format data."""
    print("🏗️  Creating OME-Zarr v0.4 test data...")

    with tempfile.TemporaryDirectory() as temp_dir:
        zarr_path = Path(temp_dir) / "test_v04.ome.zarr"

        # Create v0.4 format using Live3DPyramidWriter directly
        t, c, z, y, x = 1, 1, 32, 128, 128
        data = np.random.randint(0, 1000, (z, y, x), dtype=np.uint16)

        spec = PyramidSpec(z_size_estimate=z, y=y, x=x, levels=2)

        # Create v0.4 format
        with Live3DPyramidWriter(
            spec=spec, path=str(zarr_path), ome_version="0.4", async_close=False
        ) as writer:
            for z_idx in range(z):
                writer.push_slice(data[z_idx])

        return str(zarr_path)


def test_version_consistency():
    """Test that create_multiscales preserves OME version."""

    print("🔍 Testing OME Version Consistency")
    print("=" * 50)

    # Test v0.4 input
    v04_path = create_ome_v04_data()
    print(f"✅ Created v0.4 test data: {v04_path}")

    # Create v0.5 test data for comparison
    with tempfile.TemporaryDirectory() as temp_dir:
        v05_path = Path(temp_dir) / "test_v05.ome.zarr"

        t, c, z, y, x = 1, 1, 32, 128, 128
        data = np.random.randint(0, 1000, (z, y, x), dtype=np.uint16)

        spec = PyramidSpec(z_size_estimate=z, y=y, x=x, levels=2)

        # Create v0.5 format
        with Live3DPyramidWriter(
            spec=spec, path=str(v05_path), ome_version="0.5", async_close=False
        ) as writer:
            for z_idx in range(z):
                writer.push_slice(data[z_idx])

        print(f"✅ Created v0.5 test data: {v05_path}")

        # Test create_multiscales on both versions
        print("\n🧪 Testing create_multiscales()...")

        # Test v0.4 -> should produce v0.4 output
        with tempfile.TemporaryDirectory() as output_dir:
            v04_output = Path(output_dir) / "output_v04.ome.zarr"

            reader_v04 = OmeZarrArray(v04_path)
            print(f"📖 Input v0.4 detected OME version: {reader_v04._ome_version}")

            output_path = reader_v04.create_multiscales(
                target_path=str(v04_output), levels=2
            )

            reader_output_v04 = OmeZarrArray(output_path)
            print(f"📤 Output OME version: {reader_output_v04._ome_version}")

            # Test v0.5 -> should produce v0.5 output
            v05_output = Path(output_dir) / "output_v05.ome.zarr"

            reader_v05 = OmeZarrArray(v05_path)
            print(f"📖 Input v0.5 detected OME version: {reader_v05._ome_version}")

            output_path_v05 = reader_v05.create_multiscales(
                target_path=str(v05_output), levels=2
            )

            reader_output_v05 = OmeZarrArray(output_path_v05)
            print(f"📤 Output OME version: {reader_output_v05._ome_version}")

            # Validate results
            print("\n🎯 Validation Results:")
            v04_consistent = reader_v04._ome_version == reader_output_v04._ome_version
            v05_consistent = reader_v05._ome_version == reader_output_v05._ome_version

            print(
                f"  ✅ v0.4 consistency: {v04_consistent} ({reader_v04._ome_version} -> {reader_output_v04._ome_version})"
            )
            print(
                f"  ✅ v0.5 consistency: {v05_consistent} ({reader_v05._ome_version} -> {reader_output_v05._ome_version})"
            )

            if v04_consistent and v05_consistent:
                print("🎉 SUCCESS: OME versions are preserved correctly!")
                return True
            else:
                print("❌ FAILURE: OME version inconsistency detected!")
                return False


if __name__ == "__main__":
    success = test_version_consistency()
    exit(0 if success else 1)
