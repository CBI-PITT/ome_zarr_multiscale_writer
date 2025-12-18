#!/usr/bin/env python3
"""
Demonstration of 5D Live3DPyramidWriter functionality.

This script shows how to write (t,c,z,y,x) data using the new 5D features
while maintaining backward compatibility with existing 3D workflows.
"""

import numpy as np
from pathlib import Path
import tempfile
from ome_zarr_multiscale_writer.zarr_tools import PyramidSpec, Live3DPyramidWriter
import zarr


def demo_5d_functionality():
    """Demonstrate 5D (t,c,z,y,x) writing."""
    print("🚀 Demonstrating 5D Live3DPyramidWriter")
    print("=" * 50)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Define 5D data dimensions
        t_size, c_size, z_size, y_size, x_size = 2, 3, 8, 64, 64
        print(
            f"📊 Creating 5D test data: (t={t_size}, c={c_size}, z={z_size}, y={y_size}, x={x_size})"
        )

        # Generate synthetic 5D data
        data = np.random.randint(
            100, 5000, (t_size, c_size, z_size, y_size, x_size), dtype=np.uint16
        )
        print(f"📦 Data shape: {data.shape}, dtype: {data.dtype}")

        # Create 5D pyramid specification
        spec = PyramidSpec(
            t_size=t_size,
            c_size=c_size,
            z_size_estimate=z_size,
            y=y_size,
            x=x_size,
            levels=4,
        )
        print(f"📋 PyramidSpec: t_size={spec.t_size}, c_size={spec.c_size}")
        print(f"   Z estimate: {spec.z_size_estimate}, Y: {spec.y}, X: {spec.x}")
        print(f"   Pyramid levels: {spec.levels}")

        # Write 5D data
        output_path = temp_path / "demo_5d_output.ome.zarr"
        print(f"💾 Writing to: {output_path.name}")

        with Live3DPyramidWriter(
            spec=spec, path=output_path, async_close=False, ome_version="0.5"
        ) as writer:
            print(f"🔧 Writer 5D mode: {writer.is_5d}")

            # Write data one (t,c) pair at a time as requested
            for t in range(t_size):
                for c in range(c_size):
                    print(f"   ⏳ Processing t={t}, c={c}...")

                    # Push 3D (z,y,x) slice for specific (t,c)
                    writer.push_slice(data[t, c], t_index=t, c_index=c)

        print("✅ 5D writing completed!")

        # Verify and display results
        verify_5d_output(output_path, t_size, c_size, z_size, y_size, x_size)


def demo_backward_compatibility():
    """Demonstrate backward compatibility with existing 3D workflows."""
    print("\n🔄 Demonstrating Backward Compatibility")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Define 3D data dimensions (legacy mode)
        z_size, y_size, x_size = 12, 64, 64
        print(f"📊 Creating 3D test data: (z={z_size}, y={y_size}, x={x_size})")

        # Generate synthetic 3D data
        data = np.random.randint(100, 5000, (z_size, y_size, x_size), dtype=np.uint16)
        print(f"📦 Data shape: {data.shape}, dtype: {data.dtype}")

        # Create 3D pyramid specification (uses defaults t_size=1, c_size=1)
        spec = PyramidSpec(z_size_estimate=z_size, y=y_size, x=x_size, levels=3)
        print(f"📋 PyramidSpec: t_size={spec.t_size}, c_size={spec.c_size} (defaults)")

        # Write 3D data using traditional workflow
        output_path = temp_path / "demo_3d_output.ome.zarr"
        print(f"💾 Writing to: {output_path.name}")

        with Live3DPyramidWriter(
            spec=spec, path=output_path, async_close=False, ome_version="0.5"
        ) as writer:
            print(f"🔧 Writer 5D mode: {writer.is_5d} (should be False)")

            # Traditional usage: push 2D slices one by one
            for z in range(z_size):
                writer.push_slice(data[z])

        print("✅ 3D writing completed!")

        # Verify and display results
        verify_3d_output(output_path, z_size, y_size, x_size)


def verify_5d_output(output_path, t_size, c_size, z_size, y_size, x_size):
    """Verify 5D OME-Zarr output."""
    print("\n🔍 Verifying 5D Output:")
    print("-" * 30)

    root = zarr.open(output_path)

    # Check OME metadata
    axes = root.attrs["ome"]["multiscales"][0]["axes"]
    axis_names = [ax["name"] for ax in axes]
    print(f"📐 OME Axes: {axis_names}")
    assert axis_names == ["t", "c", "z", "y", "x"], (
        f"Expected 5D axes, got {axis_names}"
    )

    # Check array shapes and chunks
    for level in range(4):
        arr = root[str(level)]
        print(f"   Level {level}: shape={arr.shape}, chunks={arr.chunks}")

        # Verify 5D structure
        assert len(arr.shape) == 5, f"Expected 5D array, got {len(arr.shape)}D"
        assert arr.shape[0] == t_size, (
            f"t dimension mismatch: {arr.shape[0]} != {t_size}"
        )
        assert arr.shape[1] == c_size, (
            f"c dimension mismatch: {arr.shape[1]} != {c_size}"
        )

        # Verify chunking (t and c should be 1)
        assert arr.chunks[0] == 1, f"t chunk size should be 1, got {arr.chunks[0]}"
        assert arr.chunks[1] == 1, f"c chunk size should be 1, got {arr.chunks[1]}"

    print("✅ 5D verification passed!")


def verify_3d_output(output_path, z_size, y_size, x_size):
    """Verify 3D OME-Zarr output."""
    print("\n🔍 Verifying 3D Output:")
    print("-" * 30)

    root = zarr.open(output_path)

    # Check OME metadata
    axes = root.attrs["ome"]["multiscales"][0]["axes"]
    axis_names = [ax["name"] for ax in axes]
    print(f"📐 OME Axes: {axis_names}")
    assert axis_names == ["z", "y", "x"], f"Expected 3D axes, got {axis_names}"

    # Check array shapes and chunks
    for level in range(3):
        arr = root[str(level)]
        print(f"   Level {level}: shape={arr.shape}, chunks={arr.chunks}")

        # Verify 3D structure
        assert len(arr.shape) == 3, f"Expected 3D array, got {len(arr.shape)}D"

    print("✅ 3D verification passed!")


def main():
    """Main demonstration."""
    print("🧬 Live3DPyramidWriter 5D Extension Demo")
    print("This demonstrates the new (t,c,z,y,x) functionality")
    print("while maintaining full backward compatibility.")

    # Demonstrate new 5D functionality
    demo_5d_functionality()

    # Demonstrate backward compatibility
    demo_backward_compatibility()

    print("\n🎉 Demo completed successfully!")
    print("\n💡 Key Benefits:")
    print("   • Full 5D (t,c,z,y,x) support")
    print("   • Chunking optimized: (1,1,<z>,<y>,<x>)")
    print("   • Push one (t,c) at a time for memory efficiency")
    print("   • 100% backward compatible with existing 3D code")
    print("   • OME-Zarr 0.4 and 0.5 support")


if __name__ == "__main__":
    main()
