#!/usr/bin/env python3
"""
Performance comparison test for chunk-optimized z-slice generator.

This test demonstrates the I/O reduction achieved by reading entire z-chunks
instead of individual z-planes.
"""

import time
import numpy as np
import tempfile
from pathlib import Path
from ome_zarr_multiscale_writer.zarr_reader import OmeZarrArray
from ome_zarr_multiscale_writer.zarr_tools import Live3DPyramidWriter, PyramidSpec


def create_test_data():
    """Create test data with known chunk structure."""
    t, c, z, y, x = 2, 2, 64, 256, 256  # Small for quick test

    # Create data where each (t,c) has unique values
    max_uint16 = 65535
    values_per_combination = max_uint16 // (t * c)

    data = np.zeros((t, c, z, y, x), dtype=np.uint16)
    for time_idx in range(t):
        for chan_idx in range(c):
            value = (time_idx * c + chan_idx) * values_per_combination
            data[time_idx, chan_idx, :, :, :] = value

    return data, t, c, z, y, x


def test_chunked_performance():
    """Test performance of chunk-optimized vs plane-by-plane access."""

    print("🔬 Testing Chunk-Optimized Z-Slice Generator")
    print("=" * 50)

    # Create test data and save as OME-Zarr
    data, t, c, z, y, x = create_test_data()

    with tempfile.TemporaryDirectory() as temp_dir:
        zarr_path = Path(temp_dir) / "test_chunked.ome.zarr"

        # First create OME-Zarr with test data (use larger z-chunks for better I/O demo)
        # Force larger z-chunks by custom chunk scheme
        from ome_zarr_multiscale_writer.zarr_tools import ChunkScheme

        chunk_scheme = ChunkScheme(base=(32, 256, 256))  # Larger z-chunks

        spec = PyramidSpec(t_size=t, c_size=c, z_size_estimate=z, y=y, x=x, levels=1)

        with Live3DPyramidWriter(
            spec=spec, path=str(zarr_path), chunk_scheme=chunk_scheme, async_close=False
        ) as writer:
            for time_idx in range(t):
                for chan_idx in range(c):
                    writer.push_slice(
                        data[time_idx, chan_idx], t_index=time_idx, c_index=chan_idx
                    )

        # Now test reading performance
        reader = OmeZarrArray(str(zarr_path))
        dataset = reader._get_dataset()

        print(f"Array shape: {dataset.shape}")
        print(f"Chunk shape: {dataset.chunks}")
        print(f"Z-chunk size: {dataset.chunks[2]} planes per chunk")
        print()

        # Test 1: Chunk-optimized reading (new method)
        print("📊 Testing Chunk-Optimized Reading:")
        start_time = time.time()

        plane_count = 0
        for time_idx in range(t):
            for chan_idx in range(c):
                for slice_data in reader._chunked_z_slices(
                    t_index=time_idx, c_index=chan_idx
                ):
                    plane_count += 1
                    # Simulate some processing
                    _ = slice_data.mean()

        chunked_time = time.time() - start_time
        print(f"  ✅ Processed {plane_count} planes in {chunked_time:.3f}s")
        print(f"  📈 Average: {chunked_time / plane_count * 1000:.2f}ms per plane")

        # Test 2: Traditional plane-by-plane reading (old method)
        print("\n📊 Testing Traditional Plane-by-Plane Reading:")
        start_time = time.time()

        plane_count = 0
        for time_idx in range(t):
            for chan_idx in range(c):
                for z_idx in range(z):
                    # This is what the old code did
                    slice_data = reader[time_idx, chan_idx, z_idx]
                    plane_count += 1
                    # Simulate some processing
                    _ = slice_data.mean()

        traditional_time = time.time() - start_time
        print(f"  ✅ Processed {plane_count} planes in {traditional_time:.3f}s")
        print(f"  📈 Average: {traditional_time / plane_count * 1000:.2f}ms per plane")

        # Performance comparison
        print("\n🎯 Performance Comparison:")
        if chunked_time < traditional_time:
            speedup = traditional_time / chunked_time
            reduction = (1 - chunked_time / traditional_time) * 100
            print(f"  🚀 Chunk-optimized is {speedup:.1f}x faster")
            print(f"  💾 I/O reduction: {reduction:.1f}%")
        else:
            print("  ⚠️  No significant performance difference detected")

        # Theoretical I/O reduction
        z_chunk_size = dataset.chunks[2]
        total_reads_traditional = t * c * z
        total_reads_chunked = (
            t * c * (z // z_chunk_size + (1 if z % z_chunk_size else 0))
        )
        theoretical_reduction = (
            1 - total_reads_chunked / total_reads_traditional
        ) * 100

        print(f"\n📚 Theoretical Analysis:")
        print(f"  📖 Traditional: {total_reads_traditional} chunk reads")
        print(f"  📖 Chunked: {total_reads_chunked} chunk reads")
        print(f"  📉 I/O reduction: {theoretical_reduction:.1f}%")


if __name__ == "__main__":
    test_chunked_performance()
