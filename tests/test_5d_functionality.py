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


class Test5DFunctionality:
    """Test suite for 5D Live3DPyramidWriter functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_legacy_3d_backward_compatibility(self, temp_dir):
        """Test that existing 3D usage patterns still work."""
        # Legacy 3D data (z, y, x)
        data = np.arange(8 * 16 * 16, dtype=np.uint16).reshape(8, 16, 16)
        
        spec = PyramidSpec(
            z_size_estimate=8,
            y=16,
            x=16,
            levels=2
        )
        
        with Live3DPyramidWriter(
            spec=spec,
            path=temp_dir / "legacy_3d.ome.zarr",
            async_close=False,
            ome_version="0.5"
        ) as writer:
            # Traditional usage: push 2D slices one by one
            for z in range(data.shape[0]):
                writer.push_slice(data[z])
        
        # Verify OME metadata
        import zarr
        root = zarr.open(temp_dir / "legacy_3d.ome.zarr")
        axes = root.attrs["ome"]["multiscales"][0]["axes"]
        assert len(axes) == 3
        assert [ax["name"] for ax in axes] == ["z", "y", "x"]
        
        # Verify array shapes are 3D
        for level in range(2):
            assert len(root[str(level)].shape) == 3

    def test_5d_basic_functionality(self, temp_dir):
        """Test basic 5D functionality with t,c,z,y,x dimensions."""
        # 5D data: (2, 3, 8, 16, 16) - small for testing
        t_size, c_size, z_size, y_size, x_size = 2, 3, 8, 16, 16
        data = np.arange(t_size * c_size * z_size * y_size * x_size, dtype=np.uint16).reshape(
            t_size, c_size, z_size, y_size, x_size
        )
        
        spec = PyramidSpec(
            t_size=t_size,
            c_size=c_size,
            z_size_estimate=z_size,
            y=y_size,
            x=x_size,
            levels=2
        )
        
        with Live3DPyramidWriter(
            spec=spec,
            path=temp_dir / "test_5d.ome.zarr",
            async_close=False,
            ome_version="0.5"
        ) as writer:
            # Push data one (t,c) slice at a time
            for t in range(t_size):
                for c in range(c_size):
                    # Push 3D (z,y,x) slice for specific (t,c)
                    writer.push_slice(data[t, c], t_index=t, c_index=c)
        
        # Verify OME metadata
        import zarr
        root = zarr.open(temp_dir / "test_5d.ome.zarr")
        axes = root.attrs["ome"]["multiscales"][0]["axes"]
        assert len(axes) == 5
        assert [ax["name"] for ax in axes] == ["t", "c", "z", "y", "x"]
        
        # Verify array shapes are 5D
        for level in range(2):
            shape = root[str(level)].shape
            assert len(shape) == 5
            assert shape[0] == t_size  # t dimension preserved
            assert shape[1] == c_size  # c dimension preserved

    def test_5d_2d_slice_functionality(self, temp_dir):
        """Test 5D functionality using 2D (y,x) slices."""
        t_size, c_size, z_size, y_size, x_size = 2, 2, 4, 16, 16
        
        spec = PyramidSpec(
            t_size=t_size,
            c_size=c_size,
            z_size_estimate=z_size,
            y=y_size,
            x=x_size,
            levels=2
        )
        
        with Live3DPyramidWriter(
            spec=spec,
            path=temp_dir / "test_5d_2d.ome.zarr",
            async_close=False,
            ome_version="0.5"
        ) as writer:
            # Push data one 2D slice at a time
            for t in range(t_size):
                for c in range(c_size):
                    for z in range(z_size):
                        # Create 2D slice for specific (t,c,z)
                        slice_data = np.full((y_size, x_size), (t * 100 + c * 10 + z), dtype=np.uint16)
                        writer.push_slice(slice_data, t_index=t, c_index=c)
        
        # Verify OME metadata
        import zarr
        root = zarr.open(temp_dir / "test_5d_2d.ome.zarr")
        axes = root.attrs["ome"]["multiscales"][0]["axes"]
        assert [ax["name"] for ax in axes] == ["t", "c", "z", "y", "x"]
        
        # Verify data was written
        level0_data = root["0"]
        assert level0_data.shape == (t_size, c_size, z_size, y_size, x_size)

    def test_5d_chunking_scheme(self, temp_dir):
        """Test that 5D chunking works correctly."""
        spec = PyramidSpec(
            t_size=2,
            c_size=3,
            z_size_estimate=10,
            y=64,
            x=64,
            levels=3
        )
        
        # Custom chunking scheme
        chunk_scheme = ChunkScheme(
            base=(1, 1, 2, 32, 32),  # (t,c,z,y,x)
            target=(1, 1, 16, 8, 8)
        )
        
        with Live3DPyramidWriter(
            spec=spec,
            chunk_scheme=chunk_scheme,
            path=temp_dir / "test_5d_chunking.ome.zarr",
            async_close=False,
            ome_version="0.5"
        ) as writer:
            # Write some test data
            for t in range(spec.t_size):
                for c in range(spec.c_size):
                    test_slice = np.full((spec.y, spec.x), t * 50 + c * 5, dtype=np.uint16)
                    for z in range(5):  # Write 5 z-slices per (t,c)
                        writer.push_slice(test_slice, t_index=t, c_index=c)
        
        # Verify chunks are correct
        import zarr
        root = zarr.open(temp_dir / "test_5d_chunking.ome.zarr")
        level0 = root["0"]
        chunks = level0.chunks
        assert len(chunks) == 5
        assert chunks[0] == 1  # t chunk size
        assert chunks[1] == 1  # c chunk size
        assert chunks[2] >= 2  # z chunk size (will be adjusted by level)

    def test_5d_ome_zarr_04_compatibility(self, temp_dir):
        """Test 5D functionality with OME-Zarr 0.4."""
        spec = PyramidSpec(
            t_size=1,
            c_size=2,
            z_size_estimate=6,
            y=32,
            x=32,
            levels=2
        )
        
        with Live3DPyramidWriter(
            spec=spec,
            path=temp_dir / "test_5d_v04.ome.zarr",
            async_close=False,
            ome_version="0.4"
        ) as writer:
            for c in range(spec.c_size):
                for z in range(spec.z_size_estimate):
                    slice_data = np.full((spec.y, spec.x), c * 100 + z, dtype=np.uint16)
                    writer.push_slice(slice_data, t_index=0, c_index=c)
        
        # Verify OME-Zarr 0.4 metadata
        import zarr
        root = zarr.open(temp_dir / "test_5d_v04.ome.zarr")
        multiscales = root.attrs["multiscales"][0]
        axes = multiscales["axes"]
        assert [ax["name"] for ax in axes] == ["t", "c", "z", "y", "x"]

    def test_5d_input_validation(self, temp_dir):
        """Test input validation for 5D mode."""
        spec = PyramidSpec(
            t_size=2,
            c_size=2,
            z_size_estimate=4,
            y=16,
            x=16,
            levels=2
        )
        
        with Live3DPyramidWriter(
            spec=spec,
            path=temp_dir / "test_5d_validation.ome.zarr",
            async_close=False,
            ome_version="0.5"
        ) as writer:
            # Test valid inputs
            valid_slice = np.ones((16, 16), dtype=np.uint16)
            writer.push_slice(valid_slice, t_index=0, c_index=0)
            
            # Test invalid t_index
            with pytest.raises(ValueError, match="t_index.*out of range"):
                writer.push_slice(valid_slice, t_index=2, c_index=0)
            
            # Test invalid c_index  
            with pytest.raises(ValueError, match="c_index.*out of range"):
                writer.push_slice(valid_slice, t_index=0, c_index=2)
            
            # Test invalid data type
            with pytest.raises(ValueError, match="slice must be uint16"):
                writer.push_slice(np.ones((16, 16), dtype=np.float32), t_index=0, c_index=0)

    def test_mixed_3d_5d_auto_detection(self, temp_dir):
        """Test that 3D vs 5D mode is correctly detected based on t_size and c_size."""
        # Test 1: Single time/channel should behave as 3D
        spec_3d = PyramidSpec(
            t_size=1,  # Should trigger 3D mode
            c_size=1,
            z_size_estimate=4,
            y=16,
            x=16,
            levels=2
        )
        
        with Live3DPyramidWriter(
            spec=spec_3d,
            path=temp_dir / "test_auto_3d.ome.zarr",
            async_close=False
        ) as writer:
            assert not writer.is_5d
            slice_data = np.ones((16, 16), dtype=np.uint16)
            writer.push_slice(slice_data)  # Legacy call should work
        
        # Test 2: Multiple times/channels should trigger 5D mode
        spec_5d = PyramidSpec(
            t_size=2,  # Should trigger 5D mode
            c_size=1,
            z_size_estimate=4,
            y=16,
            x=16,
            levels=2
        )
        
        with Live3DPyramidWriter(
            spec=spec_5d,
            path=temp_dir / "test_auto_5d.ome.zarr",
            async_close=False
        ) as writer:
            assert writer.is_5d
            slice_data = np.ones((16, 16), dtype=np.uint16)
            # Must specify t_index and c_index in 5D mode
            writer.push_slice(slice_data, t_index=0, c_index=0)


if __name__ == "__main__":
    # Simple demonstration
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create 5D test data: (2, 3, 8, 32, 32)
        t, c, z, y, x = 2, 3, 8, 32, 32
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
            path=temp_path / "demo_5d.ome.zarr",
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
        root = zarr.open(temp_path / "demo_5d.ome.zarr")
        print(f"Output array shape: {root['0'].shape}")
        print(f"Output chunks: {root['0'].chunks}")
        axes = root.attrs['ome']['multiscales'][0]['axes']
        print(f"OME axes: {[ax['name'] for ax in axes}")
        print("✅ Verification completed!")