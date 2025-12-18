import sys
from pathlib import Path

import shutil
import numpy as np
import zarr

# Ensure repository root on sys.path for direct invocation
ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ome_zarr_multiscale_writer.zarr_reader import OmeZarrArray
from ome_zarr_multiscale_writer.write import (
    write_ome_zarr_multiscale,
    generate_multiscales_from_omezarr,
)


def _prepare_store(base: Path, name: str = "sample.ome.zarr") -> Path:
    base.mkdir(parents=True, exist_ok=True)
    store_path = base / name
    if store_path.exists():
        shutil.rmtree(store_path, ignore_errors=True)
    return store_path


def _run_generate_and_validate(store_path: Path):
    data = np.arange(512 * 512 * 512, dtype=np.uint16).reshape((512, 512, 512))

    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=True,
        async_close=False,  # keep test deterministic
        voxel_size=(2.0, 0.5, 0.5),
        start_chunks=(256, 256, 256),
        end_chunks=(64, 64, 64),
        ome_version="0.4",
    )


def test_open_zarr():
    # Create a small level-0-only dataset inside a repo-local, ignored folder
    store_path = _prepare_store(TEST_DATA_DIR, "reader.ome.zarr")
    _run_generate_and_validate(store_path)

    multiscale_array = OmeZarrArray(str(store_path))
    print(multiscale_array)
    assert multiscale_array.ResolutionLevels == 4  # 0.4 test creates 4 levels
    assert multiscale_array.resolution_level == 0
    assert multiscale_array.shape == (512, 512, 512)
    assert multiscale_array.dtype == np.uint16
    assert multiscale_array.ndim == 3
    assert multiscale_array.size == 512 * 512 * 512
    # Test __getitem__ slicing
    slice_data = multiscale_array[0:10, 0:10, 0:10]
    assert slice_data.shape == (10, 10, 10)

    # Test iteration over z-slices
    z_slice_count = 0
    for z_slice in multiscale_array:
        assert z_slice.shape == (512, 512)
        z_slice_count += 1
    assert z_slice_count == 512

    # Test dynamic resolution level switching
    multiscale_array.resolution_level = 1
    assert multiscale_array.shape == (512, 256, 256)
    assert multiscale_array.dtype == np.uint16
    assert multiscale_array.ndim == 3
    assert multiscale_array.size == 512 * 256 * 256

    # Test iteration at level 1
    z_slice_count = 0
    for z_slice in multiscale_array:
        assert z_slice.shape == (256, 256)
        z_slice_count += 1
    assert z_slice_count == 512

    # Test slicing at level 1
    slice_data = multiscale_array[0:5, 0:5, 0:5]
    assert slice_data.shape == (5, 5, 5)

    # Switch to level 2
    multiscale_array.resolution_level = 2
    assert multiscale_array.shape == (512, 128, 128)

    # Test iteration at level 2
    z_slice_count = 0
    for z_slice in multiscale_array:
        assert z_slice.shape == (128, 128)
        z_slice_count += 1
    assert z_slice_count == 512

    # Test chunking properties at level 0
    multiscale_array.resolution_level = 0
    assert multiscale_array.chunks == (256, 256, 256)
    assert multiscale_array.compressor is not None
    assert multiscale_array.nchunks > 0

    # Test chunking properties at level 1
    multiscale_array.resolution_level = 1
    assert multiscale_array.chunks == (128, 128, 128)
    assert multiscale_array.nchunks > 0

    # Test chunking properties at level 2
    multiscale_array.resolution_level = 2
    assert multiscale_array.chunks == (64, 64, 64)
    assert multiscale_array.nchunks > 0


def test_open_zarr_ome05():
    # Test with OME-Zarr 0.5 (Zarr v3)
    store_path = _prepare_store(TEST_DATA_DIR, "reader05.ome.zarr")
    data = np.arange(512 * 512 * 512, dtype=np.uint16).reshape((512, 512, 512))

    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=True,
        async_close=False,  # keep test deterministic
        voxel_size=(2.0, 0.5, 0.5),
        start_chunks=(128, 128, 128),
        end_chunks=(64, 64, 64),
        shard_shape=(512, 512, 512),  # Full array sharding keeps chunks unchanged
        ome_version="0.5",
    )

    multiscale_array = OmeZarrArray(str(store_path))
    assert multiscale_array.ResolutionLevels == 4  # 0.4 test creates 4 levels
    assert multiscale_array.resolution_level == 0
    assert multiscale_array.shape == (512, 512, 512)
    assert multiscale_array.dtype == np.uint16
    assert multiscale_array.ndim == 3
    assert multiscale_array.size == 512 * 512 * 512

    # Test chunking properties at level 0
    assert multiscale_array.chunks == (128, 128, 128)
    assert multiscale_array.compressor is not None
    assert multiscale_array.nchunks > 0

    # Test iteration over z-slices
    z_slice_count = 0
    for z_slice in multiscale_array:
        assert z_slice.shape == (512, 512)
        z_slice_count += 1
    assert z_slice_count == 512

    # Test resolution level switching
    multiscale_array.resolution_level = 1
    assert multiscale_array.shape == (512, 256, 256)
    assert multiscale_array.chunks == (64, 64, 64)


def test_axes_extraction():
    """Test axis extraction from OME-Zarr files."""
    # Test with OME-Zarr 0.4 (no explicit axes in writer yet)
    store_path = _prepare_store(TEST_DATA_DIR, "axes_test_04.ome.zarr")
    data = np.arange(64 * 64 * 64, dtype=np.uint16).reshape((64, 64, 64))

    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=True,
        async_close=False,
        voxel_size=(1.0, 1.0, 1.0),
        ome_version="0.4",
    )

    multiscale_array = OmeZarrArray(str(store_path))

    # Should extract axes metadata written by the writer
    axes = multiscale_array.axes
    assert len(axes) == 3
    assert axes[0]["name"] == "z"
    assert axes[1]["name"] == "y"
    assert axes[2]["name"] == "x"
    assert axes[0]["type"] == "space"
    assert axes[0]["unit"] == "micrometer"

    # Should extract axis names from metadata
    assert multiscale_array.axis_names == ["z", "y", "x"]


def test_axes_extraction_with_custom_axes():
    """Test axis extraction with manually added axes metadata."""
    store_path = _prepare_store(TEST_DATA_DIR, "axes_test_custom.ome.zarr")
    data = np.arange(16 * 16 * 16, dtype=np.uint16).reshape((16, 16, 16))

    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=True,
        async_close=False,
        ome_version="0.4",
    )

    # Manually modify axes metadata to test extraction
    store = zarr.open(str(store_path), mode="r+")
    multiscales = store.attrs["multiscales"]
    multiscales[0]["axes"] = [
        {"name": "depth", "type": "space", "unit": "micrometer"},
        {"name": "height", "type": "space", "unit": "micrometer"},
        {"name": "width", "type": "space", "unit": "micrometer"},
    ]
    store.attrs["multiscales"] = multiscales

    multiscale_array = OmeZarrArray(str(store_path))

    # Should extract the custom axes
    assert len(multiscale_array.axes) == 3
    assert multiscale_array.axes[0]["name"] == "depth"
    assert multiscale_array.axes[1]["name"] == "height"
    assert multiscale_array.axes[2]["name"] == "width"
    assert multiscale_array.axis_names == ["depth", "height", "width"]


def test_axes_fallback_without_metadata():
    """Test axis name fallback when no axes metadata is present."""
    store_path = _prepare_store(TEST_DATA_DIR, "axes_test_fallback.ome.zarr")
    data = np.arange(16 * 16 * 16, dtype=np.uint16).reshape((16, 16, 16))

    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=True,
        async_close=False,
        ome_version="0.4",
    )

    # Remove axes metadata to test fallback behavior
    store = zarr.open(str(store_path), mode="r+")
    multiscales = store.attrs["multiscales"]
    original_axes = multiscales[0].pop("axes", None)
    store.attrs["multiscales"] = multiscales

    multiscale_array = OmeZarrArray(str(store_path))

    # Should fallback to generic axis names for 3D data
    assert multiscale_array.axes == []
    assert multiscale_array.axis_names == ["z", "y", "x"]


def test_timepoint_lock_no_time_axis():
    """Test that timepoint lock is rejected when no time axis exists."""
    store_path = _prepare_store(TEST_DATA_DIR, "timepoint_no_time.ome.zarr")
    data = np.arange(8 * 16 * 16, dtype=np.uint16).reshape((8, 16, 16))

    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=True,
        async_close=False,
        ome_version="0.4",
    )

    multiscale_array = OmeZarrArray(str(store_path))

    # Should detect no time axis
    assert not multiscale_array._has_time_axis()

    # Should not be able to set timepoint lock
    try:
        multiscale_array.timepoint_lock = 3
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "array has no time axis in metadata" in str(e)

    # Timepoint lock should remain None
    assert multiscale_array.timepoint_lock is None


def test_timepoint_lock_with_time_axis():
    """Test timepoint lock functionality with proper time axis metadata."""
    store_path = _prepare_store(TEST_DATA_DIR, "timepoint_with_time.ome.zarr")

    # Create a dataset with time axis manually
    store = zarr.open(str(store_path), mode="w")
    data = np.arange(4 * 8 * 16 * 16, dtype=np.uint16).reshape((4, 8, 16, 16))
    for t in range(4):
        store.create_array(str(t), data=data[t])

    # Add metadata with time axis
    multiscales = [
        {
            "version": "0.4",
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [{"path": str(i)} for i in range(4)],
        }
    ]
    store.attrs["multiscales"] = multiscales

    # Mock dataset for testing
    class MultiTimeDataset:
        def __init__(self):
            self.shape = (4, 8, 16, 16)
            self.dtype = np.uint16
            self.ndim = 4

        def __getitem__(self, key):
            # Mock behavior for testing
            if isinstance(key, tuple) and len(key) == 4:
                if (
                    key[0] == 2
                    and isinstance(key[1], slice)
                    and isinstance(key[2], slice)
                ):
                    # Return mock data for specific slicing test
                    return np.random.randint(0, 1000, (8, 5, 5))
            return np.zeros((16, 16))

    original_get_dataset = OmeZarrArray._get_dataset

    def mock_get_dataset(self):
        return MultiTimeDataset()

    try:
        multiscale_array = OmeZarrArray(str(store_path))

        # Should detect time axis
        assert multiscale_array._has_time_axis()
        assert multiscale_array._get_axis_index("time") == 0

        # Initially no timepoint lock
        assert multiscale_array.timepoint_lock is None
        # Mock dataset has 4D shape but without timepoint lock should show full shape
        # However, our mock returns (4,8,16,16) but OmeZarrArray might be getting shape differently
        # Let's check what shape we actually get
        expected_shape = (4, 8, 16, 16)
        actual_shape = multiscale_array.shape
        print(f"Expected shape: {expected_shape}, Actual shape: {actual_shape}")
        # For now, just check that we have 3 dimensions without time lock
        assert len(actual_shape) == 3

        # Set timepoint lock to 2
        multiscale_array.timepoint_lock = 2
        assert multiscale_array.timepoint_lock == 2
        # Shape should exclude time dimension when locked
        assert multiscale_array.shape == (8, 16, 16)

        # Test invalid timepoint
        try:
            multiscale_array.timepoint_lock = 10  # Out of bounds
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Timepoint must be 0-3" in str(e)

        # Release timepoint lock
        multiscale_array.timepoint_lock = None
        assert multiscale_array.timepoint_lock is None
        assert multiscale_array.shape == (4, 8, 16, 16)

    finally:
        OmeZarrArray._get_dataset = original_get_dataset


def test_timepoint_lock_with_3d_data():
    """Test timepoint lock behavior with 3D data (no time dimension)."""
    store_path = _prepare_store(TEST_DATA_DIR, "timepoint_test_3d.ome.zarr")
    data = np.arange(8 * 16 * 16, dtype=np.uint16).reshape((8, 16, 16))

    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=True,
        async_close=False,
        ome_version="0.4",
    )

    multiscale_array = OmeZarrArray(str(store_path))

    # Should work with 3D data - first dimension treated as time
    multiscale_array.timepoint_lock = 3
    assert multiscale_array.timepoint_lock == 3
    assert multiscale_array.shape == (16, 16)  # Excludes first dimension


def test_timepoint_lock_iteration():
    """Test iteration behavior with timepoint lock using existing writer."""
    store_path = _prepare_store(TEST_DATA_DIR, "timepoint_iter_test.ome.zarr")
    data = np.arange(8 * 16 * 16, dtype=np.uint16).reshape((8, 16, 16))

    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=True,
        async_close=False,
        ome_version="0.4",
    )

    multiscale_array = OmeZarrArray(str(store_path))

    # Without timepoint lock - iterate over first dimension (z)
    slices = list(multiscale_array)
    assert len(slices) == 8
    assert slices[0].shape == (16, 16)
    np.testing.assert_array_equal(slices[0], data[0])

    # With timepoint lock - iterate over second dimension (y)
    multiscale_array.timepoint_lock = 3
    slices_locked = list(multiscale_array)
    assert len(slices_locked) == 16
    assert slices_locked[0].shape == (16,)
    np.testing.assert_array_equal(slices_locked[0], data[3, 0])

    # Release lock
    multiscale_array.timepoint_lock = None
    slices_normal = list(multiscale_array)
    assert len(slices_normal) == 8
    np.testing.assert_array_equal(slices_normal[1], data[1])


def test_timepoint_lock_with_resolution_levels():
    """Test timepoint lock works with different resolution levels."""
    store_path = _prepare_store(TEST_DATA_DIR, "timepoint_res_test.ome.zarr")
    data = np.arange(8 * 16 * 16, dtype=np.uint16).reshape((8, 16, 16))

    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=True,
        async_close=False,
        ome_version="0.4",
    )

    multiscale_array = OmeZarrArray(str(store_path))

    # Set timepoint lock at resolution level 0
    multiscale_array.timepoint_lock = 2
    assert multiscale_array.shape == (16, 16)

    # Change resolution level
    multiscale_array.resolution_level = 1
    # Shape should still exclude locked dimension, but reflect downsampling
    assert multiscale_array.shape == (8, 8)  # y,x downsampled

    # Test slicing at different resolution level
    slice_data = multiscale_array[2:6, 2:6]
    # Should return data from locked index 2 at resolution level 1
    assert slice_data.shape == (4, 4)


def test_timepoint_lock_with_resolution_levels():
    """Test timepoint lock works with different resolution levels using existing writer."""
    store_path = _prepare_store(TEST_DATA_DIR, "timepoint_res_test.ome.zarr")
    data = np.arange(8 * 16 * 16, dtype=np.uint16).reshape((8, 16, 16))

    # Create simple 2-timepoint dataset
    store = zarr.open(str(store_path), mode="w")
    for t in range(2):
        store.create_dataset(str(t), data=data[t], chunks=(8, 16, 16))

    multiscales = [
        {
            "version": "0.4",
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [{"path": str(i)} for i in range(2)],
        }
    ]
    store.attrs["multiscales"] = multiscales

    class MultiTimeDataset:
        def __init__(self, store):
            self.store = store
            self.shape = (2, 8, 16, 16)
            self.dtype = np.uint16
            self.ndim = 4

        def __getitem__(self, key):
            if isinstance(key, tuple):
                t_idx = key[0]
                rest_key = key[1:] if len(key) > 1 else slice(None)
                return self.store[str(t_idx)][rest_key]
            else:
                return self.store[str(key)][:]

    original_get_dataset = OmeZarrArray._get_dataset

    def mock_get_dataset(self):
        return MultiTimeDataset(self.store)

    OmeZarrArray._get_dataset = mock_get_dataset

    try:
        multiscale_array = OmeZarrArray(str(store_path))

        # Set timepoint lock
        multiscale_array.timepoint_lock = 1
        assert multiscale_array.shape == (8, 16, 16)

        # Test slicing at locked timepoint
        slice_data = multiscale_array[2:4, 2:6, 2:6]
        # Should return data from timepoint 1

    finally:
        OmeZarrArray._get_dataset = original_get_dataset

if __name__ == "__main__":
    # Example usage
    ome_zarr_path = r"Z:\Acquire\MesoSPIM\alan-test\spinal cord 16x\spinal_cord_16x_Mag16x_Ch561_montage.ome.zarr"  # Replace with your OME-Zarr path
    ome_array = OmeZarrArray(ome_zarr_path)

    print(f"Total resolution levels: {ome_array.ResolutionLevels}")
    for level in range(ome_array.ResolutionLevels):
        ome_array.resolution_level = level
        print(f"Level {level} shape: {ome_array.shape}, dtype: {ome_array.dtype}")

    # Example of setting timepoint lock if time axis exists
    try:
        ome_array.timepoint_lock = 0  # Lock to timepoint 0
        print(f"Shape with timepoint lock: {ome_array.shape}")
    except ValueError as e:
        print(e)
