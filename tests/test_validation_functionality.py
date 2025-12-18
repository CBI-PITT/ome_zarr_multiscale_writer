"""Test the new validation functionality added to OmeZarrArray."""

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
from ome_zarr_multiscale_writer.write import write_ome_zarr_multiscale


def _prepare_store(base: Path, name: str = "sample.ome.zarr") -> Path:
    base.mkdir(parents=True, exist_ok=True)
    store_path = base / name
    if store_path.exists():
        shutil.rmtree(store_path, ignore_errors=True)
    return store_path


def _create_test_array(store_path: Path, ome_version: str = "0.4"):
    """Create a simple test OME-Zarr array."""
    data = np.arange(1000, dtype=np.uint16).reshape(10, 10, 10)
    write_ome_zarr_multiscale(
        data,
        path=store_path,
        generate_multiscales=False,
        voxel_size=(1.0, 1.0, 1.0),
        ome_version=ome_version,
    )


def test_static_validation_valid_arrays():
    """Test static method validation with valid OME-Zarr arrays."""
    # Test OME-Zarr 0.4
    store_path_v4 = _prepare_store(TEST_DATA_DIR, "validation_v4.ome.zarr")
    _create_test_array(store_path_v4, "0.4")

    is_valid, error = OmeZarrArray.validate_ome_zarr_path(str(store_path_v4))
    assert is_valid, f"OME-Zarr 0.4 should be valid, got error: {error}"
    assert error is None

    # Test OME-Zarr 0.5
    store_path_v5 = _prepare_store(TEST_DATA_DIR, "validation_v5.ome.zarr")
    _create_test_array(store_path_v5, "0.5")

    is_valid, error = OmeZarrArray.validate_ome_zarr_path(str(store_path_v5))
    assert is_valid, f"OME-Zarr 0.5 should be valid, got error: {error}"
    assert error is None


def test_static_validation_invalid_arrays():
    """Test static method validation with invalid/missing OME-Zarr arrays."""
    # Test non-existent path
    is_valid, error = OmeZarrArray.validate_ome_zarr_path("/tmp/nonexistent.ome.zarr")
    assert not is_valid, "Non-existent path should be invalid"
    assert error is not None
    assert "Failed to open store" in error

    # Test malformed zarr (no multiscales)
    malformed_path = _prepare_store(TEST_DATA_DIR, "malformed.ome.zarr")
    z = zarr.open_group(str(malformed_path), mode="w")
    z.attrs["not_multiscales"] = "test"

    is_valid, error = OmeZarrArray.validate_ome_zarr_path(str(malformed_path))
    assert not is_valid, "Malformed zarr should be invalid"
    assert error is not None

    # Test detailed errors
    is_valid, detailed_error = OmeZarrArray.validate_ome_zarr_path(
        str(malformed_path), detailed_errors=True
    )
    assert not is_valid, "Malformed zarr should be invalid with detailed errors"
    assert detailed_error is not None
    assert len(detailed_error) > len(error), "Detailed error should be longer"


def test_convenience_properties():
    """Test convenience properties for validation."""
    store_path = _prepare_store(TEST_DATA_DIR, "convenience.ome.zarr")
    _create_test_array(store_path, "0.4")

    # Create OmeZarrArray instance
    ome_array = OmeZarrArray(str(store_path))

    # Test is_valid_ome_zarr property
    assert ome_array.is_valid_ome_zarr, (
        "Valid array should return True for is_valid_ome_zarr"
    )

    # Test validation_error property
    assert ome_array.validation_error is None, (
        "Valid array should have None validation_error"
    )

    # Test with malformed array - expect initialization failure
    malformed_path = _prepare_store(TEST_DATA_DIR, "convenience_malformed.ome.zarr")
    z = zarr.open_group(str(malformed_path), mode="w")
    z.attrs["test"] = "not_ome_zarr"

    try:
        malformed_array = OmeZarrArray(str(malformed_path))
        # If initialization succeeds, validation should fail
        assert not malformed_array.is_valid_ome_zarr, (
            "Malformed array should be invalid"
        )
        assert malformed_array.validation_error is not None, (
            "Malformed array should have error message"
        )
    except ValueError as e:
        # This is expected behavior - malformed arrays can't even be instantiated
        assert "Invalid OME-Zarr store" in str(e)

    # For valid arrays, convenience properties should work correctly


def test_validation_error_cases():
    """Test various error cases for validation."""
    # Test directory without zarr
    empty_dir = _prepare_store(TEST_DATA_DIR, "empty_dir")
    empty_dir.mkdir(exist_ok=True)

    is_valid, error = OmeZarrArray.validate_ome_zarr_path(str(empty_dir))
    assert not is_valid, "Empty directory should be invalid"
    assert error is not None

    # Test with detailed error
    is_valid, detailed_error = OmeZarrArray.validate_ome_zarr_path(
        str(empty_dir), detailed_errors=True
    )
    assert not is_valid, "Empty directory should be invalid with detailed errors"
    assert detailed_error is not None


if __name__ == "__main__":
    test_static_validation_valid_arrays()
    test_static_validation_invalid_arrays()
    test_convenience_properties()
    test_validation_error_cases()
    print("All validation tests passed!")
