"""
Example usage of OME-Zarr validation functionality.

This script demonstrates both static method validation and convenience properties
for validating OME-Zarr arrays using the ome-zarr-models package.
"""

import os
import shutil
import numpy as np
from pathlib import Path

# Add package to path (for development)
import sys
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ome_zarr_multiscale_writer.write import write_ome_zarr_multiscale
from ome_zarr_multiscale_writer.zarr_reader import OmeZarrArray


def create_example_arrays():
    """Create example OME-Zarr arrays for demonstration."""
    
    # Create example data
    data = np.arange(1000, dtype=np.uint16).reshape(10, 10, 10)
    
    # Clean up existing files
    examples_dir = Path("examples/ome_zarr_examples")
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Create OME-Zarr 0.4 example
    v4_path = examples_dir / "example_v4.ome.zarr"
    if v4_path.exists():
        shutil.rmtree(v4_path)
    
    write_ome_zarr_multiscale(
        data,
        path=v4_path,
        generate_multiscales=False,
        voxel_size=(1.0, 1.0, 1.0),
        ome_version="0.4"
    )
    
    # Create OME-Zarr 0.5 example
    v5_path = examples_dir / "example_v5.ome.zarr"
    if v5_path.exists():
        shutil.rmtree(v5_path)
    
    write_ome_zarr_multiscale(
        data,
        path=v5_path,
        generate_multiscales=False,
        voxel_size=(1.0, 1.0, 1.0),
        ome_version="0.5"
    )
    
    return str(v4_path), str(v5_path)


def demonstrate_static_validation(v4_path, v5_path):
    """Demonstrate static method validation."""
    
    print("=== Static Method Validation ===")
    print("Validating OME-Zarr arrays directly from paths...")
    
    # Validate OME-Zarr 0.4
    print(f"\\n1. Validating OME-Zarr 0.4: {v4_path}")
    is_valid, error = OmeZarrArray.validate_ome_zarr_path(v4_path)
    print(f"   Valid: {is_valid}")
    if error:
        print(f"   Error: {error}")
    
    # Validate OME-Zarr 0.5  
    print(f"\\n2. Validating OME-Zarr 0.5: {v5_path}")
    is_valid, error = OmeZarrArray.validate_ome_zarr_path(v5_path)
    print(f"   Valid: {is_valid}")
    if error:
        print(f"   Error: {error}")
    
    # Test error case
    print(f"\\n3. Testing error case (non-existent path)")
    is_valid, error = OmeZarrArray.validate_ome_zarr_path("/tmp/nonexistent.ome.zarr")
    print(f"   Valid: {is_valid}")
    print(f"   Error: {error}")


def demonstrate_convenience_properties(v4_path, v5_path):
    """Demonstrate convenience properties on OmeZarrArray instances."""
    
    print("\\n=== Convenience Properties ===")
    print("Using validation properties on OmeZarrArray instances...")
    
    # Create OmeZarrArray instances
    ome_array_v4 = OmeZarrArray(v4_path)
    ome_array_v5 = OmeZarrArray(v5_path)
    
    # Test convenience properties
    print(f"\\n1. OME-Zarr 0.4 Array:")
    print(f"   is_valid_ome_zarr: {ome_array_v4.is_valid_ome_zarr}")
    print(f"   validation_error: {ome_array_v4.validation_error}")
    print(f"   Resolution levels: {ome_array_v4.ResolutionLevels}")
    print(f"   Shape: {ome_array_v4.shape}")
    
    print(f"\\n2. OME-Zarr 0.5 Array:")
    print(f"   is_valid_ome_zarr: {ome_array_v5.is_valid_ome_zarr}")
    print(f"   validation_error: {ome_array_v5.validation_error}")
    print(f"   Resolution levels: {ome_array_v5.ResolutionLevels}")
    print(f"   Shape: {ome_array_v5.shape}")
    
    # Demonstrate chunk info
    print(f"\\n3. Chunk Information:")
    ome_array_v4.print_chunk_info()
    ome_array_v5.print_chunk_info()


def main():
    """Main demonstration function."""
    print("OME-Zarr Validation Functionality Demonstration")
    print("=" * 50)
    
    # Create example arrays
    v4_path, v5_path = create_example_arrays()
    
    # Demonstrate static validation
    demonstrate_static_validation(v4_path, v5_path)
    
    # Demonstrate convenience properties
    demonstrate_convenience_properties(v4_path, v5_path)
    
    print("\\n" + "=" * 50)
    print("Demonstration completed!")
    print("\\nKey takeaways:")
    print("1. Use OmeZarrArray.validate_ome_zarr_path() for direct path validation")
    print("2. Use is_valid_ome_zarr property for quick validation of open arrays")
    print("3. Use validation_error property to get detailed error messages")
    print("4. Validation supports both OME-Zarr 0.4 and 0.5 specifications")


if __name__ == "__main__":
    main()
