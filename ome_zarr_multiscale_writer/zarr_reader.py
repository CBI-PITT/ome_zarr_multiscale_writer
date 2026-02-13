import zarr
import numpy as np
from typing import List, Dict, Any, cast, Union, Tuple, Any as TypingAny, Optional
from zarr import Array
import dask.array as da
from ome_zarr_models import open_ome_zarr
from ome_zarr_models.v05.image import Image as ImageV05
from ome_zarr_models.v04.image import Image as ImageV04
from pathlib import Path
import os
import json
from .zarr_tools import Live3DPyramidWriter, PyramidSpec, ChunkScheme, FlushPad
from .helpers import key_to_slices


class ZarrToOmeZarrConverter:
    """
    Converter class for transforming generic Zarr v2/v3 arrays into OME-Zarr format.

    This class opens a Zarr store containing a generic array (not necessarily OME-Zarr)
    and provides methods to add OME-Zarr metadata, converting it to a valid OME-Zarr
    store with a single multiscale level (level 0). It does not create additional
    multiscale levels or copy data; it only adds metadata.

    Assumes the array to convert is located at the specified array_path (default "0").
    """

    def __init__(self, store_path: str, array_path: str = "0", mode: str = "r"):
        """
        Initialize the converter.

        Args:
            store_path: Path to the Zarr store.
            array_path: Path to the array within the store (default "0").
            mode: Access mode ('r', 'r+', 'a', etc.).
        """
        self.store_path = store_path
        self.array_path = array_path
        self.mode = mode
        self.store = zarr.open(store_path, mode=mode)
        self.array = self.store[array_path]
        if not isinstance(self.array, zarr.Array):
            raise ValueError(
                f"Expected zarr.Array at {array_path}, got {type(self.array)}"
            )

    def convert(
        self,
        axes: Optional[List[Dict[str, Any]]] = None,
        voxel_size: Optional[Tuple[float, ...]] = None,
        ome_version: str = "0.5",
    ) -> None:
        """
        Convert the Zarr array to OME-Zarr format by adding multiscale metadata.

        Modifies the store in place by adding OME-Zarr metadata for a single level (level 0).

        Args:
            axes: List of axis dictionaries. If None, inferred from array dimensionality.
            voxel_size: Voxel size for spatial axes (z, y, x). If None, defaults to 1.0.
            ome_version: OME-Zarr version ('0.4' or '0.5', default '0.5').
        """
        if ome_version not in ["0.4", "0.5"]:
            raise ValueError("ome_version must be '0.4' or '0.5'")

        ndim = self.array.ndim
        if axes is None:
            axes = self._infer_axes(ndim)

        spatial_dims = sum(1 for axis in axes if axis.get("type") == "space")
        if voxel_size is None:
            voxel_size = tuple(1.0 for _ in range(spatial_dims))
        if len(voxel_size) != spatial_dims:
            raise ValueError(
                f"voxel_size length {len(voxel_size)} does not match spatial dimensions {spatial_dims}"
            )

        # Build scale and translation transformations
        scale = [1.0] * len(axes)
        translation = [0.0] * len(axes)
        spatial_idx = 0
        for i, axis in enumerate(axes):
            if axis.get("type") == "space" and spatial_idx < len(voxel_size):
                scale[i] = voxel_size[spatial_idx]
                spatial_idx += 1

        datasets = [
            {
                "path": self.array_path,
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale},
                    {"type": "translation", "translation": translation},
                ],
            }
        ]

        multiscales = [
            {
                "version": ome_version,
                "axes": axes,
                "datasets": datasets,
                "name": "image",
            }
        ]

        if ome_version == "0.5":
            multiscales[0]["type"] = "image"

        # Add metadata to store attributes
        if ome_version == "0.4":
            self.store.attrs["multiscales"] = multiscales
        else:
            if "ome" not in self.store.attrs:
                self.store.attrs["ome"] = {}
            self.store.attrs["ome"]["version"] = ome_version
            self.store.attrs["ome"]["multiscales"] = multiscales

    def _infer_axes(self, ndim: int) -> List[Dict[str, Any]]:
        """Infer axes metadata from array dimensionality."""
        if ndim == 5:
            return [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ]
        elif ndim == 4:
            # Assume channel, z, y, x
            return [
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ]
        elif ndim == 3:
            return [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ]
        elif ndim == 2:
            return [
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ]
        else:
            return [
                {"name": f"axis_{i}", "type": "space" if i >= ndim - 3 else "unknown"}
                for i in range(ndim)
            ]


class OmeZarrArray:
    """Convenience class for accessing OME-Zarr v0.4/v0.5 arrays with resolution selection."""

    # Zarr v2 metadata files.
    ZARR_V2_META = {".zgroup", ".zattrs", ".zarray", ".zmetadata"}

    # Zarr v3 metadata files
    ZARR_V3_META = {"zarr.json"}

    def __init__(self, store_path: str, mode="r") -> None:
        assert os.path.exists(store_path), (
            f"Zarr store path does not exist: {store_path}"
        )
        self.store_path = store_path
        self._mode = mode
        self.store = zarr.open(store_path, mode=self._mode)
        self._get_multiscale_metadata()
        self._resolution_level: int = 0
        self._timepoint_lock: int | None = None

    def _get_multiscale_metadata(self) -> Dict[str, Any]:
        self._axes = None
        self._scale_datasets = None
        self._ome_version = None

        # Try OME-Zarr 0.5 format first (metadata under 'ome' key)
        multiscales = None
        ome = self.store.attrs.get("ome")
        if ome and isinstance(ome, dict):
            multiscales = ome.get("multiscales")

        # Fall back to OME-Zarr 0.4 format (direct 'multiscales' key)
        if not multiscales:
            multiscales = self.store.attrs.get("multiscales", [])

        if not multiscales:
            raise ValueError("Invalid OME-Zarr store: missing multiscales metadata")

        # Extract axes metadata if present
        if (
            isinstance(multiscales, list)
            and len(multiscales) > 0
            and isinstance(multiscales[0], dict)
        ):
            multiscales_dict = multiscales[0]
            self._axes = cast(List[Dict[str, Any]], multiscales_dict.get("axes", []))
            self._scale_datasets: List[Dict[str, Any]] = cast(
                List[Dict[str, Any]], multiscales_dict.get("datasets", [])
            )
        else:
            self._axes = []
            self._scale_datasets = []
        if not self._scale_datasets:
            raise ValueError("Invalid multiscales: missing datasets")

        # Store OME version for consistency in create_multiscales
        self._ome_version = "0.5" if ome and isinstance(ome, dict) else "0.4"

    def _get_dataset(self) -> Array:
        """Get the current resolution level's dataset."""
        path = self._get_dataset_path()
        dataset = self.store[path]
        if not isinstance(dataset, zarr.Array):
            raise ValueError(f"Expected zarr.Array at {path}, got {type(dataset)}")
        return dataset

    def _get_dataset_path(self) -> str:
        """Get the current resolution level's dataset path."""
        return self._scale_datasets[self.resolution_level]["path"]

    @property
    def ResolutionLevels(self) -> int:
        """Total number of available resolution levels."""
        return len(self._scale_datasets)

    @property
    def resolution_level(self) -> int:
        return self._resolution_level

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def nbytes(self) -> int:
        return self._get_dataset().nbytes

    @mode.setter
    def mode(self, mode) -> None:
        if mode.lower() == self._mode:
            return  # No change
        self._mode = mode.lower()
        self.store = zarr.open(self.store_path, mode=self._mode)

    @resolution_level.setter
    def resolution_level(self, level: int) -> None:
        if level < 0 or level >= self.ResolutionLevels:
            raise ValueError(f"Level must be 0-{self.ResolutionLevels - 1}")
        self._resolution_level = level

    @property
    def timepoint_lock(self) -> int | None:
        """Get the current timepoint lock. None means no timepoint lock."""
        return self._timepoint_lock

    @timepoint_lock.setter
    def timepoint_lock(self, timepoint: int | None) -> None:
        """Set timepoint lock. None disables timepoint lock."""
        if timepoint is not None:
            # Check if array has time axis in metadata
            if not self._has_time_axis():
                raise ValueError(
                    "Cannot set timepoint lock: array has no time axis in metadata"
                )

            # Get time axis index
            time_idx = self._get_axis_index("time")
            if time_idx is None:
                raise ValueError("Cannot set timepoint lock: time axis not found")

            shape = self._get_dataset().shape
            if time_idx >= len(shape) or timepoint < 0 or timepoint >= shape[time_idx]:
                raise ValueError(
                    f"Timepoint must be 0-{shape[time_idx] - 1}, got {timepoint}"
                )
        self._timepoint_lock = timepoint

    @property
    def shape(self) -> tuple:
        shape = self._get_dataset().shape
        if self._timepoint_lock is not None:
            time_idx = self._get_axis_index("time")
            if time_idx is not None and time_idx < len(shape):
                # Remove the time dimension when timepoint is locked
                return shape[:time_idx] + shape[time_idx + 1 :]
        return shape

    @property
    def ome(self) -> float:
        return float(self._ome_version)

    @property
    def dtype(self) -> np.dtype:
        return self._get_dataset().dtype

    @property
    def ndim(self) -> int:
        return self._get_dataset().ndim

    @property
    def size(self) -> int:
        return self._get_dataset().size

    @property
    def chunks(self) -> tuple:
        return self._get_dataset().chunks

    @property
    def compressor(self) -> object:
        dataset = self._get_dataset()
        # Handle Zarr v2 vs v3 compressor access
        if hasattr(dataset, "metadata") and dataset.metadata.zarr_format == 2:
            return dataset.compressor
        elif hasattr(dataset, "compressors"):
            return dataset.compressors
        return None

    @property
    def nchunks(self) -> int:
        """Number of chunks physically written to disk (property access)."""
        return self._get_dataset().nchunks_initialized

    @property
    def voxel_size(self) -> tuple:
        """Get voxel size from OME-Zarr metadata axes."""
        resolution_metadata = self._scale_datasets[self.resolution_level]
        coordinate_transforms = resolution_metadata.get(
            "coordinateTransformations", []
        )[0]
        if isinstance(coordinate_transforms, dict):
            if coordinate_transforms.get("type") == "scale":
                scale = coordinate_transforms.get("scale", [])
                if len(scale) >= 3:
                    return tuple(scale[-3:])  # Return last three values (z,y,x)
                else:
                    return ()
        return ()

    @property
    def cdata_shape(self) -> tuple:
        return self._get_dataset().cdata_shape

    @property
    def total_chunks(self) -> int:
        return int(np.prod(self.cdata_shape))

    @property
    def axes(self) -> List[Dict[str, Any]]:
        """Axis metadata from the multiscales specification. Returns empty list if no axes defined."""
        return self._axes

    @property
    def axis_names(self) -> List[str]:
        """List of axis names extracted from axes metadata. Returns generic names if axes not defined."""
        if self._axes:
            return [axis.get("name", f"axis_{i}") for i, axis in enumerate(self._axes)]
        else:
            # Fallback to generic axis names based on dimensionality
            ndim = self.ndim
            if ndim == 5:
                return ["t", "c", "z", "y", "x"]
            elif ndim == 4:
                return ["c", "z", "y", "x"]
            elif ndim == 3:
                return ["z", "y", "x"]
            elif ndim == 2:
                return ["y", "x"]
            else:
                return [f"axis_{i}" for i in range(ndim)]

    def _get_axis_index(self, axis_type: str) -> int | None:
        """Get the index of a specific axis type from axes metadata."""
        if not self._axes:
            return None

        for i, axis in enumerate(self._axes):
            if axis.get("type") == axis_type:
                return i
        return None

    def _has_time_axis(self) -> bool:
        """Check if the array has a time axis in metadata."""
        return self._get_axis_index("time") is not None

    def _has_channel_axis(self) -> bool:
        """Check if the array has a channel axis in metadata."""
        return self._get_axis_index("channel") is not None

    def __getitem__(
        self, key: Union[int, slice, Tuple, np.ndarray, TypingAny]
    ) -> Union[np.ndarray, TypingAny]:
        dataset = self._get_dataset()
        if self._timepoint_lock is not None:
            time_idx = self._get_axis_index("time")
            if time_idx is not None:
                # Insert timepoint lock at the correct position
                if isinstance(key, tuple):
                    # If key is already a tuple, insert the timepoint at the right position
                    key = key[:time_idx] + (self._timepoint_lock,) + key[time_idx:]
                else:
                    # If key is a single slice/index, create a tuple with timepoint
                    key = tuple(
                        slice(None) if i == time_idx else key
                        for i in range(dataset.ndim)
                    )
                    key = key[:time_idx] + (self._timepoint_lock,) + key[time_idx + 1 :]
        return dataset[key]

    def __setitem__(
        self,
        key: Union[int, slice, Tuple, np.ndarray, TypingAny],
        value: Union[np.ndarray, TypingAny],
    ) -> Union[np.ndarray, TypingAny]:
        if not self.mode in ["a", "r+"]:
            raise ValueError("""Array is not opened in write mode. Must be 'a'.
            Example: OmeZarrArray(store_path, mode='a').
            OmeZarrArrayObj.mode = 'a'""")
        if not self.resolution_level == 0:
            raise ValueError("""Can only write to resolution level 0 (full resolution data).
            Set OmeZarrArrayObj.resolution_level = 0 before writing.""")

        print(f"{key=}, {self.shape=}")
        key = key_to_slices(key, self.shape)

        dataset = self._get_dataset()
        if self._timepoint_lock is not None:
            time_idx = self._get_axis_index("time")
            if time_idx is not None:
                # Insert timepoint lock at the correct position
                if isinstance(key, tuple):
                    # If key is already a tuple, insert the timepoint at the right position
                    key = key[:time_idx] + (self._timepoint_lock,) + key[time_idx:]
                else:
                    # If key is a single slice/index, create a tuple with timepoint
                    key = tuple(
                        slice(None) if i == time_idx else key
                        for i in range(dataset.ndim)
                    )
                    key = key[:time_idx] + (self._timepoint_lock,) + key[time_idx + 1 :]
        dataset[key] = value

    def print_chunk_info(self) -> None:
        """Print number of initialized chunks out of total chunks for current resolution level."""
        dataset = self._get_dataset()
        initialized = self.nchunks
        total = self.total_chunks
        print(f"Initialized chunks: {initialized} / {total}")

    @staticmethod
    def validate_ome_zarr_path(
        store_path: str, detailed_errors: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if the OME-Zarr store at the given path is up to spec.

        Args:
            store_path: Path to the OME-Zarr store
            detailed_errors: If True, return detailed error messages

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: Boolean indicating if valid OME-Zarr
            - error_message: None if valid, otherwise error description
        """
        try:
            # Open the zarr group for validation
            zarr_group = zarr.open_group(store_path, mode="r")

            # Try OME-Zarr 0.5 validation first
            try:
                ImageV05.from_zarr(zarr_group)
                return True, None
            except Exception:
                pass

            # Try OME-Zarr 0.4 validation
            try:
                ImageV04.from_zarr(zarr_group)
                return True, None
            except Exception:
                pass

            # Try auto-detection as fallback
            try:
                open_ome_zarr(zarr_group)
                return True, None
            except Exception as e:
                error_msg = (
                    str(e)
                    if detailed_errors
                    else f"Validation failed: {type(e).__name__}"
                )
                return False, error_msg

        except Exception as e:
            error_msg = (
                str(e)
                if detailed_errors
                else f"Failed to open store: {type(e).__name__}"
            )
            return False, error_msg

    @property
    def is_valid_ome_zarr(self) -> bool:
        """Convenience property to check if currently opened array is valid OME-Zarr."""
        return self.validate_ome_zarr_path(self.store_path)[0]

    @property
    def validation_error(self) -> Optional[str]:
        """Convenience property to get validation error message for current array."""
        return self.validate_ome_zarr_path(self.store_path, detailed_errors=True)[1]

    def create_multiscales(
        self,
        target_path: Optional[Union[str, Path]] = None,
        voxel_size: Optional[Tuple[float, float, float]] = None,
        levels: Optional[int] = None,
        start_chunks: Optional[Tuple[int, int, int]] = None,
        end_chunks: Optional[Tuple[int, int, int]] = None,
        compressor: Optional[str] = None,
        compression_level: int = 5,
        flush_pad: FlushPad = FlushPad.DUPLICATE_LAST,
        async_close=True,
        **kwargs,
    ) -> str:
        """
        Create or recreate multiscales from the full resolution data in this OmeZarrArray.

        Uses Live3DPyramidWriter to generate a proper multiscale pyramid from the
        highest resolution data (level 0) in the current array.

        Args:
            target_path: Optional destination path for the new multiscales.
                        If None, recreates multiscales in the current store.
            voxel_size: Physical voxel size (z, y, x). If None, attempts to infer from metadata.
            levels: Number of pyramid levels to generate.
            start_chunks: Chunk shape for level 0 (z, y, x).
            end_chunks: Target chunk shape for coarsest level (z, y, x).
            compressor: Compressor name (e.g., 'zstd', 'lz4').
            compression_level: Compression level for the compressor.
            flush_pad: How to handle partially filled chunks.
            **kwargs: Additional arguments passed to Live3DPyramidWriter.

        Returns:
            Path to the store containing the created multiscales.

        Raises:
            ValueError: If the array doesn't contain level 0 data or other validation errors.
        """
        # Ensure we're working with level 0 (full resolution)
        original_level = self.resolution_level
        self.resolution_level = 0

        try:
            # Get the full resolution data shape
            full_res_shape = self.shape

            # Determine if this is 5D data (t, c, z, y, x) or 3D (z, y, x)
            if len(full_res_shape) == 5:
                t_size, c_size, z_size, y_size, x_size = full_res_shape
            elif len(full_res_shape) == 4:
                # Could be (t, z, y, x) or (c, z, y, x)
                if self._has_time_axis():
                    t_size = full_res_shape[0]
                    c_size = 1
                    z_size, y_size, x_size = full_res_shape[1:]
                else:
                    t_size = 1
                    c_size = full_res_shape[0]
                    z_size, y_size, x_size = full_res_shape[1:]
            elif len(full_res_shape) == 3:
                t_size = 1
                c_size = 1
                z_size, y_size, x_size = full_res_shape
            else:
                raise ValueError(
                    f"Unsupported array dimensionality: {len(full_res_shape)}D"
                )

            if levels is None and self.ResolutionLevels == 1:
                levels = 5  # Default to 5 levels if none exist
            elif levels is None and self.ResolutionLevels > 1:
                levels = self.ResolutionLevels  # Use existing number of levels

            # Create PyramidSpec
            spec = PyramidSpec(
                z_size_estimate=z_size,
                y=y_size,
                x=x_size,
                levels=levels,
                t_size=t_size,
                c_size=c_size,
            )

            # Set chunks for new multiscales
            chunk_scheme = None
            if start_chunks is None and end_chunks is None:
                # Determine chunk_scheme based on existing levels if available
                try:
                    chunks_per_level = []
                    for l in range(levels):
                        self.resolution_level = l
                        chunks_per_level.append(
                            self.chunks
                        )  # Will throw error if level missing
                    chunk_scheme = ChunkScheme(hard_coded=chunks_per_level)
                except:
                    chunks_per_level = []
                finally:
                    self.resolution_level = 0

            if chunk_scheme is None:
                if not start_chunks:
                    self.resolution_level = 0  # get chunks for res 0
                    start_chunks = self.chunks

                if not end_chunks and self.ResolutionLevels > 1:
                    self.resolution_level = (
                        self.ResolutionLevels - 1
                    )  # get chunks for last res
                    try:
                        end_chunks = self.chunks
                    except KeyError:
                        print(
                            "The last multiscale is missing, it may have been deleted. Ending chunks is being sent to same value as Resolution Level 0."
                        )
                        end_chunks = start_chunks
                    self.resolution_level = 0  # Restore to res 0
                if not end_chunks:
                    # If only one level exists, set default end chunks
                    end_chunks = (256, 256, 256)

                chunk_scheme = ChunkScheme(base=start_chunks, target=end_chunks)

            # Set up compressor if specified
            if not compressor:
                compressor_obj = self.compressor  # Use existing compressor
            else:
                if compressor == "zstd":
                    from zarr.codecs import BloscCodec, BloscShuffle

                    compressor_obj = BloscCodec(
                        cname="zstd",
                        clevel=compression_level,
                        shuffle=BloscShuffle.bitshuffle,
                    )
                elif compressor == "lz4":
                    from zarr.codecs import BloscCodec, BloscShuffle

                    compressor_obj = BloscCodec(
                        cname="lz4",
                        clevel=compression_level,
                        shuffle=BloscShuffle.bitshuffle,
                    )
                else:
                    compressor_obj = None

            # Infer voxel size if not provided
            if voxel_size is None:
                # Try to get from metadata, otherwise use default
                voxel_size = self.voxel_size
            if len(voxel_size) == 0:
                voxel_size = (1.0, 1.0, 1.0)  # default fallback

            # Determine target path
            if target_path is None:
                target_path = self.store_path

                # Delete existing multiscale levels except level 0
                if len(self._scale_datasets) > 1:
                    for i in range(
                        1, len(self._scale_datasets)
                    ):  # Skip level 0 (index 0)
                        level_path = self._scale_datasets[i]["path"]
                        if level_path in self.store:
                            del self.store[level_path]

                # Do not write level 0 again to preserve level0 data when overwriting in place
                write_level0 = False
            else:
                # When writing to a new path, always write level 0
                write_level0 = True

            # Set up Live3DPyramidWriter with consistent OME version
            writer = Live3DPyramidWriter(
                spec=spec,
                voxel_size=voxel_size,
                path=target_path,
                chunk_scheme=chunk_scheme,
                compressor=compressor_obj,
                flush_pad=flush_pad,
                ome_version=self._ome_version,
                write_level0=write_level0,
                async_close=async_close,
                **kwargs,
            )

            # Stream the data
            with writer:
                if t_size == 1 and c_size == 1:
                    # 3D case: stream z slices using chunked generator
                    for slice_data in self._chunked_z_slices():
                        writer.push_slice(slice_data)
                else:
                    # 5D case: handle time and channels
                    for t in range(t_size):
                        for c in range(c_size):
                            if self._has_time_axis() and self._has_channel_axis():
                                # Both time and channel axes - use chunked generator
                                for slice_data in self._chunked_z_slices(
                                    t_index=t, c_index=c
                                ):
                                    writer.push_slice(slice_data, t_index=t, c_index=c)
                            elif self._has_time_axis():
                                # Only time axis - use chunked generator
                                for slice_data in self._chunked_z_slices(
                                    t_index=t, c_index=None
                                ):
                                    writer.push_slice(slice_data, t_index=t)
                            elif self._has_channel_axis():
                                # Only channel axis - use chunked generator
                                for slice_data in self._chunked_z_slices(
                                    t_index=None, c_index=c
                                ):
                                    writer.push_slice(slice_data, c_index=c)
                            else:
                                # Fallback: treat as 3D for each t,c combination
                                for z in range(z_size):
                                    # Build indexing tuple dynamically
                                    indices = []
                                    for i, dim_size in enumerate(full_res_shape):
                                        if i == 0:  # time dimension
                                            indices.append(
                                                t if t_size > 1 else slice(None)
                                            )
                                        elif i == 1 and c_size > 1:  # channel dimension
                                            indices.append(c)
                                        elif (
                                            i == len(full_res_shape) - 3
                                        ):  # z dimension
                                            indices.append(z)
                                        else:
                                            indices.append(slice(None))
                                    slice_data = self[tuple(indices)]
                                    # Ensure we're passing a 2D slice
                                    if slice_data.ndim == 2:
                                        writer.push_slice(
                                            slice_data, t_index=t, c_index=c
                                        )
                                    else:
                                        # If we got a higher-dimensional slice, take the first 2D
                                        writer.push_slice(
                                            slice_data.reshape(
                                                -1, slice_data.shape[-1]
                                            ),
                                            t_index=t,
                                            c_index=c,
                                        )

            return str(target_path)

        finally:
            # Reload multiscale metadata
            self._get_multiscale_metadata()
            # Restore original resolution level
            self.resolution_level = original_level

    def __repr__(self) -> str:
        """Simple representation of OME-Zarr multiscale array."""
        return (
            f"OmeZarrArray({self.store_path}) "
            f"shape={self.shape} "
            f"dtype={self.dtype} "
            f"levels={self.ResolutionLevels} "
            f"current_level={self.resolution_level} "
            f"ome_v{self._ome_version}"
        )

    @property
    def summary(self) -> str:
        """Brief summary of all resolution levels in repr-style format."""

        current_resolution = self.resolution_level
        for l in range(self.ResolutionLevels):
            self.resolution_level = l
            dataset = self._get_dataset()
            message = f"Level {l}: shape={dataset.shape}, chunks={dataset.chunks}, dtype={dataset.dtype}"
            if l == current_resolution:
                message += " <-- CURRENT LEVEL"
            print(message)
        self.resolution_level = current_resolution

    def __iter__(self):
        dataset = self._get_dataset()
        if self._timepoint_lock is not None:
            time_idx = self._get_axis_index("time")
            if time_idx is not None:
                # When timepoint is locked, iterate over the first available dimension
                # that comes after the time axis
                next_dim = time_idx + 1 if time_idx + 1 < dataset.ndim else 0
                for i in range(dataset.shape[next_dim]):
                    # Create a full slicing tuple with the locked timepoint
                    key = tuple(slice(None) for _ in range(dataset.ndim))
                    key = key[:time_idx] + (self._timepoint_lock,) + key[time_idx + 1 :]
                    key = key[:next_dim] + (i,) + key[next_dim + 1 :]
                    yield dataset[key]
                return
        else:
            # Normal iteration over first dimension
            for i in range(dataset.shape[0]):
                yield dataset[i]

    def _chunked_z_slices(
        self, t_index: Optional[int] = None, c_index: Optional[int] = None
    ):
        """
        Generator that yields z-slices one at a time while reading entire z-chunks.

        This minimizes disk I/O by reading full z-chunks into memory, then
        yielding individual z-slices from the cached chunk.

        Args:
            t_index: Fixed time index for 5D data (None for 3D or when not fixed)
            c_index: Fixed channel index for 5D data (None for 3D or when not fixed)

        Yields:
            Individual z-slices as numpy arrays
        """
        dataset = self._get_dataset()

        # Determine array shape and chunk structure
        if t_index is not None and c_index is not None:
            # 5D with both t and c fixed: array[t, c, z, y, x]
            total_z = dataset.shape[2]  # z is third dimension
            chunk_size = dataset.chunks[2]  # z-chunk size
        elif t_index is not None or c_index is not None:
            # 5D with one dimension fixed: array[dim, z, y, x]
            total_z = dataset.shape[1]  # z is second dimension
            chunk_size = dataset.chunks[1]  # z-chunk size
        else:
            # 3D case: array[z, y, x]
            total_z = dataset.shape[0]  # z is first dimension
            chunk_size = dataset.chunks[0]  # z-chunk size

        # Process chunks one at a time
        for chunk_start in range(0, total_z, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_z)

            # Build appropriate indexing tuple for chunk read
            if t_index is not None and c_index is not None:
                # Read full z-chunk for fixed (t,c)
                chunk_data = dataset[t_index, c_index, chunk_start:chunk_end]
            elif t_index is not None:
                # Read full z-chunk for fixed t
                chunk_data = dataset[t_index, chunk_start:chunk_end]
            elif c_index is not None:
                # Read full z-chunk for fixed c
                chunk_data = dataset[c_index, chunk_start:chunk_end]
            else:
                # Read full z-chunk for 3D case
                chunk_data = dataset[chunk_start:chunk_end]

            # Yield individual z-slices from the cached chunk (memory access only)
            for local_z in range(chunk_end - chunk_start):
                yield chunk_data[local_z]

    def _safe_read_node_type(self, zarr_json_path: Path) -> Optional[str]:
        """Read zarr.json and return node_type if possible, else None."""
        try:
            with open(zarr_json_path, "r") as f:
                meta = json.load(f)
            nt = meta.get("node_type")
            if isinstance(nt, str):
                return nt
        except Exception:
            pass
        return None

    def omezarr_like(self, target_path: Union[str, Path]) -> "OmeZarrArray":
        """
        Copy only Zarr metadata (v2 and/or v3) from source to target, while pruning recursion
        below array nodes to avoid scanning chunk data.

        Pruning rules:
          - If a directory contains ".zarray" (v2 array node) => copy metadata and STOP recursing.
          - If a directory contains "zarr.json" whose node_type == "array" (v3 array node) => copy and STOP.
        """
        from pathlib import Path
        import shutil

        target_path = Path(target_path)
        source_path = Path(self.store_path)

        # Validate source
        if not source_path.exists():
            raise ValueError(f"Source zarr store does not exist: {source_path}")

        # Remove target if it exists
        if target_path.exists():
            print(f"Target {target_path} already exists, removing it first.")
            shutil.rmtree(target_path)

        # Create parent directories
        target_path.parent.mkdir(parents=True, exist_ok=True)

        def copy_file_if_exists(p: Path) -> None:
            if p.exists() and p.is_file():
                rel = p.relative_to(source_path)
                out = target_path / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)

        def recurse(dirpath: Path) -> None:
            # One scandir per directory; don't rglob.
            try:
                entries = list(os.scandir(dirpath))
                # print(f'{entries=} in {dirpath=}')
            except (FileNotFoundError, NotADirectoryError):
                return

            # Fast presence checks without extra stat() calls:
            # We only need names + (dir/file) for recursion.
            names = {e.name for e in entries}

            # Copy any metadata files present in this directory
            for meta_name in self.ZARR_V2_META | self.ZARR_V3_META:
                # print(f'{meta_name=} in {dirpath=}')
                if meta_name in names:
                    # print('Copying', dirpath / meta_name)
                    copy_file_if_exists(dirpath / meta_name)

            # --- Prune recursion if this is an array node ---

            # Zarr v2 array node
            if ".zarray" in names:
                return

            # Zarr v3 array node
            if "zarr.json" in names:
                node_type = self._safe_read_node_type(dirpath / "zarr.json")
                if node_type == "array":
                    return

            # Otherwise, recurse into subdirectories
            for e in entries:
                if e.is_dir(follow_symlinks=False):
                    recurse(Path(e.path))

        recurse(source_path)
        # Return new OmeZarrArray instance
        return OmeZarrArray(str(target_path))

    def to_dask(
        self,
        *,
        chunks: tuple | str | None = None,
        name: str | None = None,
        lock: bool = False,
        inline_array: bool = True,
    ) -> da.Array:
        """
        Return a dask.array backed by the current resolution level (and respecting timepoint_lock).

        - chunks=None  -> keep Zarr's on-disk chunking (recommended)
        - chunks="auto" or a tuple -> rechunk in Dask (may increase overhead)
        """
        z = self._get_dataset()

        # Build the dask array from the zarr array
        x = da.from_zarr(
            z, chunks=chunks, name=name, inline_array=inline_array, lock=lock
        )

        # If timepoint is locked, slice out that time index in Dask too (so shape matches self.shape)
        if self._timepoint_lock is not None:
            t_idx = self._get_axis_index("time")
            if t_idx is not None:
                sl = [slice(None)] * x.ndim
                sl[t_idx] = self._timepoint_lock
                x = x[tuple(sl)]

        return x
