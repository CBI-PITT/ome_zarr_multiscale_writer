import zarr
import numpy as np
from typing import List, Dict, Any, cast, Union, Tuple, Any as TypingAny, Optional
from zarr import Array
from ome_zarr_models import open_ome_zarr
from ome_zarr_models.v05.image import Image as ImageV05
from ome_zarr_models.v04.image import Image as ImageV04
from pathlib import Path
from .zarr_tools import Live3DPyramidWriter, PyramidSpec, ChunkScheme, FlushPad


class OmeZarrArray:
    """Convenience class for accessing OME-Zarr v0.4/v0.5 arrays with resolution selection."""

    def __init__(self, store_path: str) -> None:
        self.store_path = store_path
        self.store = zarr.open(store_path, mode="a")
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
        coordinate_transforms = resolution_metadata.get("coordinateTransformations", [])[0]
        if isinstance(coordinate_transforms,dict):
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
                        chunks_per_level.append(self.chunks)  # Will throw error if level missing
                    chunk_scheme = ChunkScheme(hard_coded=chunks_per_level)
                except:
                    chunks_per_level = []
                finally:
                    self.resolution_level = 0

            if chunk_scheme is None:
                if not start_chunks:
                    self.resolution_level = 0 # get chunks for res 0
                    start_chunks = self.chunks

                if not end_chunks and self.ResolutionLevels > 1:
                    self.resolution_level = self.ResolutionLevels - 1 # get chunks for last res
                    try:
                        end_chunks = self.chunks
                    except KeyError:
                        print('The last multiscale is missing, it may have been deleted. Ending chunks is being sent to same value as Resolution Level 0.')
                        end_chunks = start_chunks
                    self.resolution_level = 0 # Restore to res 0
                if not end_chunks:
                    # If only one level exists, set default end chunks
                    end_chunks = (256, 256, 256)

                chunk_scheme = ChunkScheme(base=start_chunks, target=end_chunks)

            # Set up compressor if specified
            if not compressor:
                compressor_obj = self.compressor # Use existing compressor
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
                voxel_size = (1.0,1.0,1.0) # default fallback

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
