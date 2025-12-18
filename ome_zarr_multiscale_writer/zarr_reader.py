import zarr
import numpy as np
from typing import List, Dict, Any, cast, Union, Tuple, Any as TypingAny
from zarr import Array


class OmeZarrArray:
    """Convenience class for accessing OME-Zarr v0.4/v0.5 arrays with resolution selection."""

    def __init__(self, store_path: str) -> None:
        self.store = zarr.open(store_path, mode="r")

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
            raise ValueError(
                "Invalid multiscales format: expected list with dict as first element"
            )
        if not self._scale_datasets:
            raise ValueError("Invalid multiscales: missing datasets")
        self._resolution_level: int = 0
        self._timepoint_lock: int | None = None

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
        return self._get_dataset().nchunks_initialized

    @property
    def cdata_shape(self) -> tuple:
        return self._get_dataset().cdata_shape

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
