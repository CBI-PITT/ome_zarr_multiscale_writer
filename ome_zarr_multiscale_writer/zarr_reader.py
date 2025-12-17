import zarr
import numpy as np
from typing import List, Dict, Any, cast


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

        self._scale_datasets: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], multiscales[0].get("datasets", [])
        )
        if not self._scale_datasets:
            raise ValueError("Invalid multiscales: missing datasets")
        self._resolution_level: int = 0

    @property
    def resolution_level(self) -> int:
        return self._resolution_level

    @resolution_level.setter
    def resolution_level(self, level: int) -> None:
        if level < 0 or level >= len(self._scale_datasets):
            raise ValueError(f"Level must be 0-{len(self._scale_datasets) - 1}")
        self._resolution_level = level

    def __array__(self, dtype: Any = None) -> np.ndarray:
        dataset_path = self._scale_datasets[self.resolution_level]["path"]
        return np.array(self.store[dataset_path], dtype=dtype)

    @property
    def shape(self) -> tuple:
        dataset_path = self._scale_datasets[self.resolution_level]["path"]
        return self.store[dataset_path].shape

    @property
    def dtype(self) -> np.dtype:
        dataset_path = self._scale_datasets[self.resolution_level]["path"]
        return self.store[dataset_path].dtype

    @property
    def ndim(self) -> int:
        dataset_path = self._scale_datasets[self.resolution_level]["path"]
        return self.store[dataset_path].ndim

    @property
    def size(self) -> int:
        dataset_path = self._scale_datasets[self.resolution_level]["path"]
        return self.store[dataset_path].size

    @property
    def chunks(self) -> tuple:
        dataset_path = self._scale_datasets[self.resolution_level]["path"]
        return self.store[dataset_path].chunks

    @property
    def compressor(self) -> object:
        dataset_path = self._scale_datasets[self.resolution_level]["path"]
        dataset = self.store[dataset_path]
        # Handle Zarr v2 vs v3 compressor access
        if hasattr(dataset, "metadata") and dataset.metadata.zarr_format == 2:
            return dataset.compressor
        elif hasattr(dataset, "compressors"):
            return dataset.compressors
        return None

    @property
    def nchunks(self) -> int:
        dataset_path = self._scale_datasets[self.resolution_level]["path"]
        return self.store[dataset_path].nchunks_initialized

    @property
    def cdata_shape(self) -> tuple:
        dataset_path = self._scale_datasets[self.resolution_level]["path"]
        return self.store[dataset_path].cdata_shape

    def __getitem__(self, key):
        dataset_path = self._scale_datasets[self.resolution_level]["path"]
        return self.store[dataset_path][key]

    def __iter__(self):
        dataset_path = self._scale_datasets[self.resolution_level]["path"]
        dataset = self.store[dataset_path]
        for i in range(dataset.shape[0]):
            yield dataset[i]
