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

    def _get_dataset(self):
        """Get the current resolution level's dataset."""
        return self.store[self._get_dataset_path()]

    def _get_dataset_path(self) -> str:
        """Get the current resolution level's dataset path."""
        return self._scale_datasets[self.resolution_level]["path"]

    @property
    def resolution_level(self) -> int:
        return self._resolution_level

    @resolution_level.setter
    def resolution_level(self, level: int) -> None:
        if level < 0 or level >= len(self._scale_datasets):
            raise ValueError(f"Level must be 0-{len(self._scale_datasets) - 1}")
        self._resolution_level = level

    @property
    def shape(self) -> tuple:
        return self._get_dataset().shape

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

    def __getitem__(self, key):
        return self._get_dataset()[key]

    def __iter__(self):
        dataset = self._get_dataset()
        for i in range(dataset.shape[0]):
            yield dataset[i]
