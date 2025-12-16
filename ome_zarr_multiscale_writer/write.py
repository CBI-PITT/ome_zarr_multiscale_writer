# import typer
from typing import Tuple
from pathlib import Path
import os
import numpy as np

from zarr.codecs import BloscCodec, BloscShuffle, ShardingCodec

from helpers import plan_levels, compute_xy_only_levels
from zarr_tools import ChunkScheme, FlushPad, PyramidSpec, Live3DPyramidWriter

# app = typer.Typer()

# A function to simplify writing a numpy array to an OME-Zarr multiscale pyramid.
MAX_WORKERS = min(8, os.cpu_count() or 4)
OME_VERSION = "0.4" # "0.5"
COMPRESSOR = 'zstd'  # default compressor
COMPRESSION_LEVEL = 5  # default compression level
START_CHUNKS = (256, 256, 256)  # default starting chunk shape (z,y,x)
END_CHUNKS = (256, 256, 256)    # default ending chunk shape (z,y,x)
VOXEL_SIZE = (1,1,1)  # default voxel size (z,y,x) in microns
GENERATE_MULTISCALES = True  # whether to generate multiscale pyramid

TRANSLATION = (0,0,0)  # default translation
INGEST_QUEUE_SIZE = 8
MAX_INFLIGHT_CHUNKS = MAX_WORKERS // 2 if MAX_WORKERS >=2 else 1

ASYNC_CLOSE = False

def write_ome_zarr_multiscale(
        data: np.ndarray,
        path: Path | str,
        voxel_size: Tuple[int, int, int]=VOXEL_SIZE,
        generate_multiscales: bool=GENERATE_MULTISCALES,
        start_chunks: Tuple[int, int, int]=START_CHUNKS,
        end_chunks: Tuple[int, int, int]=END_CHUNKS,
        compressor: str=COMPRESSOR,
        compression_level: int=COMPRESSION_LEVEL,
        ingest_queue_size: int = INGEST_QUEUE_SIZE,
        max_inflight_chunks: int | None = MAX_INFLIGHT_CHUNKS,
        shard_shape: Tuple[int, int, int] | None = None,
        translation: Tuple[int,int,int] = TRANSLATION,
        ome_version: str = OME_VERSION,
        max_workers: int=MAX_WORKERS,
        async_close: bool = ASYNC_CLOSE,
        flush_pad: FlushPad = FlushPad.DUPLICATE_LAST
        ):

    """Write a numpy array to an OME-Zarr multiscale pyramid."""

    shard_shape = shard_shape or data.shape # default to full shape if not provided, only applies to Zarr v3

    # ZARR Writer setup
    Z_EST, Y, X = data.shape

    xy_levels = compute_xy_only_levels(voxel_size)

    if generate_multiscales:
        levels = plan_levels(Z_EST, Y, X, xy_levels, min_dim=64)
    else:
        levels = 1

    spec = PyramidSpec(
        z_size_estimate=Z_EST,  # big upper bound; we'll truncate at the end
        y=Y, x=X, levels=levels,
    )

    scheme = ChunkScheme(base=start_chunks, target=end_chunks)

    if compressor:
        compressor = BloscCodec(cname=compressor, clevel=compression_level, shuffle=BloscShuffle.bitshuffle)

    with Live3DPyramidWriter(
        spec,
        voxel_size=voxel_size,
        path=path,
        max_workers=max_workers,
        chunk_scheme=scheme,
        compressor=compressor,
        shard_shape=shard_shape,
        flush_pad=flush_pad,  # keeps alignment, no RMW
        async_close=async_close,
        translation=translation,
        ome_version=ome_version,
        ingest_queue_size=ingest_queue_size,
        max_inflight_chunks=max_inflight_chunks,
    ) as omezarr_writer:

        for idx, zplane in enumerate(data):
            print(f'Pushing slice {idx+1}/{data.shape[0]}')
            omezarr_writer.push_slice(zplane)


def generate_multiscales_from_omezarr(
        source_path: Path | str,
        target_path: Path | str,
        voxel_size: Tuple[int, int, int]=VOXEL_SIZE,
        start_chunks: Tuple[int, int, int]=START_CHUNKS,
        end_chunks: Tuple[int, int, int]=END_CHUNKS,
        compressor: str=COMPRESSOR,
        compression_level: int=COMPRESSION_LEVEL,
        ingest_queue_size: int = INGEST_QUEUE_SIZE,
        max_inflight_chunks: int | None = MAX_INFLIGHT_CHUNKS,
        shard_shape: Tuple[int, int, int] | None = None,
        translation: Tuple[int,int,int] = TRANSLATION,
        ome_version: str = OME_VERSION,
        max_workers: int=MAX_WORKERS,
        async_close: bool = ASYNC_CLOSE,
        flush_pad: FlushPad = FlushPad.DUPLICATE_LAST
        ):
    """Generate multiscale pyramid from existing OME-Zarr dataset."""
    import zarr

    source_group = zarr.open(source_path, mode='r')
    full_res_array = source_group['0']

    write_ome_zarr_multiscale(
        data=full_res_array,
        path=target_path,
        voxel_size=voxel_size,
        generate_multiscales=True,
        start_chunks=start_chunks,
        end_chunks=end_chunks,
        compressor=compressor,
        compression_level=compression_level,
        ingest_queue_size=ingest_queue_size,
        max_inflight_chunks=max_inflight_chunks,
        shard_shape=shard_shape,
        translation=translation,
        ome_version=ome_version,
        max_workers=max_workers,
        async_close=async_close,
        flush_pad=flush_pad
    ))


if __name__ == '__main__':
    import numpy as np

    output = '/CBI_FastStore/Acquire/MesoSPIM/alan-test/output_example.ome.zarr'
    output = 'z:/Acquire/MesoSPIM/alan-test/output_example4.ome.zarr'

    # Example usage
    # data = np.random.randint(0, 65535, size=(2000, 1024, 1024), dtype=np.uint16)

    class FakeData():
        '''A fake data source that generates random uint16 data on-the-fly.
        Has an interator function to allow indexing.'''

        def __init__(self, shape:Tuple[int, int, int]):
            self.shape = shape
            self.data = np.random.randint(0, 65535, size=(self.shape[1], self.shape[2]), dtype=np.uint16)
        def __getitem__(self, idx):
            z = idx
            np.random.seed(z)  # for reproducibility
            self.data[:] = np.random.randint(0, 65535, size=(self.shape[1], self.shape[2]), dtype=np.uint16)
            return self.data
        def __iter__(self):
            for z in range(self.shape[0]):
                yield self[z]
        def __len__(self):
            return self.shape[0]

    data = FakeData((2000, 20000, 24000))


    write_ome_zarr_multiscale(
        data,
        path=output,
        voxel_size=(2, 0.5, 0.5),
        generate_multiscales=True,
        start_chunks=(32, 256, 256),
        end_chunks=(256, 256, 256),
        compressor='zstd',
        compression_level=5,
        ome_version="0.4",
    )
