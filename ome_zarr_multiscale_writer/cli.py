from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import typer

from .write import (
    ASYNC_CLOSE,
    COMPRESSOR,
    COMPRESSION_LEVEL,
    END_CHUNKS,
    INGEST_QUEUE_SIZE,
    MAX_INFLIGHT_CHUNKS,
    MAX_WORKERS,
    OME_VERSION,
    START_CHUNKS,
    TRANSLATION,
    generate_multiscales_from_omezarr,
)
from .zarr_tools import FlushPad

app = typer.Typer(
    add_completion=False,
    help="Build OME-Zarr multiscale pyramids from existing level 0 data.",
)


@app.command("generate")
def generate(
    source: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        resolve_path=True,
        help="Path to an existing OME-Zarr store that already contains level 0 data.",
    ),
    target: Optional[Path] = typer.Option(
        None,
        "--target",
        "-t",
        file_okay=True,
        dir_okay=True,
        resolve_path=True,
        help="Optional destination store; defaults to the source store.",
    ),
    voxel_size: Optional[Tuple[float, float, float]] = typer.Option(
        None,
        "--voxel-size",
        metavar="Z Y X",
        help="Override voxel size in physical units (z y x). Falls back to store metadata.",
    ),
    start_chunks: Tuple[int, int, int] = typer.Option(
        START_CHUNKS,
        "--start-chunks",
        metavar="Z Y X",
        help="Chunk shape for level 0 (z y x).",
    ),
    end_chunks: Tuple[int, int, int] = typer.Option(
        END_CHUNKS,
        "--end-chunks",
        metavar="Z Y X",
        help="Chunk shape target for the coarsest level (z y x).",
    ),
    shard_shape: Optional[Tuple[int, int, int]] = typer.Option(
        None,
        "--shard-shape",
        metavar="Z Y X",
        help="Optional shard shape for Zarr v3 stores (z y x).",
    ),
    translation: Tuple[int, int, int] = typer.Option(
        TRANSLATION,
        "--translation",
        metavar="Z Y X",
        help="Physical translation applied to the dataset (z y x).",
    ),
    compressor: str = typer.Option(
        COMPRESSOR,
        "--compressor",
        help="Compressor to use for all pyramid levels (e.g. zstd, lz4).",
    ),
    compression_level: int = typer.Option(
        COMPRESSION_LEVEL,
        "--compression-level",
        help="Blosc compression level (higher is slower but smaller).",
    ),
    ingest_queue_size: int = typer.Option(
        INGEST_QUEUE_SIZE,
        "--ingest-queue-size",
        help="Number of planes buffered before writers consume them.",
    ),
    max_inflight_chunks: Optional[int] = typer.Option(
        MAX_INFLIGHT_CHUNKS,
        "--max-inflight-chunks",
        help="Throttle for concurrent chunk writes; defaults to a per-worker heuristic.",
    ),
    ome_version: str = typer.Option(
        OME_VERSION,
        "--ome-version",
        help="OME-NGFF metadata version emitted into the target store.",
    ),
    async_close: bool = typer.Option(
        ASYNC_CLOSE,
        "--async-close/--no-async-close",
        help="Finalize writers asynchronously to overlap with ingestion.",
    ),
    max_workers: int = typer.Option(
        MAX_WORKERS,
        "--max-workers",
        help="Maximum number of worker threads used for chunk generation.",
    ),
    flush_pad: FlushPad = typer.Option(
        FlushPad.DUPLICATE_LAST,
        "--flush-pad",
        case_sensitive=False,
        help="How partially filled chunks are padded before writing.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Rebuild multiscales even if the destination already contains them.",
    ),
) -> None:
    """Generate or update the multiscale hierarchy for an existing OME-Zarr store."""
    destination = target or source
    try:
        result = generate_multiscales_from_omezarr(
            source_path=source,
            target_path=destination,
            voxel_size=voxel_size,
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
            flush_pad=flush_pad,
            force=force,
        )
    except Exception as exc:  # pragma: no cover - defensive failure path
        typer.secho(f"Failed to generate multiscales: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.secho(f"Multiscales written to {result}", fg=typer.colors.GREEN)


def main() -> None:
    """Entrypoint for python -m execution."""
    app()


if __name__ == "__main__":
    main()
