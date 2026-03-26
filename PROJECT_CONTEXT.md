# Project Context (session handoff notes)

## Recent Changes

### `iter_chunks()` generator on `OmeZarrArray`
- Added at `zarr_reader.py:817`. Yields `(chunk_index, slices, data)` one chunk at a time in C-order. Respects `timepoint_lock`.

### `to_tiff_stack()` method on `OmeZarrArray`
- Added at `zarr_reader.py:1064`. Exports current resolution level as individual 2D TIFF files (`{basename}_c{C}_z{Z}.tiff`), one per channel per z-layer.
- Uses `tifffile.memmap` to assemble full YX planes from multiple chunk tiles without buffering entire planes in Python heap. Only one zarr chunk is in memory at a time.
- Handles 3D (z,y,x), 4D (c,z,y,x), and 5D (t,c,z,y,x) arrays. Defaults to t=0 with warning if time axis present and no `timepoint_lock`.
- `tifffile` added as hard dependency in `setup.cfg`.

## Open / Future Work

### Compressed TIFF output for `to_tiff_stack()`
- **Status:** Investigated and planned, not yet implemented.
- **Problem:** `tifffile.memmap` only produces uncompressed TIFFs (memory-mapping requires fixed byte offsets).
- **Proposed approach:** Add `compression` and `compression_level` parameters. When compression is requested, use raw `np.memmap` scratch files (OS-paged, not Python heap) to accumulate tiles, then flush each completed plane through `tifffile.imwrite(compression=...)` which reads in strips (~1.5 MB each). When `compression is None`, keep current direct-memmap path for zero overhead.
- **Memory:** Peak Python heap stays at one zarr chunk (~550 MB for typical data). Temporary disk cost is one z-chunk-range of scratch files (~2 GB for typical data), cleaned up per z-chunk-range.
- **Supported codecs:** Anything `tifffile` supports — `zlib`, `lzw`, `zstd`, `lzma`, `jpeg`, etc.
- **Key finding from testing:** `tifffile.imwrite` can read directly from an `np.memmap` array and process it in strips with `rowsperstrip=256`, so strip compression buffer is only ~1.5 MB regardless of plane size.

## Validation Gaps

- No automated tests yet for `iter_chunks` or `to_tiff_stack`. Manual testing covered 3D, 4D (multi-channel), and 5D (time axis) cases with 2x2 YX chunk grids.
