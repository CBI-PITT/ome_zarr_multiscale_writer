# Repository Guidelines

## Project Structure & Module Organization
- Core package lives in `ome_zarr_multiscale_writer/` with three modules: `write.py` (user-facing helpers for writing pyramids), `zarr_tools.py` (chunking/sharding, writers, codecs), and `helpers.py` (math/downsampling utilities).
- Packaging metadata is in `setup.cfg` and `setup.py`; version is tracked in `_version.py`.
- No dedicated `tests/` directory yet—add new test modules there (`tests/test_<feature>.py`) as you contribute features.

## Build, Test, and Development Commands
- Create and activate a Python 3.12+ environment, then install deps in editable mode: `python -m pip install -e .`.
- Run an ad-hoc script entry point by importing the writer (example): `python - <<'PY'\nfrom ome_zarr_multiscale_writer.write import write_ome_zarr_multiscale\n# TODO: call with your numpy array\nPY`.
- When tests exist, run them with `python -m pytest`; add `-q` for concise output.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; prefer snake_case for functions/variables and PascalCase for classes/dataclasses.
- Keep numpy array operations vectorized; avoid unnecessary Python loops in hot paths.
- Use type hints (already present in public APIs) and explicit tuples for shapes `(z, y, x)`.
- Keep module-level defaults near the top of files (see `write.py`) and document new constants.

## Testing Guidelines
- Target fast, deterministic tests using `pytest`; prefer synthetic arrays over large fixtures.
- Name tests `test_<behavior>` and group by feature or module (e.g., `tests/test_zarr_tools.py`).
- When adding pyramid/chunk logic, include shape/chunk/shard assertions and edge cases (odd dimensions, small volumes, non-default voxel sizes).

## Commit & Pull Request Guidelines
- Commit messages are short and present-tense (see existing history); keep each commit focused on one change.
- PRs should include: what changed, why it is needed, and how to exercise it. Link issues when applicable and note any API changes or defaults you altered.
- Add before/after notes when touching performance-sensitive paths (chunking, compression). Mention any new config knobs or defaults.

## Security & Configuration Tips
- Avoid committing paths to external storage (e.g., lab share mounts); use placeholders in examples.
- Keep dependencies minimal—`zarr` (and its transitive `numcodecs`) are required. If you add optional extras, guard imports and document flags clearly.
