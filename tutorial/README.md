# GNX Tutorial Manual Build

This directory contains the Jupyter tutorial notebooks that can be built into a single GNX-py manual PDF.

## Requirements

Install GNX-py with the tutorial extra from the repository root:

```bash
python -m pip install -e ".[tutorial]"
```

The build also requires:

- `nbconvert` (with `nbclient`), `nbformat` and `ipykernel`, installed by the `tutorial` extra (which also brings
  JupyterLab for working with the notebooks interactively), and `PyYAML` (`python -m pip install pyyaml` if it is
  missing),
- Pandoc,
- a LaTeX engine, preferably `xelatex`,
- the bibliography file `bibliography.bib`,
- the LaTeX title page `titlepage.tex`.

The map figures (ionosphere chapters) draw coastlines and borders from Natural Earth through Cartopy, which
downloads these files on first use into its data directory (`cartopy.config["data_dir"]`). For an offline build,
fetch them once in advance:

```bash
python -c "import cartopy.io.shapereader as s; [s.natural_earth(resolution=r, category=c, name=n) for r in ('110m', '50m') for c, n in (('physical', 'land'), ('physical', 'ocean'), ('physical', 'coastline'), ('cultural', 'admin_0_boundary_lines_land'))]"
```

## Build The Manual

The notebooks are stored without outputs, so the manual is built by executing them. From the repository root:

```bash
python tutorial/build_manual.py --clean --execute
```

The final PDF is written to:

```text
tutorial/_build/GNX_manual.pdf
```

The complete build executes all seven chapters on `sample_data/` and the STEC network in
`tutorial/ionosphere/network/`; it does not download anything apart from the Natural Earth files mentioned above.
The `GNX_manual.pdf` in the repository root is a copy of this output.

For a faster smoke build that executes only the lighter chapters:

```bash
python tutorial/build_manual.py --clean --execute --skip-heavy --output GNX_manual_smoke.pdf
```

## Useful Options

- `--execute` executes notebook copies before export.
- `--no-execute` uses existing notebook outputs (the stored notebooks have none, so this only makes sense for
  notebooks you executed and saved yourself).
- `--skip-heavy` omits chapters marked as heavy in `manual_config.yaml`. Do not use this option for the final complete manual.
- `--keep-intermediate` keeps the isolated runtime copy under `tutorial/_build/runtime_repo`.
- `--output NAME.pdf` changes the final PDF filename.
- `--chapters introduction ppp orbits` builds only selected chapter ids.
- `--verbose` prints the Pandoc command and extra progress information.
- `--build-dir DIR` changes the build directory; keep it inside the repository (the notebooks look for the
  repository root, the folder with `pyproject.toml`, above their own location).

## Chapter Order

The chapter order is controlled by:

```text
tutorial/manual_config.yaml
```

To add a new notebook, add a new entry under `manual.chapters` with:

- `id`,
- `title`,
- `path`,
- `heavy`,
- `execute`.

Heavy chapters are skipped when `--skip-heavy` is used. This is intended for data-dependent or longer-running notebooks such as activity-index and kriging workflows.

## Bibliography

Pandoc citeproc uses:

```text
tutorial/bibliography.bib
```

The build script checks citation keys in notebook Markdown cells before building. If a citation key is missing from the `.bib` file, the build stops with a readable error instead of guessing.

The bibliography is inserted at the end through `references.md`.

## Title Page

The title page is defined by:

```text
tutorial/titlepage.tex
```

The build script includes it as LaTeX before the manual body. A prepared copy is written under `tutorial/_build/`, so source files are not rewritten during the build.

## PDF Export Normalization

The PDF build applies a few export-only normalizations to make notebook output usable in a manual:

- each notebook becomes one chapter, and notebook headings are demoted to sections,
- notebook-local `Table of Contents` sections are omitted because the manual has a global table of contents,
- pandas/HTML and Markdown tables are converted to compact LaTeX tables,
- very wide tables are reduced to a readable PDF preview with omitted-column notes,
- notebook image outputs are converted to non-floating LaTeX figures with curated captions,
- very long plain-text object representations are shortened in the PDF copy,
- absolute paths of the build machine in notebook outputs are shown as `<repo>` (repository) and `~` (home
  directory).

These transformations are applied only to files under `tutorial/_build/`; source notebooks are not rewritten.

## Build Artifacts

All build artifacts are written under:

```text
tutorial/_build/
```

Important subdirectories:

- `executed/` contains executed or copied notebook inputs used for export,
- `chapters/` contains Markdown chapters and extracted images,
- `logs/` contains build plans, reports, and Pandoc logs,
- `runtime_repo/` is a temporary isolated runtime used during `--execute` builds.
  It links `gnx_py/`, `sample_data/`, `examples/` and the STEC network of the repository; where links are not
  allowed (Windows without Developer Mode or administrator rights) it copies them, about 120 MB.

The original notebooks are not overwritten during execution.
