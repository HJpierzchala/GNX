# Windows (pip) install

## Prerequisites
- **Python 3.12 (64-bit)**

## Create venv + install (editable)
From the repository root:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -e .
```

## Tutorials and developer tools
```bat
pip install -e ".[tutorial]"
pip install -e ".[dev]"
```

Check the installation with the SPP example on the sample data (one hour):

```bat
python examples\run_spp_example.py
```

### Notes
- `pyproject.toml` is the single source of truth of the requirements; `requirements-runtime.txt` and
  `requirements-tutorial.txt` mirror it.
- We pin **NumPy < 2.4** because **Numba 0.63.1** does not yet support NumPy 2.4.  (See Numba issue tracking.)
- Cartopy 0.24.1 provides a Windows CPython 3.12 wheel, so it can be installed via pip on win_amd64.
- The continuous integration installs the package on Windows with Python 3.12, imports it and runs the SPP
  example (one hour); the full validation of v1.0.0 ran on macOS.
