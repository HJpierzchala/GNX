# Windows (pip) developer install

## Prerequisites
- **Python 3.12 (64-bit)**

## Create venv + install (editable)
From the repository root:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip

REM Option A (recommended, reproducible)
pip install -e . -c constraints-win.txt

REM Option B (lets pip choose versions inside allowed ranges)
pip install -e .
```

## Dev tools (tests/lint/notebooks)
```bat
pip install -r requirements-dev.txt -c constraints-win.txt
```

### Notes
- We pin **NumPy < 2.4** because **Numba 0.63.1** does not yet support NumPy 2.4.  (See Numba issue tracking.) 
- Cartopy 0.24.1 provides a Windows CPython 3.12 wheel, so it can be installed via pip on win_amd64.
