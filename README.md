# GNX-py
*A modular GNSS analysis library for PPP, SISRE/URE evaluation, and ionospheric modelling*

---

## Overview

**GNX-py** is a Python library for advanced GNSS data processing. It provides a modular architecture covering three major domains in high-precision GNSS analysis:

- **PPP** – Precise Point Positioning (code, phase, uncombined PPP)
- **SISRE / URE** – Satellite orbit and clock quality assessment
- **Ionosphere** – STEC, VTEC, GIM, spherical harmonics modelling, DCB estimation, and ionospheric monitoring

This project is designed for researchers, engineers, and students requiring high-accuracy GNSS tools in pure Python.

---

## Features

### Precise Point Positioning (PPP)
- Code-based, phase-based and uncombined PPP
- Kalman filter framework (EKF)
- Satellite and receiver corrections (PCO/PCV, tides, troposphere, relativity)

### SISRE / URE Analysis
- Orbit and clock comparison
- R/A/C decomposition
- SISRE and URE computation
- Quality control metrics

### Ionospheric Tools
- STEC computation
- VTEC estimation
- Kriging 
- Ionospheric activity monitoring tools

---

## Repository Structure

```
GNX/
├─ gnx_py/        # Python package (import gnx_py)
│   ├─ ppp/
│   ├─ ionosphere/
│   ├─ orbits/
│   └─ core modules
├─ tutorials/     # Lightweight example notebooks
├─ setup.py       # Package installer
└─ README.md
```

**Large files such as full tutorial bundles (ZIP > 100 MB)** are stored as *Release Assets* under GitHub Releases.

---

## Installation

### Install in development mode

```
git clone https://github.com/HJpierzchala/GNX.git
cd GNX
pip install -e .
```

### Usage example

```python
import gnx_py as gnx 
from gnx_py.ppp import PPPConfig, PPPSession
from gnx_py.ionosphere import TECConfig STECSession
from gnx_py.orbits import SISConfig, SISEngine
```

---

## Tutorials & Documentation

A complete tutorial bundle is available under:

👉 **Releases → Latest → tutorials.zip**

This includes:
- PPP workflows
- Orbit comparison & SISRE analysis
- Ionospheric modelling (STEC → VTEC → Kriging, Calibration & Activity indices)
- Visualization tools

Lightweight example notebooks are also available in the `tutorials/` directory.

---

## Development Setup

```
git clone https://github.com/HJpierzchala/GNX.git
cd GNX
pip install -e ".[dev]"
```

Recommended Python version: **3.10+**

---

## Roadmap

Planned features:
- PPP-AR full pipeline
- Real-time PPP
- Global/regional GIM estimation (SH + KF)
- Advanced ionospheric indices (ROTI, SIDX, GIX)
- Cython & parallel acceleration

Contributions and suggestions are welcome.

---

## Contributing

Pull requests are welcome. You may contribute:
- new modules
- improved algorithms
- documentation & notebooks
- SISRE & ionosphere tools

For ideas or issues, open a GitHub Issue.

---

## License

Released under the **MIT License**.
See `LICENSE` for full terms.

---

## Citation

If you use GNX-py in academic work, please cite:

```
Pierzchała, H. (2025). GNX-py: A Python toolbox for PPP, SISRE and ionospheric modelling.
```

A `CITATION.cff` file will be added in a future release.

---

## Acknowledgements

GNX-py aligns with standards and products from:
- IGS (SP3, CLK, RINEX)
- CODE, JPL, WHU
- Classical PPP & ionospheric modelling literature

---

### ⭐ If this project is useful, consider giving it a star on GitHub!
