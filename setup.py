from pathlib import Path
from setuptools import setup, find_packages


def read_requirements(filename: str = "requirements.txt") -> list[str]:
    req_path = Path(__file__).parent / filename
    if not req_path.exists():
        return []

    reqs: list[str] = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "--requirement", "-e")):
            continue
        reqs.append(line)
    return reqs


ROOT = Path(__file__).parent
readme = ROOT / "README.md"
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""

setup(
    name="gps_lib",  # dowolna nazwa (nie musi być unikalna na PyPI)
    version="0.1.0",
    description="GNSS processing library (GitHub-only)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hubert Pierzchała",
    packages=find_packages(exclude=("tests", "docs", "tutorials", "examples")),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=read_requirements(),
)
