from setuptools import setup, find_packages

setup(
    name="gnx-py",  # nazwa projektu (np. na PyPI)
    version="0.1.0",
    description="GNX-py: GNSS tools for PPP, SISRE/URE, and ionosphere modelling",
    author="Hubert Pierzchała",
    author_email="your_email@example.com",
    url="https://github.com/hpierzchala1/GNX",  # popraw na swój login jeśli inny
    packages=find_packages(),  # znajdzie 'gnx_py' i podpakiety
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "xarray",
        # dorzuć swoje zależności
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)
