"""OpenDrug package configuration.

The package is primarily a research framework; we keep the install recipe
minimal so that users may layer in their own CUDA / DGL builds as needed.
The ``[docs]`` extra installs everything required to build the Sphinx
documentation (also pulled in via ``docs/requirements.txt``).
"""
from pathlib import Path
from setuptools import find_packages, setup

README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8") \
    if (Path(__file__).parent / "README.md").exists() else ""

DOCS_REQUIRES = [
    "sphinx>=7.0,<9.0",
    "sphinx-rtd-theme>=2.0",
    # 3.12.x triggers native crashes when introspecting torch /
    # torch_geometric on newer Python interpreters.
    "sphinx-autodoc-typehints>=2.0,<3.5",
    "sphinx-copybutton>=0.5",
    "myst-parser>=3.0",
    "sphinx-togglebutton>=0.4",
    "pygments>=2.17",
]

setup(
    name="opendrug",
    version="0.1.0",
    description="Unified drug-related prediction framework (DDI / DTA / DTI / PPI / Cold-start).",
    long_description=README,
    long_description_content_type="text/markdown",
    author="OpenDrug Contributors",
    license="MIT",
    url="https://github.com/your-org/opendrug",
    packages=find_packages(exclude=("tests", "docs", "examples")),
    python_requires=">=3.9",
    include_package_data=True,
    extras_require={
        # Build the Sphinx docs locally.
        "docs": DOCS_REQUIRES,
        # Convenience: install the docs extra + lint/test tools.
        "dev": DOCS_REQUIRES + [
            "pytest>=7.0",
            "ruff>=0.4",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)