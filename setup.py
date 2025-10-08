"""
Reinforced Molecular Dynamics (rMD) Setup Script
Physics-infused generative machine learning for protein conformational exploration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="reinforced-molecular-dynamics",
    version="0.1.0",
    author="István Kolossváry, Rory Coffey (Original Paper); Recreation Team",
    author_email="",
    description="Physics-infused generative ML model for protein conformational exploration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShababKhan/Reinforcement-MD",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
            "sphinx>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rmd-train=rmd.training:main",
            "rmd-generate=rmd.structure_generation:main",
        ],
    },
)
