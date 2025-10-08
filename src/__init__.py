"""
Reinforced Molecular Dynamics (rMD)
Physics-infused generative machine learning for protein conformational exploration
"""

__version__ = "0.1.0"
__author__ = "István Kolossváry, Rory Coffey (Original Paper); Recreation Team"

from . import collective_variables
from . import data_preparation
from . import autoencoder
from . import training
from . import structure_generation

__all__ = [
    "collective_variables",
    "data_preparation",
    "autoencoder",
    "training",
    "structure_generation",
]
