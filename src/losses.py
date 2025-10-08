# src/losses.py

"""
Module for implementing the specialized loss functions required for the rMD autoencoder.

This module contains the implementations for Loss2 (Reconstruction Loss / RMSD)
and Loss1 (Latent Space Mapping Loss) as described in the rMD paper.
"""

import numpy as np
import tensorflow as tf

# --- Constants based on paper and T1.1 setup ---
# The input vector length is 9696 (all heavy atoms coordinates)
INPUT_VECTOR_DIM = 9696
LATENT_DIM = 3


def loss2_rmsd(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the square of the Root Mean Square Deviation (RMSD) between true
    and predicted structures, which is minimized by the 'predLoss' layer
    (Loss2). This implementation uses the squared distance for easier backpropagation,
    which is equivalent to minimizing RMSD.

    The paper states: "Loss2 loss function, which is the average root mean square
    distance (RMSD) between inputs and their predicted outputs". Since we are
    dealing with flattened vectors, the MSE between them is directly proportional
    to the squared RMSD, fulfilling the requirement.

    @param y_true: The true input structure coordinates (flattened vector).
    @param y_pred: The predicted (reconstructed) structure coordinates.
    @return: The mean squared error loss value (proportional to squared RMSD).
    """
    # Calculate Mean Squared Error (MSE) between the flattened vectors.
    # MSE = (1/N) * sum((y_true - y_pred)^2)
    # This is proportional to RMSD^2, and minimizing it minimizes RMSD.
    error = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return error


def loss1_cv_mapping(ls_coords: tf.Tensor, cv_targets: tf.Tensor) -> tf.Tensor:
    """
    Calculates the loss enforcing a one-to-one correspondence between the
    Low-Dimensional Latent Space (LS) coordinates and the Collective Variable (CV)
    coordinates. This loss is minimized by the 'latentLoss' layer (Loss1).

    The target for Loss1 is the distance between the LS coordinates and the CV
    coordinates in the same dimension space.

    @param ls_coords: The projected coordinates from the Autoencoder's latent space.
    @param cv_targets: The target Collective Variable coordinates (e.g., the CVs).
    @return: The mean squared error, where LS coordinates correspond to CV values.
    """
    # Check dimensions based on paper: LS and CV space are both 3D.
    if ls_coords.shape[-1] != LATENT_DIM or cv_targets.shape[-1] != LATENT_DIM:
        raise ValueError(
            f"Expected latent and CV dimensions to be {LATENT_DIM}, "
            f"but got LS shape {ls_coords.shape} and CV shape {cv_targets.shape}"
        )

    # Calculate MSE between the latent space points and the target CV points.
    error = tf.keras.losses.mean_squared_error(cv_targets, ls_coords)
    return error