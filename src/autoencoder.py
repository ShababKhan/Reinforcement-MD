"""
Autoencoder Module for rMD
Implements the informed autoencoder architecture with dual loss functions
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_encoder(input_dim, latent_dim=3, hidden_layers=None):
    """
    Build the encoder network.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input (9696 for CRBN heavy atoms)
    latent_dim : int
        Dimension of latent space (3 for CV space)
    hidden_layers : list, optional
        List of hidden layer dimensions
        
    Returns
    -------
    encoder : keras.Model
        Encoder model
    """
    if hidden_layers is None:
        # Default architecture from paper (gradually decreasing)
        hidden_layers = [4096, 2048, 1024, 512, 256, 128, 64]
    
    inputs = keras.Input(shape=(input_dim,), name='encoder_input')
    x = inputs
    
    # Build hidden layers with Swish activation
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, name=f'encoder_dense_{i}')(x)
        x = layers.Activation(tf.nn.swish, name=f'encoder_swish_{i}')(x)
    
    # Latent space (no activation)
    latent = layers.Dense(latent_dim, name='latent_space')(x)
    
    encoder = keras.Model(inputs, latent, name='encoder')
    return encoder


def build_decoder(latent_dim, output_dim, hidden_layers=None):
    """
    Build the decoder network.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of latent space
    output_dim : int
        Dimension of output (9696 for CRBN heavy atoms)
    hidden_layers : list, optional
        List of hidden layer dimensions
        
    Returns
    -------
    decoder : keras.Model
        Decoder model
    """
    if hidden_layers is None:
        # Mirror of encoder (gradually increasing)
        hidden_layers = [64, 128, 256, 512, 1024, 2048, 4096]
    
    inputs = keras.Input(shape=(latent_dim,), name='decoder_input')
    x = inputs
    
    # Build hidden layers with Swish activation
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, name=f'decoder_dense_{i}')(x)
        x = layers.Activation(tf.nn.swish, name=f'decoder_swish_{i}')(x)
    
    # Output layer (no activation for regression)
    outputs = layers.Dense(output_dim, name='decoder_output')(x)
    
    decoder = keras.Model(inputs, outputs, name='decoder')
    return decoder


def build_rmd_autoencoder(input_dim=9696, latent_dim=3):
    """
    Build the complete rMD informed autoencoder.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input vectors
    latent_dim : int
        Dimension of latent/CV space
        
    Returns
    -------
    model : keras.Model
        Complete autoencoder model
    encoder : keras.Model
        Encoder submodel
    decoder : keras.Model
        Decoder submodel
    """
    encoder = build_encoder(input_dim, latent_dim)
    decoder = build_decoder(latent_dim, input_dim)
    
    # Complete autoencoder
    inputs = keras.Input(shape=(input_dim,), name='input')
    latent = encoder(inputs)
    outputs = decoder(latent)
    
    autoencoder = keras.Model(inputs, outputs, name='rmd_autoencoder')
    
    return autoencoder, encoder, decoder
