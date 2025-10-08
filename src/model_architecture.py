import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Model
import numpy as np

# --- Configuration based on Paper (S06, S07) ---
VECTOR_LENGTH = 9696  # Input/Output dimension
LATENT_DIM = 3        # Latent space dimension (S07)

def swish_activation(x):
    """
    Implements the Swish activation function: x * sigmoid(x).
    Defined as per paper supplement (S11).
    """
    return x * tf.keras.activations.sigmoid(x)

def build_rMD_autoencoder(input_dim: int, latent_dim: int):
    """
    Builds the base structure of the rMD Autoencoder (Encoder + Decoder).
    This implements the generic structure without the latent loss (Loss1), focusing
    on the prediction loss (Loss2) mechanism (S06).

    @param input_dim: The dimension of the flattened protein structure vector.
    @param latent_dim: The dimension of the latent space.
    @return: The autoencoder Keras Model.
    @raises ValueError: If dimensions are ill-defined.
    """
    if input_dim <= 0 or latent_dim <= 0:
        raise ValueError("Input and latent dimensions must be positive.")

    # Define a simple, symmetric layer structure for demonstration purposes.
    # The paper describes a 'cascading' series of shrinking/expanding layers.
    
    # Common hidden layer size (Representative of 'shrinking' layers)
    HIDDEN_SIZE_1 = 2048
    HIDDEN_SIZE_2 = 512
    
    # --- Encoder ---
    input_layer = Input(shape=(input_dim,), name='input_structure')
    
    # Layer 1 (Shrinking)
    x = Dense(HIDDEN_SIZE_1, name='enc_dense_1')(input_layer)
    x = Activation(swish_activation, name='enc_activation_1')(x)
    
    # Layer 2 (Shrinking further)
    x = Dense(HIDDEN_SIZE_2, name='enc_dense_2')(x)
    x = Activation(swish_activation, name='enc_activation_2')(x)
    
    # Latent Space (S07)
    latent_space = Dense(latent_dim, name='latent_space')(x)

    # --- Decoder ---
    # Layer 3 (Expanding)
    x = Dense(HIDDEN_SIZE_2, name='dec_dense_1')(latent_space)
    x = Activation(swish_activation, name='dec_activation_1')(x)
    
    # Layer 4 (Expanding further)
    x = Dense(HIDDEN_SIZE_1, name='dec_dense_2')(x)
    x = Activation(swish_activation, name='dec_activation_2')(x)
    
    # Output Layer (Reconstruction)
    output_layer = Dense(input_dim, activation='linear', name='reconstruction_output')(x)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer, name='rMD_Base_Autoencoder')
    return autoencoder

if __name__ == '__main__':
    print("--- Testing Autoencoder Model Initialization (T1.2) ---")
    
    # Build the model
    rMD_model = build_rMD_autoencoder(VECTOR_LENGTH, LATENT_DIM)
    
    # Print summary to verify structure and latent dimension
    rMD_model.summary()
    
    # Creating a small file to confirm listing/getting works as well
    with open("tool_check_list.tmp", "w") as f:
        f.write("T1.2 Model Check. Initializing T1.3 and Training Setup.")
        
    print("\\nModel built successfully with a 3D latent space.")
    # Note: We are not compiling or training here, only defining the structure.
    # Compilation and implementation of losses (T1.3, T2.1, T2.2) are next.