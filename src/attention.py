import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Computes the softmax function along the last axis of the input array.

    @param x: The input numpy array (e.g., the result of QK^T / sqrt(d_k)).
    @return: The resulting array after applying the softmax function.
    """
    # Stable softmax implementation
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Computes the scaled dot-product attention mechanism as defined in
    "Attention Is All You Need" (Vaswani et al., 2017).

    The computation follows the formula: Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k))V

    @param Q: Query matrix of shape (..., seq_len_q, d_k).
    @param K: Key matrix of shape (..., seq_len_k, d_k).
    @param V: Value matrix of shape (..., seq_len_k, d_v).
    @param mask: Optional attention mask of shape (..., seq_len_q, seq_len_k).
                 If provided, attention scores corresponding to False values will be set to -infinity
                 before the softmax operation.
    @return: The context matrix (output) of shape (..., seq_len_q, d_v).
    """
    # 1. Get the dimension d_k from the shape of K
    d_k = K.shape[-1]
    scale_factor = np.sqrt(d_k)

    # 2. Calculate the unscaled attention scores: QK^T
    # (..., seq_len_q, d_k) @ (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
    scores = np.matmul(Q, np.swapaxes(K, -2, -1))

    # 3. Scale the scores
    scaled_scores = scores / scale_factor

    # 4. Apply optional mask
    if mask is not None:
        # Masking sets logits to a very large negative number so softmax outputs 0 for those positions.
        scaled_scores = np.where(mask, scaled_scores, -1e9)

    # 5. Apply softmax to get attention weights
    attention_weights = softmax(scaled_scores)

    # 6. Multiply weights by V to get the context vector
    # (..., seq_len_q, seq_len_k) @ (..., seq_len_k, d_v) -> (..., seq_len_q, d_v)
    output = np.matmul(attention_weights, V)

    return output