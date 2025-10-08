import numpy as np
import unittest
import sys
import os

# Add the source directory to the path for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the module containing the functions to test
try:
    from attention import scaled_dot_product_attention, softmax
except ImportError:
    print("Error: Could not import 'attention' module. Make sure src/attention.py exists.")
    # Create a dummy test case if import fails to ensure test runner completes
    class DummyTest(unittest.TestCase):
        def test_dummy_pass(self):
            self.assertTrue(True)
    sys.exit(1)

class TestAttention(unittest.TestCase):
    """Unit tests for functions in attention.py based on Vaswani et al., 2017."""

    def setUp(self):
        """Set up standard input shapes and values for testing."""
        self.d_k = 64  # Dimension of keys/queries
        self.scale_factor = 1.0 / np.sqrt(self.d_k)
        self.batch_size = 2
        self.seq_len = 10
        self.d_model = 512 # Often d_model is larger, but for isolated test, we use d_k

        # Q, K, V shapes: (batch_size, seq_len, d_k) or (batch_size, num_heads, seq_len, d_k)
        # For this function, we test assuming Q, K, V are already split into one head shape: (B, S, Dk)
        self.Q = np.random.rand(self.batch_size, self.seq_len, self.d_k).astype(np.float32)
        self.K = np.random.rand(self.batch_size, self.seq_len, self.d_k).astype(np.float32)
        self.V = np.random.rand(self.batch_size, self.seq_len, self.d_k).astype(np.float32)

    def test_softmax_numerical_stability(self):
        """
        Test the custom softmax implementation ensures outputs sum to ~1 on the last axis.
        """
        input_data = np.array([
            [1000, 1001, 1002],
            [-50, 0, 50]
        ], dtype=np.float32)

        result = softmax(input_data)

        # Check shape: (2, 3)
        self.assertEqual(result.shape, (2, 3))

        # Check sum to 1 (or very close) across the last dimension
        sums = np.sum(result, axis=-1)
        np.testing.assert_allclose(sums, [1.0, 1.0], rtol=1e-6)

        # Check stability (i.e., large inputs don't result in NaN/Inf)
        self.assertTrue(np.all(np.isfinite(result)))


    def test_scaled_dot_product_attention_shapes(self):
        """
        Test that the output shape matches the V matrix shape (B, S, Dk).
        """
        output = scaled_dot_product_attention(self.Q, self.K, self.V, self.d_k)
        
        # Expected shape must match V: (Batch, SeqLen, Dk)
        expected_shape = self.V.shape
        
        self.assertEqual(output.shape, expected_shape)

    def test_scaled_dot_product_attention_output_values(self):
        """
        Test that the output vector norms are expected (sum of attention weights applied to V).
        While exact check is hard without fixed inputs, we check that outputs are not zero
        and that weights sum to 1 per query position.
        """
        output = scaled_dot_product_attention(self.Q, self.K, self.V, self.d_k)
        
        # Check that the output is not just the input V (it should be a weighted sum)
        self.assertFalse(np.allclose(output, self.V))

        # Check that the attention weights (before applying V) sum to 1 across the seq_len dimension
        # We must re-calculate to check the weights only: softmax(QK^T / sqrt(d_k))
        scores = np.matmul(self.Q, np.transpose(self.K, (0, 2, 1))) * self.scale_factor
        attention_weights = softmax(scores) # Shape (B, S_q, S_k)
        
        weights_sum = np.sum(attention_weights, axis=-1)
        # Check that for every query vector, the weights applied to all key vectors sum to 1
        np.testing.assert_allclose(weights_sum, np.ones_like(weights_sum), rtol=1e-6)
        
        # Check if output is sensitive to input V (a simple check)
        V_scaled = self.V * 0.1
        output_scaled_v = scaled_dot_product_attention(self.Q, self.K, V_scaled, self.d_k)
        self.assertTrue(np.allclose(output_scaled_v, output * 0.1))


if __name__ == '__main__':
    # Run tests only if the script is executed directly (not imported)
    unittest.main(argv=sys.argv[:1], exit=False)