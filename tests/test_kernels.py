try:
    import numpy as np
    has_numpy = True
except ImportError:
    has_numpy = False
    np = None

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

from apex_metal import ApexMetal

try:
    metal = ApexMetal()
    has_metal = True
except (ImportError, Exception):
    metal = None
    has_metal = False

pytestmark = pytest.mark.skipif(not has_numpy, reason="numpy not available")

class TestAttentionCPU:
    def test_softmax_weights_when_v_is_one(self):
        rng = np.random.default_rng(7)
        q = rng.standard_normal((2, 8, 16)).astype(np.float32)
        k = rng.standard_normal((2, 8, 16)).astype(np.float32)
        v = np.ones((2, 8, 16), dtype=np.float32)
        out = ApexMetal.attention_forward_cpu(q, k, v)
        np.testing.assert_allclose(out, np.ones_like(out), rtol=1e-5, atol=1e-5)

    def test_rejects_shape_mismatch(self):
        q = np.zeros((1, 4, 8), dtype=np.float32)
        k = np.zeros((1, 4, 7), dtype=np.float32)
        v = np.zeros((1, 4, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="Q,K,V"):
            ApexMetal.attention_forward_cpu(q, k, v)

@pytest.mark.skipif(not has_metal, reason="Metal framework not available")
class TestMatMul:
    def test_correctness(self):
        A = np.random.randn(64, 64).astype(np.float32)
        B = np.random.randn(64, 64).astype(np.float32)
        
        out_np = A @ B
        out_metal = metal.matmul(A, B)
        
        np.testing.assert_allclose(out_np, out_metal, rtol=1e-3, atol=1e-3)

@pytest.mark.skipif(not has_metal, reason="Metal framework not available")
class TestAttention:
    def test_shape(self):
        numHeads, seqLen, dModel = 4, 128, 64
        Q = np.random.randn(numHeads, seqLen, dModel).astype(np.float32)
        K = np.random.randn(numHeads, seqLen, dModel).astype(np.float32)
        V = np.random.randn(numHeads, seqLen, dModel).astype(np.float32)
        
        out = metal.attention_forward(Q, K, V)
    def test_softmax_sums_to_one(self):
        numHeads, seqLen, dModel = 2, 64, 32
        Q = np.random.randn(numHeads, seqLen, dModel).astype(np.float32)
        K = np.random.randn(numHeads, seqLen, dModel).astype(np.float32)
        V = np.ones((numHeads, seqLen, dModel), dtype=np.float32)
        out = metal.attention_forward(Q, K, V)
        # When V is all 1.0, weighted attention output must equal 1.0 everywhere
        np.testing.assert_allclose(out, np.ones_like(out), rtol=1e-3, atol=1e-3)
