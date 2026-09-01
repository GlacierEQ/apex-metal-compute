import numpy as np
import pytest

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from apex_metal import ApexMetal
    metal = ApexMetal()
    has_metal = True
except ImportError:
    has_metal = False

pytestmark = pytest.mark.skipif(not has_metal, reason="Metal library not found")

class TestMatMul:
    def test_correctness(self):
        A = np.random.randn(64, 64).astype(np.float32)
        B = np.random.randn(64, 64).astype(np.float32)
        
        out_np = A @ B
        out_metal = metal.matmul(A, B)
        
        np.testing.assert_allclose(out_np, out_metal, rtol=1e-3, atol=1e-3)

class TestAttention:
    def test_shape(self):
        numHeads, seqLen, dModel = 4, 128, 64
        Q = np.random.randn(numHeads, seqLen, dModel).astype(np.float32)
        K = np.random.randn(numHeads, seqLen, dModel).astype(np.float32)
        V = np.random.randn(numHeads, seqLen, dModel).astype(np.float32)
        
        out = metal.attention_forward(Q, K, V)
        assert out.shape == (numHeads, seqLen, dModel)
        
    def test_softmax_sums_to_one(self):
        pass
