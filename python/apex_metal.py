import ctypes
import os
import sys
import numpy as np
from pathlib import Path

class ApexMetal:
    def __init__(self):
        # Look for the dylib
        possible_paths = [
            Path(__file__).parent.parent / ".build" / "debug" / "libApexMetal.dylib",
            Path(__file__).parent.parent / ".build" / "release" / "libApexMetal.dylib",
        ]
        
        self.lib = None
        for p in possible_paths:
            if p.exists():
                self.lib = ctypes.CDLL(str(p))
                break
                
        if self.lib is None:
            raise ImportError("libApexMetal.dylib not found. Run 'swift build' first.")
            
        self.lib.apex_metal_init.restype = ctypes.c_void_p
        self.lib.apex_metal_attention.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        self.lib.apex_metal_attention.restype = ctypes.c_int32
        
        self.lib.apex_metal_matmul.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        self.lib.apex_metal_matmul.restype = ctypes.c_int32
        
        self.lib.apex_metal_free.argtypes = [ctypes.c_void_p]
        
        self.ctx = self.lib.apex_metal_init()
        if not self.ctx:
            raise RuntimeError("Failed to initialize Metal pipeline")
            
    def __del__(self):
        if hasattr(self, 'lib') and self.lib and hasattr(self, 'ctx') and self.ctx:
            self.lib.apex_metal_free(self.ctx)
            
    def attention_forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        K = np.ascontiguousarray(K, dtype=np.float32)
        V = np.ascontiguousarray(V, dtype=np.float32)
        out = np.zeros_like(Q)
        
        numHeads, seqLen, dModel = Q.shape
        
        q_ptr = Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        k_ptr = K.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        v_ptr = V.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        res = self.lib.apex_metal_attention(self.ctx, q_ptr, k_ptr, v_ptr, out_ptr, seqLen, dModel, numHeads)
        if res != 0:
            raise RuntimeError("Attention kernel failed")
        return out
        
    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        M, K_dim = A.shape
        K_dim2, N = B.shape
        assert K_dim == K_dim2
        
        out = np.zeros((M, N), dtype=np.float32)
        
        a_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        res = self.lib.apex_metal_matmul(self.ctx, a_ptr, b_ptr, out_ptr, M, N, K_dim)
        if res != 0:
            raise RuntimeError("Matmul kernel failed")
        return out
