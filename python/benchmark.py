import numpy as np
import time
import json
import sys

try:
    from apex_metal import ApexMetal
    metal = ApexMetal()
    has_metal = True
except ImportError:
    has_metal = False
    print("Warning: libApexMetal.dylib not found. Running numpy baseline only.", file=sys.stderr)

def bench_matmul():
    results = []
    sizes = [512, 1024, 2048]
    for s in sizes:
        A = np.random.randn(s, s).astype(np.float32)
        B = np.random.randn(s, s).astype(np.float32)
        
        t0 = time.time()
        _ = A @ B
        np_time = time.time() - t0
        
        metal_time = None
        if has_metal:
            # warmup
            _ = metal.matmul(A, B)
            t0 = time.time()
            _ = metal.matmul(A, B)
            metal_time = time.time() - t0
            
        flops = 2.0 * s * s * s
        gflops_np = (flops / np_time) / 1e9
        gflops_metal = (flops / metal_time) / 1e9 if metal_time else 0
        
        results.append({
            "size": f"{s}x{s}",
            "numpy_gflops": gflops_np,
            "metal_gflops": gflops_metal
        })
    return results

def bench_attention():
    results = []
    seq_lens = [128, 256, 512]
    dModel = 512
    numHeads = 8
    
    for s in seq_lens:
        Q = np.random.randn(numHeads, s, dModel).astype(np.float32)
        K = np.random.randn(numHeads, s, dModel).astype(np.float32)
        V = np.random.randn(numHeads, s, dModel).astype(np.float32)
        
        t0 = time.time()
        # mock numpy attention
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(dModel)
        np_time = time.time() - t0
        
        metal_time = None
        if has_metal:
            _ = metal.attention_forward(Q, K, V)
            t0 = time.time()
            _ = metal.attention_forward(Q, K, V)
            metal_time = time.time() - t0
            
        results.append({
            "seq_len": s,
            "dModel": dModel,
            "numpy_ms": np_time * 1000,
            "metal_ms": metal_time * 1000 if metal_time else 0
        })
    return results

if __name__ == "__main__":
    res = {
        "matmul": bench_matmul(),
        "attention": bench_attention()
    }
    print(json.dumps(res, indent=2))
