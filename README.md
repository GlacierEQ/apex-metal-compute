# APEX Metal Compute

Hardware accelerated SWARM kernels using Apple Metal.

## Architecture
```
[ Python / NumPy ]
       |
  (ctypes binding)
       |
[ Swift / Metal ] -> Shaders (Tiled MatMul, Attn)
```

## Benchmarks
(Placeholder)
