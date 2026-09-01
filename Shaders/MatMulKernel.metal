#include <metal_stdlib>
using namespace metal;

struct MatMulParams {
    uint M;
    uint N;
    uint K;
};

#define TILE_SIZE 16

kernel void tiledMatMul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatMulParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    threadgroup float A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float B_tile[TILE_SIZE][TILE_SIZE];
    
    float acc = 0.0;
    uint numTiles = (params.K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < numTiles; ++t) {
        uint k_idx = t * TILE_SIZE + tid.x;
        if (row < params.M && k_idx < params.K) {
            A_tile[tid.y][tid.x] = A[row * params.K + k_idx];
        } else {
            A_tile[tid.y][tid.x] = 0.0;
        }
        
        k_idx = t * TILE_SIZE + tid.y;
        if (k_idx < params.K && col < params.N) {
            B_tile[tid.y][tid.x] = B[k_idx * params.N + col];
        } else {
            B_tile[tid.y][tid.x] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint k = 0; k < TILE_SIZE; ++k) {
            acc += A_tile[tid.y][k] * B_tile[k][tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < params.M && col < params.N) {
        C[row * params.N + col] = acc;
    }
}
