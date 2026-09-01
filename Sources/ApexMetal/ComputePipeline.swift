import Metal
import Foundation

public enum MetalError: Error {
    case deviceNotFound
    case pipelineCompileFailed
    case bufferAllocationFailed
}

public class ApexMetalPipeline {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let attentionPipelineState: MTLComputePipelineState
    let matmulPipelineState: MTLComputePipelineState
    
    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalError.deviceNotFound
        }
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw MetalError.deviceNotFound
        }
        self.commandQueue = queue
        
        guard let defaultLibrary = try? device.makeDefaultLibrary(bundle: Bundle.module) else {
            throw MetalError.pipelineCompileFailed
        }
        
        guard let attentionFunction = defaultLibrary.makeFunction(name: "scaledDotProductAttention"),
              let matmulFunction = defaultLibrary.makeFunction(name: "tiledMatMul") else {
            throw MetalError.pipelineCompileFailed
        }
        
        do {
            attentionPipelineState = try device.makeComputePipelineState(function: attentionFunction)
            matmulPipelineState = try device.makeComputePipelineState(function: matmulFunction)
        } catch {
            throw MetalError.pipelineCompileFailed
        }
    }
    
    public func attentionForward(Q: [Float], K: [Float], V: [Float], seqLen: Int, dModel: Int, numHeads: Int) throws -> [Float] {
        let size = Q.count * MemoryLayout<Float>.stride
        guard let bufQ = device.makeBuffer(bytes: Q, length: size, options: .storageModeShared),
              let bufK = device.makeBuffer(bytes: K, length: size, options: .storageModeShared),
              let bufV = device.makeBuffer(bytes: V, length: size, options: .storageModeShared),
              let bufOut = device.makeBuffer(length: size, options: .storageModeShared) else {
            throw MetalError.bufferAllocationFailed
        }
        
        var params = [UInt32(seqLen), UInt32(dModel), UInt32(numHeads), Float32(1.0 / sqrt(Float(dModel)))]
        guard let bufParams = device.makeBuffer(bytes: &params, length: MemoryLayout<UInt32>.stride * 3 + MemoryLayout<Float32>.stride, options: .storageModeShared) else {
            throw MetalError.bufferAllocationFailed
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.deviceNotFound
        }
        
        encoder.setComputePipelineState(attentionPipelineState)
        encoder.setBuffer(bufQ, offset: 0, index: 0)
        encoder.setBuffer(bufK, offset: 0, index: 1)
        encoder.setBuffer(bufV, offset: 0, index: 2)
        encoder.setBuffer(bufOut, offset: 0, index: 3)
        encoder.setBuffer(bufParams, offset: 0, index: 4)
        
        let w = attentionPipelineState.threadExecutionWidth
        let h = attentionPipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)
        let threadsPerGrid = MTLSize(width: dModel, height: seqLen, depth: numHeads)
        
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let result = bufOut.contents().bindMemory(to: Float.self, capacity: Q.count)
        return Array(UnsafeBufferPointer(start: result, count: Q.count))
    }
    
    public func matmul(A: [Float], B: [Float], M: Int, N: Int, K: Int) throws -> [Float] {
        let sizeA = M * K * MemoryLayout<Float>.stride
        let sizeB = K * N * MemoryLayout<Float>.stride
        let sizeC = M * N * MemoryLayout<Float>.stride
        
        guard let bufA = device.makeBuffer(bytes: A, length: sizeA, options: .storageModeShared),
              let bufB = device.makeBuffer(bytes: B, length: sizeB, options: .storageModeShared),
              let bufC = device.makeBuffer(length: sizeC, options: .storageModeShared) else {
            throw MetalError.bufferAllocationFailed
        }
        
        var params = [UInt32(M), UInt32(N), UInt32(K)]
        guard let bufParams = device.makeBuffer(bytes: &params, length: MemoryLayout<UInt32>.stride * 3, options: .storageModeShared) else {
            throw MetalError.bufferAllocationFailed
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.deviceNotFound
        }
        
        encoder.setComputePipelineState(matmulPipelineState)
        encoder.setBuffer(bufA, offset: 0, index: 0)
        encoder.setBuffer(bufB, offset: 0, index: 1)
        encoder.setBuffer(bufC, offset: 0, index: 2)
        encoder.setBuffer(bufParams, offset: 0, index: 3)
        
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadsPerGrid = MTLSize(width: N, height: M, depth: 1)
        
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let result = bufC.contents().bindMemory(to: Float.self, capacity: M * N)
        return Array(UnsafeBufferPointer(start: result, count: M * N))
    }
}
