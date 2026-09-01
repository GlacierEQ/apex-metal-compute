import Foundation
import ApexMetal

public class PipelineWrapper {
    public let pipeline: ApexMetalPipeline
    init() throws {
        self.pipeline = try ApexMetalPipeline()
    }
}

@_cdecl("apex_metal_init")
public func apexMetalInit() -> OpaquePointer? {
    do {
        let wrapper = try PipelineWrapper()
        return Unmanaged.passRetained(wrapper).toOpaque()
    } catch {
        return nil
    }
}

@_cdecl("apex_metal_attention")
public func apexMetalAttention(_ ctx: OpaquePointer, _ q: UnsafePointer<Float>, _ k: UnsafePointer<Float>, _ v: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>, _ seqLen: Int32, _ dModel: Int32, _ numHeads: Int32) -> Int32 {
    let wrapper = Unmanaged<PipelineWrapper>.fromOpaque(ctx).takeUnretainedValue()
    let count = Int(seqLen * dModel * numHeads)
    let arrQ = Array(UnsafeBufferPointer(start: q, count: count))
    let arrK = Array(UnsafeBufferPointer(start: k, count: count))
    let arrV = Array(UnsafeBufferPointer(start: v, count: count))
    
    do {
        let res = try wrapper.pipeline.attentionForward(Q: arrQ, K: arrK, V: arrV, seqLen: Int(seqLen), dModel: Int(dModel), numHeads: Int(numHeads))
        for i in 0..<count {
            out[i] = res[i]
        }
        return 0
    } catch {
        return -1
    }
}

@_cdecl("apex_metal_matmul")
public func apexMetalMatMul(_ ctx: OpaquePointer, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>, _ m: Int32, _ n: Int32, _ k: Int32) -> Int32 {
    let wrapper = Unmanaged<PipelineWrapper>.fromOpaque(ctx).takeUnretainedValue()
    let arrA = Array(UnsafeBufferPointer(start: a, count: Int(m * k)))
    let arrB = Array(UnsafeBufferPointer(start: b, count: Int(k * n)))
    
    do {
        let res = try wrapper.pipeline.matmul(A: arrA, B: arrB, M: Int(m), N: Int(n), K: Int(k))
        for i in 0..<Int(m * n) {
            out[i] = res[i]
        }
        return 0
    } catch {
        return -1
    }
}

@_cdecl("apex_metal_free")
public func apexMetalFree(_ ctx: OpaquePointer) {
    Unmanaged<PipelineWrapper>.fromOpaque(ctx).release()
}
