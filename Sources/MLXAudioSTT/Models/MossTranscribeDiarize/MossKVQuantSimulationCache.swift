//
//  MossKVQuantSimulationCache.swift
//  MLXAudioSwift
//
//  Created by Valerii Ivanov on 28.08.2026.
//  Copyright © 2026 TapMediaLtd. All rights reserved.
//

import MLX
import MLXFast
import MLXLMCommon

final class MossKVQuantSimulationCache: KVCache {
    private let storage: KVCacheSimple
    private var keyMinimum: MLXArray?
    private var keyMaximum: MLXArray?

    init(step: Int = 256) {
        storage = KVCacheSimple()
        storage.step = step
    }

    var offset: Int { storage.offset }
    var maxSize: Int? { storage.maxSize }
    var state: [MLXArray] {
        get { storage.state }
        set { storage.state = newValue }
    }
    var metaState: [String] {
        get { storage.metaState }
        set { storage.metaState = newValue }
    }
    var isTrimmable: Bool { storage.isTrimmable }

    func innerState() -> [MLXArray] {
        storage.innerState() + [keyMinimum, keyMaximum].compactMap { $0 }
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        storage.update(keys: keys, values: values)
    }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        storage.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        storage.trim(n)
    }

    func copy() -> any KVCache {
        let copied = MossKVQuantSimulationCache(step: storage.step)
        copied.state = state.map { $0[.ellipsis] }
        copied.metaState = metaState
        copied.keyMinimum = keyMinimum.map { $0[.ellipsis] }
        copied.keyMaximum = keyMaximum.map { $0[.ellipsis] }
        return copied
    }

    func roundTrip(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let currentMinimum = keys.min(axis: 2, keepDims: true)
        let currentMaximum = keys.max(axis: 2, keepDims: true)
        if keys.dim(2) > 1 || keyMinimum == nil || keyMaximum == nil {
            keyMinimum = keyMinimum.map { minimum($0, currentMinimum) } ?? currentMinimum
            keyMaximum = keyMaximum.map { maximum($0, currentMaximum) } ?? currentMaximum
        }
        let quantizedKeys = affineRoundTrip(
            keys,
            minimum: keyMinimum!,
            maximum: keyMaximum!,
            preserveOutliers: true
        )
        let quantizedValues = affineRoundTrip(
            values,
            minimum: values.min(axis: -1, keepDims: true),
            maximum: values.max(axis: -1, keepDims: true),
            preserveOutliers: false
        )
        return (quantizedKeys, quantizedValues)
    }

    private func affineRoundTrip(
        _ values: MLXArray,
        minimum lower: MLXArray,
        maximum upper: MLXArray,
        preserveOutliers: Bool
    ) -> MLXArray {
        let scale = MLX.maximum(
            (upper - lower) / 15,
            MLXArray(Float(1e-6)).asType(values.dtype)
        )
        let codes = MLX.clip(MLX.round((values - lower) / scale), min: 0, max: 15)
        let reconstructed = codes * scale + lower
        guard preserveOutliers else { return reconstructed }
        return MLX.where((values .< lower) | (values .> upper), values, reconstructed)
    }
}
