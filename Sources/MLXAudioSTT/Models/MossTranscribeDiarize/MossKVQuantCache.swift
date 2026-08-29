//
//  MossKVQuantCache.swift
//  MLXAudioSwift
//
//  Created by Valerii Ivanov on 28.08.2026.
//  Copyright © 2026 TapMediaLtd. All rights reserved.
//

import MLX
import MLXFast
import MLXLMCommon

final class MossKVQuantCache: KVCache {
    static let keyBits = 8
    static let valueBits = 4
    static let groupSize = 64

    private let step = 256
    private var capacity = 0
    private var keys: (MLXArray, MLXArray, MLXArray?)?
    private var values: (MLXArray, MLXArray, MLXArray?)?
    private(set) var offset = 0
    var maxSize: Int? { nil }

    static func make() -> KVCacheSimple {
        KVCacheSimple()
    }

    static func converting(_ cache: KVCacheSimple) -> MossKVQuantCache {
        let converted = MossKVQuantCache()
        let state = cache.state
        if state.count == 2, cache.offset > 0 {
            _ = converted.append(
                keys: state[0][0..., 0..., ..<cache.offset, 0...],
                values: state[1][0..., 0..., ..<cache.offset, 0...]
            )
        }
        return converted
    }

    func innerState() -> [MLXArray] {
        state
    }

    var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            return [keys.0, keys.1, keys.2!, values.0, values.1, values.2!]
        }
        set {
            precondition(newValue.count == 6)
            keys = (newValue[0], newValue[1], newValue[2])
            values = (newValue[3], newValue[4], newValue[5])
            capacity = newValue[0].dim(2)
        }
    }

    var metaState: [String] {
        get { [String(offset), String(capacity)] }
        set {
            precondition(newValue.count == 2)
            offset = Int(newValue[0])!
            capacity = Int(newValue[1])!
        }
    }

    var isTrimmable: Bool { true }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    func copy() -> any KVCache {
        let copied = MossKVQuantCache()
        if !state.isEmpty {
            copied.state = state.map { $0[.ellipsis] }
        }
        copied.metaState = metaState
        return copied
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("MossKVQuantCache must be consumed through its quantized attention path")
    }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 {
            return .none
        }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    func attention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let state = append(keys: keys, values: values)
        let batch = queries.dim(0)
        let queryHeads = queries.dim(1)
        let queryLength = queries.dim(2)
        let dimension = queries.dim(3)
        let keyHeads = keys.dim(1)
        let repeats = queryHeads / keyHeads
        var scaledQueries = queries * scale
        var quantizedKeys = state.0
        var quantizedValues = state.1
        if repeats > 1 {
            scaledQueries = scaledQueries.reshaped(
                batch, keyHeads, repeats, queryLength, dimension
            )
            quantizedKeys = expanded(quantizedKeys)
            quantizedValues = expanded(quantizedValues)
        }
        var scores = quantizedMM(
            scaledQueries,
            quantizedKeys.0,
            scales: quantizedKeys.1,
            biases: quantizedKeys.2,
            transpose: true,
            groupSize: Self.groupSize,
            bits: Self.keyBits
        )
        scores = apply(mask: mask, to: scores)
        var output = quantizedMM(
            softmax(scores, axis: -1),
            quantizedValues.0,
            scales: quantizedValues.1,
            biases: quantizedValues.2,
            transpose: false,
            groupSize: Self.groupSize,
            bits: Self.valueBits
        )
        if repeats > 1 {
            output = output.reshaped(batch, queryHeads, queryLength, dimension)
        }
        return output
    }

    private func append(
        keys newKeys: MLXArray,
        values newValues: MLXArray
    ) -> (
        (MLXArray, MLXArray, MLXArray?),
        (MLXArray, MLXArray, MLXArray?)
    ) {
        let previous = offset
        let count = newKeys.dim(2)
        ensureCapacity(
            batch: newKeys.dim(0),
            heads: newKeys.dim(1),
            dimension: newKeys.dim(3),
            required: previous + count,
            dtype: newKeys.dtype
        )
        let quantizedKeys = quantized(
            newKeys,
            groupSize: Self.groupSize,
            bits: Self.keyBits
        )
        let quantizedValues = quantized(
            newValues,
            groupSize: Self.groupSize,
            bits: Self.valueBits
        )
        let range = previous..<(previous + count)
        assign(&keys!, quantizedKeys, range: range)
        assign(&values!, quantizedValues, range: range)
        offset += count
        return (trimmed(keys!), trimmed(values!))
    }

    private func ensureCapacity(
        batch: Int,
        heads: Int,
        dimension: Int,
        required: Int,
        dtype: DType
    ) {
        guard required > capacity else { return }
        let added = ((required - capacity + step - 1) / step) * step
        keys = grow(
            keys,
            batch: batch,
            heads: heads,
            count: added,
            dimension: dimension,
            dtype: dtype,
            bits: Self.keyBits
        )
        values = grow(
            values,
            batch: batch,
            heads: heads,
            count: added,
            dimension: dimension,
            dtype: dtype,
            bits: Self.valueBits
        )
        capacity += added
    }

    private func grow(
        _ current: (MLXArray, MLXArray, MLXArray?)?,
        batch: Int,
        heads: Int,
        count: Int,
        dimension: Int,
        dtype: DType,
        bits: Int
    ) -> (MLXArray, MLXArray, MLXArray?) {
        let addition = quantized(
            MLXArray.zeros([batch, heads, count, dimension], dtype: dtype),
            groupSize: Self.groupSize,
            bits: bits
        )
        guard let current else {
            return (addition.wq, addition.scales, addition.biases)
        }
        return (
            concatenated([current.0, addition.wq], axis: 2),
            concatenated([current.1, addition.scales], axis: 2),
            concatenated([current.2!, addition.biases!], axis: 2)
        )
    }

    private func assign(
        _ destination: inout (MLXArray, MLXArray, MLXArray?),
        _ source: (wq: MLXArray, scales: MLXArray, biases: MLXArray?),
        range: Range<Int>
    ) {
        destination.0[0..., 0..., range, 0...] = source.wq
        destination.1[0..., 0..., range, 0...] = source.scales
        destination.2![0..., 0..., range, 0...] = source.biases!
    }

    private func trimmed(
        _ value: (MLXArray, MLXArray, MLXArray?)
    ) -> (MLXArray, MLXArray, MLXArray?) {
        (
            value.0[0..., 0..., ..<offset, 0...],
            value.1[0..., 0..., ..<offset, 0...],
            value.2![0..., 0..., ..<offset, 0...]
        )
    }

    private func expanded(
        _ value: (MLXArray, MLXArray, MLXArray?)
    ) -> (MLXArray, MLXArray, MLXArray?) {
        (
            expandedDimensions(value.0, axis: -3),
            expandedDimensions(value.1, axis: -3),
            expandedDimensions(value.2!, axis: -3)
        )
    }

    private func apply(
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        to scores: MLXArray
    ) -> MLXArray {
        switch mask {
        case .causal:
            let queryIndices = MLXArray(0..<scores.dim(-2))
                + MLXArray(scores.dim(-1) - scores.dim(-2))
            let keyIndices = MLXArray(0..<scores.dim(-1))
            return MLX.where(
                queryIndices[0..., .newAxis] .>= keyIndices[.newAxis],
                scores,
                MLXArray(Float(-1e9)).asType(scores.dtype)
            )
        case .array(let array):
            return array.dtype == .bool
                ? MLX.where(array, scores, MLXArray(Float(-1e9)).asType(scores.dtype))
                : scores + array
        case .arrays(let arrays):
            guard let array = arrays.first else { return scores }
            return array.dtype == .bool
                ? MLX.where(array, scores, MLXArray(Float(-1e9)).asType(scores.dtype))
                : scores + array
        case .none:
            return scores
        }
    }
}
