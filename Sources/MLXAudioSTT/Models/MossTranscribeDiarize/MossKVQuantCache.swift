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

private final class MossKVQuantKernelManager: Sendable {
    static let shared = MossKVQuantKernelManager()

    let keyScores = MLXFast.metalKernel(
        name: "moss_kvquant_key_scores",
        inputNames: [
            "queries", "key_codes", "codebook", "key_center", "key_scale",
            "outlier_indices", "outlier_residuals", "attention_scale", "key_length", "capacity",
        ],
        outputNames: ["scores"],
        source: """
            uint elem = thread_position_in_grid.x;
            uint actual_key_length = uint(key_length);
            uint actual_capacity = uint(capacity);
            uint key_position = elem % actual_key_length;
            uint q = (elem / actual_key_length) % QUERY_LENGTH;
            uint query_head = (elem / (actual_key_length * QUERY_LENGTH)) % QUERY_HEADS;
            uint batch = elem / (actual_key_length * QUERY_LENGTH * QUERY_HEADS);
            uint kv_head = query_head / HEAD_REPEATS;
            uint query_base = ((batch * QUERY_HEADS + query_head) * QUERY_LENGTH + q) * HEAD_DIM;
            uint key_base = ((batch * KV_HEADS + kv_head) * actual_capacity + key_position) * PACKED_DIM;
            float sum = 0.0f;
            for (uint pair = 0; pair < HALF_DIM; ++pair) {
                uint first = pair;
                uint second = pair + HALF_DIM;
                uint first_byte = key_codes[key_base + first / 2];
                uint second_byte = key_codes[key_base + second / 2];
                uint first_code = (first & 1) == 0 ? first_byte & 15 : first_byte >> 4;
                uint second_code = (second & 1) == 0 ? second_byte & 15 : second_byte >> 4;
                uint first_channel = kv_head * HEAD_DIM + first;
                uint second_channel = kv_head * HEAD_DIM + second;
                float first_value = float(codebook[first_code]) * float(key_scale[first_channel]) + float(key_center[first_channel]);
                float second_value = float(codebook[second_code]) * float(key_scale[second_channel]) + float(key_center[second_channel]);
                float frequency = metal::fast::exp2(-float(pair) * 19.9315685693f / float(HALF_DIM));
                float angle = float(key_position) * frequency;
                float cosine = metal::fast::cos(angle);
                float sine = metal::fast::sin(angle);
                float rotated_first = first_value * cosine - second_value * sine;
                float rotated_second = first_value * sine + second_value * cosine;
                sum += float(queries[query_base + first]) * rotated_first;
                sum += float(queries[query_base + second]) * rotated_second;
            }
            uint sparse_base = ((batch * KV_HEADS + kv_head) * actual_capacity + key_position) * OUTLIERS;
            for (uint sparse = 0; sparse < OUTLIERS; ++sparse) {
                uint dimension = outlier_indices[sparse_base + sparse];
                float residual = float(outlier_residuals[sparse_base + sparse]);
                uint pair = dimension < HALF_DIM ? dimension : dimension - HALF_DIM;
                float frequency = metal::fast::exp2(-float(pair) * 19.9315685693f / float(HALF_DIM));
                float angle = float(key_position) * frequency;
                float cosine = metal::fast::cos(angle);
                float sine = metal::fast::sin(angle);
                if (dimension < HALF_DIM) {
                    sum += residual * (float(queries[query_base + pair]) * cosine + float(queries[query_base + pair + HALF_DIM]) * sine);
                } else {
                    sum += residual * (-float(queries[query_base + pair]) * sine + float(queries[query_base + pair + HALF_DIM]) * cosine);
                }
            }
            scores[elem] = static_cast<T>(sum * float(attention_scale));
            """
    )

    let valueOutput = MLXFast.metalKernel(
        name: "moss_kvquant_value_output",
        inputNames: [
            "weights", "value_codes", "codebook", "value_center", "value_scale",
            "outlier_indices", "outlier_residuals", "key_length", "capacity",
        ],
        outputNames: ["output"],
        source: """
            uint elem = thread_position_in_grid.x;
            uint actual_key_length = uint(key_length);
            uint actual_capacity = uint(capacity);
            uint dimension = elem % HEAD_DIM;
            uint q = (elem / HEAD_DIM) % QUERY_LENGTH;
            uint query_head = (elem / (HEAD_DIM * QUERY_LENGTH)) % QUERY_HEADS;
            uint batch = elem / (HEAD_DIM * QUERY_LENGTH * QUERY_HEADS);
            uint kv_head = query_head / HEAD_REPEATS;
            uint weight_base = ((batch * QUERY_HEADS + query_head) * QUERY_LENGTH + q) * actual_key_length;
            float sum = 0.0f;
            for (uint key_position = 0; key_position < actual_key_length; ++key_position) {
                uint value_base = ((batch * KV_HEADS + kv_head) * actual_capacity + key_position) * PACKED_DIM;
                uint code_byte = value_codes[value_base + dimension / 2];
                uint code = (dimension & 1) == 0 ? code_byte & 15 : code_byte >> 4;
                uint parameter_index = (batch * KV_HEADS + kv_head) * actual_capacity + key_position;
                float value = float(codebook[code]) * float(value_scale[parameter_index]) + float(value_center[parameter_index]);
                uint sparse_base = parameter_index * OUTLIERS;
                for (uint sparse = 0; sparse < OUTLIERS; ++sparse) {
                    if (outlier_indices[sparse_base + sparse] == dimension) {
                        value += float(outlier_residuals[sparse_base + sparse]);
                    }
                }
                sum += float(weights[weight_base + key_position]) * value;
            }
            output[elem] = static_cast<T>(sum);
            """
    )
}

final class MossKVQuantCache: KVCache {
    private let step: Int
    let outlierCount: Int
    private(set) var offset = 0
    var maxSize: Int? { nil }
    private var capacity = 0
    private var keyCodes: MLXArray?
    private var valueCodes: MLXArray?
    private var keyCodebook: MLXArray?
    private var keyCenter: MLXArray?
    private var keyScale: MLXArray?
    private var valueCodebook: MLXArray?
    private var valueCenter: MLXArray?
    private var valueScale: MLXArray?
    private var keyOutlierIndices: MLXArray?
    private var keyOutlierResiduals: MLXArray?
    private var valueOutlierIndices: MLXArray?
    private var valueOutlierResiduals: MLXArray?

    init(step: Int = 256, outlierCount: Int = 4) {
        precondition(step > 0)
        precondition(outlierCount > 0 && outlierCount <= 128)
        self.step = step
        self.outlierCount = outlierCount
    }

    var persistentBytes: Int {
        innerState().reduce(0) { $0 + $1.nbytes }
    }

    var packedKeyShape: [Int]? { keyCodes?.shape }
    var packedValueShape: [Int]? { valueCodes?.shape }
    var calibratedKeyCodebook: MLXArray? { keyCodebook }
    var calibratedValueCodebook: MLXArray? { valueCodebook }
    var sparseKeyIndexType: DType? { keyOutlierIndices?.dtype }
    var sparseValueIndexType: DType? { valueOutlierIndices?.dtype }

    func innerState() -> [MLXArray] {
        [
            keyCodes, valueCodes, keyCodebook, keyCenter, keyScale, valueCodebook,
            valueCenter, valueScale, keyOutlierIndices, keyOutlierResiduals,
            valueOutlierIndices, valueOutlierResiduals,
        ].compactMap { $0 }
    }

    var state: [MLXArray] {
        get { innerState() }
        set {
            precondition(newValue.count == 12)
            keyCodes = newValue[0]
            valueCodes = newValue[1]
            keyCodebook = newValue[2]
            keyCenter = newValue[3]
            keyScale = newValue[4]
            valueCodebook = newValue[5]
            valueCenter = newValue[6]
            valueScale = newValue[7]
            keyOutlierIndices = newValue[8]
            keyOutlierResiduals = newValue[9]
            valueOutlierIndices = newValue[10]
            valueOutlierResiduals = newValue[11]
            capacity = keyCodes!.dim(2)
        }
    }

    var metaState: [String] {
        get { [String(step), String(outlierCount), String(offset), String(capacity)] }
        set {
            precondition(newValue.count == 4)
            guard let restoredOffset = Int(newValue[2]), let restoredCapacity = Int(newValue[3]) else {
                preconditionFailure()
            }
            offset = restoredOffset
            capacity = restoredCapacity
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
        let copied = MossKVQuantCache(step: step, outlierCount: outlierCount)
        if !state.isEmpty {
            copied.state = state.map { $0[.ellipsis] }
        }
        copied.metaState = metaState
        return copied
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("MossKVQuantCache must be consumed through its Metal attention path")
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
        append(keys: keys, values: values)
        let batch = queries.dim(0)
        let queryHeads = queries.dim(1)
        let queryLength = queries.dim(2)
        let headDimension = queries.dim(3)
        let kvHeads = keys.dim(1)
        let repeats = queryHeads / kvHeads
        let scoreCount = batch * queryHeads * queryLength * offset
        let keyLength = MLXArray(Int32(offset))
        let cacheCapacity = MLXArray(Int32(capacity))
        var scores = MossKVQuantKernelManager.shared.keyScores(
            [
                queries, keyCodes!, keyCodebook!, keyCenter!, keyScale!,
                keyOutlierIndices!, keyOutlierResiduals!, MLXArray(scale), keyLength, cacheCapacity,
            ],
            template: [
                ("T", queries.dtype),
                ("QUERY_LENGTH", queryLength),
                ("QUERY_HEADS", queryHeads),
                ("KV_HEADS", kvHeads),
                ("HEAD_REPEATS", repeats),
                ("HEAD_DIM", headDimension),
                ("HALF_DIM", headDimension / 2),
                ("PACKED_DIM", headDimension / 2),
                ("OUTLIERS", outlierCount),
            ],
            grid: (scoreCount, 1, 1),
            threadGroup: (min(256, scoreCount), 1, 1),
            outputShapes: [[batch, queryHeads, queryLength, offset]],
            outputDTypes: [queries.dtype]
        )[0]
        scores = apply(mask: mask, to: scores)
        let weights = softmax(scores, axis: -1)
        let outputCount = batch * queryHeads * queryLength * headDimension
        return MossKVQuantKernelManager.shared.valueOutput(
            [
                weights, valueCodes!, valueCodebook!, valueCenter!, valueScale!,
                valueOutlierIndices!, valueOutlierResiduals!, keyLength, cacheCapacity,
            ],
            template: [
                ("T", queries.dtype),
                ("QUERY_LENGTH", queryLength),
                ("QUERY_HEADS", queryHeads),
                ("KV_HEADS", kvHeads),
                ("HEAD_REPEATS", repeats),
                ("HEAD_DIM", headDimension),
                ("PACKED_DIM", headDimension / 2),
                ("OUTLIERS", outlierCount),
            ],
            grid: (outputCount, 1, 1),
            threadGroup: (min(256, outputCount), 1, 1),
            outputShapes: [[batch, queryHeads, queryLength, headDimension]],
            outputDTypes: [queries.dtype]
        )[0]
    }

    private func append(keys: MLXArray, values: MLXArray) {
        precondition(keys.shape == values.shape)
        precondition(keys.dim(3).isMultiple(of: 2))
        let batch = keys.dim(0)
        let heads = keys.dim(1)
        let count = keys.dim(2)
        let dimension = keys.dim(3)
        if keyCodebook == nil {
            calibrate(keys: keys, values: values)
        }
        ensureCapacity(batch: batch, heads: heads, dimension: dimension, required: offset + count, dtype: keys.dtype)
        let keyPacked = quantizeKeys(keys)
        let valuePacked = quantizeValues(values)
        let range = offset ..< offset + count
        keyCodes![0..., 0..., range, 0...] = keyPacked.codes
        valueCodes![0..., 0..., range, 0...] = valuePacked.codes
        keyOutlierIndices![0..., 0..., range, 0...] = keyPacked.indices
        keyOutlierResiduals![0..., 0..., range, 0...] = keyPacked.residuals
        valueOutlierIndices![0..., 0..., range, 0...] = valuePacked.indices
        valueOutlierResiduals![0..., 0..., range, 0...] = valuePacked.residuals
        valueCenter![0..., 0..., range] = valuePacked.center
        valueScale![0..., 0..., range] = valuePacked.scale
        offset += count
    }

    private func calibrate(keys: MLXArray, values: MLXArray) {
        let count = keys.dim(0) * keys.dim(2)
        let orderedKeys = keys.transposed(1, 3, 0, 2).reshaped(keys.dim(1), keys.dim(3), count)
        let sortedKeys = MLX.sorted(orderedKeys, axis: -1)
        let tail = min(max(0, count / 200), max(0, (count - 1) / 2))
        let lower = sortedKeys[0..., 0..., tail]
        let upper = sortedKeys[0..., 0..., count - tail - 1]
        keyCenter = ((lower + upper) / 2).asType(keys.dtype)
        keyScale = MLX.maximum(
            (upper - lower) / 2,
            MLXArray(Float(1e-6)).asType(keys.dtype)
        )
        let keyNormalized = MLX.clip(
            (keys - keyCenter![.newAxis, 0..., .newAxis, 0...])
                / keyScale![.newAxis, 0..., .newAxis, 0...],
            min: -1,
            max: 1
        )
        keyCodebook = calibratedCodebook(keyNormalized).asType(keys.dtype)
        let valueParameters = perTokenParameters(values)
        let valueNormalized = MLX.clip(
            (values - valueParameters.center[0..., 0..., 0..., .newAxis])
                / valueParameters.scale[0..., 0..., 0..., .newAxis],
            min: -1,
            max: 1
        )
        valueCodebook = calibratedCodebook(valueNormalized).asType(values.dtype)
    }

    private func calibratedCodebook(_ values: MLXArray) -> MLXArray {
        let ordered = MLX.sorted(values.reshaped(-1), axis: 0)
        let last = ordered.size - 1
        let indices = (0..<16).map { Int32(($0 * last) / 15) }
        return MLX.take(ordered, MLXArray(indices))
    }

    private func perTokenParameters(_ values: MLXArray) -> (center: MLXArray, scale: MLXArray) {
        let ordered = MLX.sorted(values, axis: -1)
        let dimension = values.dim(-1)
        let tail = min(max(1, outlierCount / 2), max(1, (dimension - 1) / 2))
        let lower = ordered[.ellipsis, tail]
        let upper = ordered[.ellipsis, dimension - tail - 1]
        let center = ((lower + upper) / 2).asType(values.dtype)
        let scale = MLX.maximum(
            (upper - lower) / 2,
            MLXArray(Float(1e-6)).asType(values.dtype)
        )
        return (center, scale)
    }

    private func quantizeKeys(_ keys: MLXArray) -> (codes: MLXArray, indices: MLXArray, residuals: MLXArray) {
        let center = keyCenter![.newAxis, 0..., .newAxis, 0...]
        let scale = keyScale![.newAxis, 0..., .newAxis, 0...]
        let normalized = MLX.clip((keys - center) / scale, min: -1, max: 1)
        let codes = MLX.abs(normalized[.ellipsis, .newAxis] - keyCodebook!).argMin(axis: -1)
        let reconstruction = MLX.take(keyCodebook!, codes) * scale + center
        let sparse = sparseResiduals(original: keys, reconstruction: reconstruction)
        return (pack(codes), sparse.indices, sparse.residuals)
    }

    private func quantizeValues(_ values: MLXArray) -> (
        codes: MLXArray, indices: MLXArray, residuals: MLXArray,
        center: MLXArray, scale: MLXArray
    ) {
        let parameters = perTokenParameters(values)
        let center = parameters.center[0..., 0..., 0..., .newAxis]
        let scale = parameters.scale[0..., 0..., 0..., .newAxis]
        let normalized = MLX.clip((values - center) / scale, min: -1, max: 1)
        let codes = MLX.abs(normalized[.ellipsis, .newAxis] - valueCodebook!).argMin(axis: -1)
        let reconstruction = MLX.take(valueCodebook!, codes) * scale + center
        let sparse = sparseResiduals(original: values, reconstruction: reconstruction)
        return (pack(codes), sparse.indices, sparse.residuals, parameters.center, parameters.scale)
    }

    private func sparseResiduals(original: MLXArray, reconstruction: MLXArray) -> (
        indices: MLXArray, residuals: MLXArray
    ) {
        let residual = original - reconstruction
        let dimension = original.dim(-1)
        let partition = argPartition(MLX.abs(residual), kth: dimension - outlierCount, axis: -1)
        let indices = partition[.ellipsis, (dimension - outlierCount)...]
        return (
            indices.asType(.uint8),
            MLX.takeAlong(residual, indices, axis: -1).asType(original.dtype)
        )
    }

    private func pack(_ codes: MLXArray) -> MLXArray {
        let bytes = codes.asType(.uint8)
        let lower = bytes[.ellipsis, .stride(by: 2)]
        let upper = bytes[.ellipsis, .stride(from: 1, by: 2)] << 4
        return lower | upper
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
        let newCapacity = capacity + added
        func expanded(_ current: MLXArray?, tail: [Int], dtype: DType) -> MLXArray {
            let addition = MLXArray.zeros([batch, heads, added] + tail, dtype: dtype)
            guard let current else { return addition }
            return concatenated([current, addition], axis: 2)
        }
        keyCodes = expanded(keyCodes, tail: [dimension / 2], dtype: .uint8)
        valueCodes = expanded(valueCodes, tail: [dimension / 2], dtype: .uint8)
        keyOutlierIndices = expanded(keyOutlierIndices, tail: [outlierCount], dtype: .uint8)
        valueOutlierIndices = expanded(valueOutlierIndices, tail: [outlierCount], dtype: .uint8)
        keyOutlierResiduals = expanded(keyOutlierResiduals, tail: [outlierCount], dtype: dtype)
        valueOutlierResiduals = expanded(valueOutlierResiduals, tail: [outlierCount], dtype: dtype)
        valueCenter = expanded(valueCenter, tail: [], dtype: dtype)
        valueScale = expanded(valueScale, tail: [], dtype: dtype)
        capacity = newCapacity
    }

    private func apply(
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        to scores: MLXArray
    ) -> MLXArray {
        switch mask {
        case .causal:
            let queryLength = scores.dim(-2)
            let keyLength = scores.dim(-1)
            let queryIndices = MLXArray(0..<queryLength) + MLXArray(keyLength - queryLength)
            let keyIndices = MLXArray(0..<keyLength)
            let causal = queryIndices[0..., .newAxis] .>= keyIndices[.newAxis]
            return MLX.where(causal, scores, MLXArray(Float(-1e9)).asType(scores.dtype))
        case .array(let array):
            if array.dtype == .bool {
                return MLX.where(array, scores, MLXArray(Float(-1e9)).asType(scores.dtype))
            }
            return scores + array
        case .arrays(let arrays):
            guard let array = arrays.first else { return scores }
            if array.dtype == .bool {
                return MLX.where(array, scores, MLXArray(Float(-1e9)).asType(scores.dtype))
            }
            return scores + array
        case .none:
            return scores
        }
    }
}
