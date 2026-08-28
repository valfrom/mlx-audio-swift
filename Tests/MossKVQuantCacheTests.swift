//
//  MossKVQuantCacheTests.swift
//  MLXAudioSwift
//
//  Created by Valerii Ivanov on 28.08.2026.
//  Copyright © 2026 TapMediaLtd. All rights reserved.
//

import Metal
import MLX
import MLXFast
import MLXNN
import Testing

@testable import MLXAudioSTT

@Suite struct MossKVQuantCacheTests {
    @Test func packedStorageAndCalibration() {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        MLXRandom.seed(7)
        let length = 128
        let dimension = 128
        let queries = MLXRandom.normal([1, 4, length, dimension]).asType(.float16)
        let keys = MLXRandom.normal([1, 2, length, dimension]).asType(.float16)
        let values = MLXRandom.normal([1, 2, length, dimension]).asType(.float16)
        let cache = MossKVQuantCache(step: length, outlierCount: 4)
        let output = cache.attention(
            queries: RoPE(dimensions: dimension, traditional: false, base: 1_000_000)(queries),
            keys: keys,
            values: values,
            scale: pow(Float(dimension), -0.5),
            mask: .causal
        )
        eval(output)
        #expect(cache.offset == length)
        #expect(cache.packedKeyShape == [1, 2, length, dimension / 2])
        #expect(cache.packedValueShape == [1, 2, length, dimension / 2])
        #expect(cache.sparseKeyIndexType == .uint8)
        #expect(cache.sparseValueIndexType == .uint8)
        #expect(cache.persistentBytes < 2 * 1 * 2 * length * dimension * 2)
        #expect(!cache.state.contains { $0.shape == [1, 2, length, dimension] })
        let keyCodebook = cache.calibratedKeyCodebook!.asArray(Float.self)
        let valueCodebook = cache.calibratedValueCodebook!.asArray(Float.self)
        #expect(keyCodebook.count == 16)
        #expect(valueCodebook.count == 16)
        let keyGaps = zip(keyCodebook.dropFirst(), keyCodebook).map { $0.0 - $0.1 }
        let valueGaps = zip(valueCodebook.dropFirst(), valueCodebook).map { $0.0 - $0.1 }
        #expect((keyGaps.max()! - keyGaps.min()!) > 1e-3)
        #expect((valueGaps.max()! - valueGaps.min()!) > 1e-3)
    }

    @Test func metalAttentionConsumesPackedCache() {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        MLXRandom.seed(11)
        let length = 12
        let dimension = 128
        let scale = pow(Float(dimension), -0.5)
        let rawQueries = MLXRandom.normal([1, 4, length, dimension]).asType(.float16)
        let rawKeys = MLXRandom.normal([1, 2, length, dimension]).asType(.float16)
        let values = MLXRandom.normal([1, 2, length, dimension]).asType(.float16)
        let rope = RoPE(dimensions: dimension, traditional: false, base: 1_000_000)
        let queries = rope(rawQueries)
        let keys = rope(rawKeys)
        let repeatedKeys = MLX.repeated(keys, count: 2, axis: 1)
        let repeatedValues = MLX.repeated(values, count: 2, axis: 1)
        let expected = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: repeatedKeys,
            values: repeatedValues,
            scale: scale,
            mask: .causal
        )
        let cache = MossKVQuantCache(step: length, outlierCount: dimension)
        let actual = cache.attention(
            queries: queries,
            keys: rawKeys,
            values: values,
            scale: scale,
            mask: .causal
        )
        eval(expected, actual)
        #expect(MLX.abs(expected - actual).max().item(Float.self) < 0.03)
    }

    @Test func incrementalDecodeMatchesFullAttention() {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        MLXRandom.seed(19)
        let firstLength = 7
        let secondLength = 5
        let dimension = 128
        let scale = pow(Float(dimension), -0.5)
        let firstQueries = MLXRandom.normal([1, 4, firstLength, dimension]).asType(.float16)
        let secondQueries = MLXRandom.normal([1, 4, secondLength, dimension]).asType(.float16)
        let firstKeys = MLXRandom.normal([1, 2, firstLength, dimension]).asType(.float16)
        let secondKeys = MLXRandom.normal([1, 2, secondLength, dimension]).asType(.float16)
        let firstValues = MLXRandom.normal([1, 2, firstLength, dimension]).asType(.float16)
        let secondValues = MLXRandom.normal([1, 2, secondLength, dimension]).asType(.float16)
        let rope = RoPE(dimensions: dimension, traditional: false, base: 1_000_000)
        let cache = MossKVQuantCache(step: 8, outlierCount: dimension)
        let first = cache.attention(
            queries: rope(firstQueries),
            keys: firstKeys,
            values: firstValues,
            scale: scale,
            mask: .causal
        )
        eval(first)
        let actual = cache.attention(
            queries: rope(secondQueries, offset: firstLength),
            keys: secondKeys,
            values: secondValues,
            scale: scale,
            mask: .causal
        )
        let allKeys = concatenated([firstKeys, secondKeys], axis: 2)
        let allValues = concatenated([firstValues, secondValues], axis: 2)
        let expected = MLXFast.scaledDotProductAttention(
            queries: rope(secondQueries, offset: firstLength),
            keys: MLX.repeated(rope(allKeys), count: 2, axis: 1),
            values: MLX.repeated(allValues, count: 2, axis: 1),
            scale: scale,
            mask: .causal
        )
        eval(expected, actual)
        #expect(cache.offset == firstLength + secondLength)
        #expect(cache.packedKeyShape == [1, 2, 16, dimension / 2])
        #expect(MLX.abs(expected - actual).max().item(Float.self) < 0.03)
    }
}
