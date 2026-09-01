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
    @Test func cacheSchemeSelection() {
        var scheme = MossKVCacheScheme.modelPrecision
        scheme = .kvQuant4

        #expect(scheme == .kvQuant4)
    }

    @Test func packedStorage() {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        MLXRandom.seed(7)
        let length = 128
        let dimension = 128
        let keys = MLXRandom.normal([1, 2, length, dimension]).asType(.float16)
        let values = MLXRandom.normal([1, 2, length, dimension]).asType(.float16)
        let simple = MossKVQuantCache.make()
        _ = simple.update(keys: keys, values: values)
        let cache = MossKVQuantCache.converting(simple)
        eval(cache.state)
        #expect(cache.offset == length)
        #expect(cache.state[0].shape == [1, 2, 256, dimension / 4])
        #expect(cache.state[3].shape == [1, 2, 256, dimension / 8])
        #expect(cache.state.reduce(0) { $0 + $1.nbytes } < 2 * 1 * 2 * 256 * dimension * 2)
        #expect(!cache.state.contains { $0.shape == [1, 2, 256, dimension] })
    }

    @Test func quantizedAttentionConsumesPackedCache() {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        MLXRandom.seed(11)
        let length = 12
        let dimension = 128
        let scale = pow(Float(dimension), -0.5)
        let queries = MLXRandom.normal([1, 4, length, dimension]).asType(.float16)
        let keys = MLXRandom.normal([1, 2, length, dimension]).asType(.float16)
        let values = MLXRandom.normal([1, 2, length, dimension]).asType(.float16)
        let expected = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: MLX.repeated(keys, count: 2, axis: 1),
            values: MLX.repeated(values, count: 2, axis: 1),
            scale: scale,
            mask: .causal
        )
        let cache = MossKVQuantCache()
        let actual = cache.attention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .causal
        )
        eval(expected, actual)
        #expect(MLX.abs(expected - actual).max().item(Float.self) < 0.3)
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
        let cache = MossKVQuantCache()
        _ = cache.attention(
            queries: firstQueries,
            keys: firstKeys,
            values: firstValues,
            scale: scale,
            mask: .causal
        )
        let actual = cache.attention(
            queries: secondQueries,
            keys: secondKeys,
            values: secondValues,
            scale: scale,
            mask: .causal
        )
        let expected = MLXFast.scaledDotProductAttention(
            queries: secondQueries,
            keys: MLX.repeated(concatenated([firstKeys, secondKeys], axis: 2), count: 2, axis: 1),
            values: MLX.repeated(concatenated([firstValues, secondValues], axis: 2), count: 2, axis: 1),
            scale: scale,
            mask: .causal
        )
        eval(expected, actual)
        #expect(cache.offset == firstLength + secondLength)
        #expect(MLX.abs(expected - actual).max().item(Float.self) < 0.2)
    }
}
