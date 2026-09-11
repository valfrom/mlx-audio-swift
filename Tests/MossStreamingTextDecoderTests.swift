//
//  MossStreamingTextDecoderTests.swift
//  MLXAudioSwift
//
//  Created by Valerii Ivanov on 11.09.2026.
//  Copyright © 2026 TapMediaLtd. All rights reserved.
//

import Foundation
import Hub
import Testing
import Tokenizers

@testable import MLXAudioSTT

struct MossStreamingTextDecoderTests {
    @Test(arguments: [
        "й щ ї І Й Щ Ї Є є ґ Україна",
        "你好，世界。简体中文与繁體中文測試！",
        "中文 English 123，標點：！？；「漢字」𠀀",
        "😀 👩🏽‍💻 e\u{301} и\u{306}",
        "ASCII 123\t spaces\nnext line\r\n",
    ])
    func splitEveryUTF8Byte(_ text: String) throws {
        let tokenizer = try makeTokenizer()
        let tokens = tokenizer.encode(text: text, addSpecialTokens: false)
        #expect(tokens.count == text.utf8.count)
        var decoder = makeDecoder(tokenizer)
        var output = ""
        for token in tokens {
            let delta = decoder.consume(token)
            #expect(!delta.contains("\u{FFFD}"))
            output += delta
            #expect(Array(text.utf8).starts(with: Array(output.utf8)))
        }
        output += decoder.finish()
        #expect(Array(output.utf8) == Array(text.utf8))
        #expect(Array(output.utf8) == Array(tokenizer.decode(tokens: tokens, skipSpecialTokens: true).utf8))
        #expect(decoder.finish().isEmpty)
    }

    @Test func reproducedUkrainianTokenPairs() throws {
        let tokenizer = try makeTokenizer()
        for (tokens, expected) in [([1278, 117], " й"), ([8839, 231], " щ")] {
            #expect(tokens.map { tokenizer.decode(tokens: [$0]) }.joined() == " ��")
            var decoder = makeDecoder(tokenizer)
            #expect(decoder.consume(tokens[0]).isEmpty)
            #expect(decoder.consume(tokens[1]) == expected)
            #expect(decoder.finish().isEmpty)
        }
    }

    @Test func skipsSpecialTokensInsideSplitCharacter() throws {
        let tokenizer = try makeTokenizer()
        var decoder = makeDecoder(tokenizer)
        #expect(decoder.consume(1278).isEmpty)
        #expect(decoder.consume(256).isEmpty)
        #expect(decoder.consume(117) == " й")
        #expect(decoder.consume(256).isEmpty)
        #expect(decoder.finish().isEmpty)
    }

    @Test func offsetsSplitTagsAndResetsBetweenChunks() throws {
        let tokenizer = try makeTokenizer()
        let text = "[0.48][S01]你好，їжак😀[1,66][2"
        let tokens = tokenizer.encode(text: text, addSpecialTokens: false)
        for offset in [0.0, 30.0, 60.0] {
            var decoder = makeDecoder(tokenizer, offset: offset)
            var output = tokens.map { decoder.consume($0) }.joined()
            output += decoder.finish()
            let expected = MossTranscribeDiarizeModel.offsetTimestampTags(in: text, by: offset)
            #expect(Array(output.utf8) == Array(expected.utf8))
            #expect(decoder.finish().isEmpty)
        }
        var unfinished = makeDecoder(tokenizer)
        #expect(unfinished.consume(1278).isEmpty)
        var next = makeDecoder(tokenizer)
        let output = tokenizer.encode(text: "中文", addSpecialTokens: false).map { next.consume($0) }.joined() + next.finish()
        #expect(output == "中文")
    }

    @Test func emptyAndASCIIDeltas() throws {
        let tokenizer = try makeTokenizer()
        var decoder = makeDecoder(tokenizer)
        #expect(decoder.finish().isEmpty)
        for token in tokenizer.encode(text: "A \n", addSpecialTokens: false) {
            #expect(decoder.consume(token) == tokenizer.decode(tokens: [token]))
        }
        #expect(decoder.finish().isEmpty)
    }

    @Test func preservesLiteralReplacementAndIncompleteEnd() throws {
        let tokenizer = try makeTokenizer()
        let cases = [
            tokenizer.encode(text: "a�中文", addSpecialTokens: false),
            tokenizer.encode(text: "a�", addSpecialTokens: false),
            [1278],
            [8839],
        ]
        for tokens in cases {
            var decoder = makeDecoder(tokenizer)
            let output = tokens.map { decoder.consume($0) }.joined() + decoder.finish()
            let expected = tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
            #expect(Array(output.utf8) == Array(expected.utf8))
            #expect(decoder.finish().isEmpty)
        }
    }

    private func makeDecoder(_ tokenizer: any Tokenizer, offset: Double = 0) -> MossStreamingTextDecoder {
        MossStreamingTextDecoder(offsetSeconds: offset) {
            tokenizer.decode(tokens: $0, skipSpecialTokens: true)
        }
    }

    private func makeTokenizer() throws -> any Tokenizer {
        let visible = Set(Array(33...126) + Array(161...172) + Array(174...255))
        var extra = 256
        var vocab: [String: Int] = ["<skip>": 256, "ĠÐ": 1278, "ĠÑ": 8839]
        for byte in 0...255 {
            let scalar = visible.contains(byte) ? byte : extra
            if !visible.contains(byte) { extra += 1 }
            vocab[String(UnicodeScalar(scalar)!)] = byte == 185 ? 117 : byte == 137 ? 231 : 1000 + byte
        }
        let config = Config([
            "tokenizer_class": "GPT2Tokenizer",
            "clean_up_tokenization_spaces": false,
            "additional_special_tokens": ["<skip>"],
        ] as [NSString: Any])
        let data = Config([
            "added_tokens": [["id": 256, "content": "<skip>", "special": true]],
            "model": ["type": "BPE", "vocab": vocab, "merges": []] as [String: Any],
            "pre_tokenizer": ["type": "ByteLevel", "add_prefix_space": false, "use_regex": false] as [String: Any],
            "decoder": ["type": "ByteLevel"],
        ] as [NSString: Any])
        return try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
    }
}
