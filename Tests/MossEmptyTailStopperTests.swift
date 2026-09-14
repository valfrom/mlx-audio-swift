//
//  MossEmptyTailStopperTests.swift
//  MLXAudioSwift
//
//  Created by Valerii Ivanov on 11.09.2026.
//  Copyright © 2026 TapMediaLtd. All rights reserved.
//

import Testing

@testable import MLXAudioSTT

struct MossEmptyTailStopperTests {
    @Test func stopsOnlyAfterAnEmptySegmentReachesTheEnd() {
        expectStops([false, true], after: [
            "[0][S01][1][1][S02][2.07]",
            "[2.07][S01][2.08]",
        ], end: 2.074625)
    }

    @Test func keepsSpeechAtTheEndAndEmptySegmentsBeforeIt() {
        expectStops([false, false, true], after: [
            "[0][S01][1][1][S01]Слава Україні![3]",
            "[3][S02]Still speech[4]",
            "[4][S02] \n[4]",
        ], end: 3)
    }

    @Test func handlesSplitTagsCommaDecimalsAndOffset() {
        let deltas = "[30][S01][33,49][33,49][S02][33,50".map(String.init)
        expectStops(Array(repeating: false, count: deltas.count) + [true], after: deltas + ["]"], end: 33.5)
    }

    @Test func supportsTimestampOnlyEmptyOutput() {
        expectStops([false, true], after: ["[0]speech[2]", "[2][2]"], end: 2)
    }

    @Test func doesNotStopWhenTheSameDeltaContainsMoreSpeech() {
        expectStops([false, false, false, true], after: [
            "[0][S01][2][2][S01]More[3]",
            "[3][S01][3][3][S01]Still speaking",
            "[4]",
            "[4][S01][4][",
        ], end: 2)
    }

    @Test func ignoresIncompleteReversedAndInvalidTimestamps() {
        expectStops([false, true], after: ["[4][S01][3][0][S02][3", "]"], end: 3)
        for duration in [Double.nan, .infinity, 0, -1] {
            expectStops([false], after: ["[0][S01][3]"], end: duration)
        }
    }

    private func expectStops(_ expected: [Bool], after deltas: [String], end: Double) {
        var stopper = MossEmptyTailStopper(end: end)
        let results = deltas.map { stopper.consume($0) }
        #expect(results == expected)
    }
}
