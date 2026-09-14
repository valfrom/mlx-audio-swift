//
//  MossEmptyTailStopper.swift
//  MLXAudioSwift
//
//  Created by Valerii Ivanov on 11.09.2026.
//  Copyright © 2026 TapMediaLtd. All rights reserved.
//

import Foundation

struct MossEmptyTailStopper {
    private static let pattern = try! NSRegularExpression(
        pattern: #"\[(\d+(?:[\.,]\d+)?)\](?:\[S\d+\])?(.*?)\[(\d+(?:[\.,]\d+)?)\]"#,
        options: [.dotMatchesLineSeparators]
    )
    let end: Double
    private var buffer = ""

    init(end: Double) {
        self.end = end
    }

    mutating func consume(_ text: String) -> Bool {
        guard end.isFinite, end > 0 else { return false }
        buffer += text
        guard text.contains("]") else { return false }
        let input = buffer as NSString
        let matches = Self.pattern.matches(in: buffer, range: NSRange(location: 0, length: input.length))
        guard let last = matches.last else { return false }
        buffer = input.substring(from: NSMaxRange(last.range))
        guard let start = Double(input.substring(with: last.range(at: 1)).replacingOccurrences(of: ",", with: ".")),
              let timestamp = Double(input.substring(with: last.range(at: 3)).replacingOccurrences(of: ",", with: ".")),
              start.isFinite, timestamp.isFinite, timestamp >= start, timestamp >= end,
              input.substring(with: last.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return false }
        let remaining = buffer.replacingOccurrences(of: #"\[(?:S\d+|\d+(?:[\.,]\d+)?)\]"#, with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return remaining.isEmpty || remaining.range(of: #"^\[(?:S\d*|\d*(?:[\.,]\d*)?)$"#, options: .regularExpression) != nil
    }
}
