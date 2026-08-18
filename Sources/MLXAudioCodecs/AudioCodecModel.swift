import Foundation
import MLX

extension MLXArray {
    static func arange(_ stop: Int) -> MLXArray {
        MLXArray(0..<stop)
    }

    static func arange(_ stop: Int, dtype: DType) -> MLXArray {
        arange(stop).asType(dtype)
    }

    static func arange(_ start: Int, _ stop: Int, dtype: DType) -> MLXArray {
        MLXArray(start..<stop).asType(dtype)
    }

    static func arange(_ start: Int, _ stop: Int, step: Int, dtype: DType) -> MLXArray {
        MLXArray(Swift.stride(from: start, to: stop, by: step)).asType(dtype)
    }
}

public protocol AudioDecoderModel {
    associatedtype DecoderInput

    /// Sampling rate in Hz when known. Returns nil when model output rate depends on external context.
    var codecSampleRate: Double? { get }

    /// Decode latent/features into waveform audio.
    /// Expected waveform layout is model-specific.
    func decodeAudio(_ input: DecoderInput) -> MLXArray
}

public protocol AudioCodecModel: AudioDecoderModel where DecoderInput == EncodedAudio {
    associatedtype EncodedAudio

    /// Encode waveform audio into model-specific latent representation.
    /// Expected waveform layout is model-specific.
    func encodeAudio(_ waveform: MLXArray) -> EncodedAudio
}

public extension AudioCodecModel {
    func reconstruct(_ waveform: MLXArray) -> MLXArray {
        decodeAudio(encodeAudio(waveform))
    }
}
