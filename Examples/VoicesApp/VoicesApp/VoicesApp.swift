import Darwin
import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTT
import SwiftUI
#if os(iOS)
import UIKit
#endif

private func processMemory() -> [String: UInt64] {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard status == KERN_SUCCESS else { return [:] }
    return [
        "physical_footprint_bytes": UInt64(info.phys_footprint),
        "resident_size_bytes": UInt64(info.resident_size)
    ]
}

private func peakResidentBytes() -> UInt64 {
    var usage = rusage()
    guard getrusage(RUSAGE_SELF, &usage) == 0 else { return 0 }
    return UInt64(usage.ru_maxrss)
}

private func mlxMemory() -> [String: Int] {
    let snapshot = Memory.snapshot()
    return [
        "active_bytes": snapshot.activeMemory,
        "cache_bytes": snapshot.cacheMemory,
        "peak_bytes": snapshot.peakMemory
    ]
}

@main
struct VoicesApp: App {
    var body: some Scene {
        WindowGroup {
            BenchmarkView()
        }
    }
}

private struct BenchmarkView: View {
    @State private var started = false
    @State private var status = "Starting"

    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
            Text("MOSS Optimization")
                .font(.title2)
            Text(status)
                .font(.system(.body, design: .monospaced))
                .multilineTextAlignment(.center)
        }
        .padding()
        .task {
            guard !started else { return }
            started = true
            await runBenchmark()
        }
    }

    @MainActor
    private func runBenchmark() async {
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = true
        UIDevice.current.isBatteryMonitoringEnabled = true
        #endif
        do {
            let environment = ProcessInfo.processInfo.environment
            let variant = environment["MOSS_BENCHMARK_VARIANT"] ?? "baseline"
            let chunkDuration = Float(environment["MOSS_CHUNK_DURATION"] ?? "30") ?? 30
            let kvBits = environment["MOSS_KV_BITS"].flatMap(Int.init)
            let modelDirectoryName = environment["MOSS_MODEL_DIRECTORY"] ?? "MOSS-Transcribe-Diarize"
            let modelIdentifier = environment["MOSS_MODEL_IDENTIFIER"] ?? "OpenMOSS-Team/MOSS-Transcribe-Diarize"
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let modelDirectory = documents.appendingPathComponent("Models/\(modelDirectoryName)", isDirectory: true)
            let audioURL = documents.appendingPathComponent("Media/children_vs_adults.m4a")
            status = "Loading model"
            let processBeforeModelLoad = processMemory()
            let loadStarted = Date()
            let model = try await MossTranscribeDiarizeModel.fromModelDirectory(modelDirectory)
            let modelLoadSeconds = Date().timeIntervalSince(loadStarted)
            let processAfterModelLoad = processMemory()
            status = "Loading audio"
            let audioLoadStarted = Date()
            let (sampleRate, audio) = try loadAudioArray(from: audioURL, sampleRate: 16_000)
            let audioLoadSeconds = Date().timeIntervalSince(audioLoadStarted)
            let audioSeconds = Double(audio.size) / Double(sampleRate)
            let parameters = STTGenerateParameters(
                maxTokens: 4_096,
                temperature: 0,
                topP: 1,
                topK: 0,
                verbose: false,
                language: nil,
                chunkDuration: chunkDuration,
                minChunkDuration: 1,
                repetitionPenalty: 1,
                repetitionContextSize: 100,
                kvBits: kvBits,
                kvGroupSize: 64,
                quantizedKVStart: 0
            )
            status = "Warming up"
            let warmupAudio = audio[0..<min(audio.size, sampleRate * 10)]
            let warmupStarted = Date()
            _ = model.generate(audio: warmupAudio, generationParameters: parameters)
            let warmupSeconds = Date().timeIntervalSince(warmupStarted)
            Memory.peakMemory = 0
            status = "Benchmarking \(String(format: "%.0f", chunkDuration))s chunks"
            let processBeforeGeneration = processMemory()
            let mlxBeforeGeneration = mlxMemory()
            let generationStarted = Date()
            let output = model.generate(audio: audio, generationParameters: parameters)
            let generationSeconds = Date().timeIntervalSince(generationStarted)
            let processAfterGeneration = processMemory()
            let mlxAfterGeneration = mlxMemory()
            #if os(iOS)
            let device: [String: Any] = [
                "battery_level": UIDevice.current.batteryLevel,
                "model": UIDevice.current.model,
                "name": UIDevice.current.name,
                "system_name": UIDevice.current.systemName,
                "system_version": UIDevice.current.systemVersion,
                "thermal_state": ProcessInfo.processInfo.thermalState.rawValue
            ]
            #else
            let device: [String: Any] = [
                "model": "Mac",
                "name": Host.current().localizedName ?? "unknown",
                "system_name": "macOS",
                "system_version": ProcessInfo.processInfo.operatingSystemVersionString,
                "thermal_state": ProcessInfo.processInfo.thermalState.rawValue
            ]
            #endif
            let result: [String: Any] = [
                "audio": [
                    "duration_seconds": audioSeconds,
                    "load_seconds": audioLoadSeconds,
                    "path": "Media/children_vs_adults.m4a"
                ],
                "created_at": ISO8601DateFormatter().string(from: Date()),
                "device": device,
                "generation": [
                    "generation_tokens": output.generationTokens,
                    "generation_tokens_per_second": output.generationTps,
                    "prompt_tokens": output.promptTokens,
                    "prompt_tokens_per_second": output.promptTps,
                    "seconds": generationSeconds,
                    "speed_factor": audioSeconds / generationSeconds
                ],
                "implementation": [
                    "backend": "MLX Metal",
                    "branch": "codex/moss-optimization",
                    "repository": "MOSSOptimization"
                ],
                "memory": [
                    "mlx_after_generation": mlxAfterGeneration,
                    "mlx_before_generation": mlxBeforeGeneration,
                    "process_after_generation": processAfterGeneration,
                    "process_after_model_load": processAfterModelLoad,
                    "process_before_generation": processBeforeGeneration,
                    "process_before_model_load": processBeforeModelLoad,
                    "process_peak_resident_bytes": peakResidentBytes()
                ],
                "model": modelIdentifier,
                "model_load_seconds": modelLoadSeconds,
                "schema_version": 1,
                "segments": output.segments ?? [],
                "settings": [
                    "chunk_duration_seconds": chunkDuration,
                    "kv_bits": kvBits.map { $0 as Any } ?? NSNull(),
                    "maximum_tokens_per_chunk": 4_096,
                    "temperature": 0
                ],
                "transcript": output.text,
                "type": "moss_optimization_benchmark",
                "variant": variant,
                "warmup_seconds": warmupSeconds
            ]
            let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
            let resultDirectory = documents.appendingPathComponent("Results", isDirectory: true)
            try FileManager.default.createDirectory(at: resultDirectory, withIntermediateDirectories: true)
            let resultURL = resultDirectory.appendingPathComponent("\(variant).json")
            try data.write(to: resultURL, options: .atomic)
            print("BENCHMARK_RESULT_PATH \(resultURL.path)")
            print("BENCHMARK_RESULT \(String(decoding: data, as: UTF8.self))")
            status = String(format: "Complete: %.2fx realtime", audioSeconds / generationSeconds)
        } catch {
            print("BENCHMARK_ERROR \(String(reflecting: error))")
            status = "Error: \(error.localizedDescription)"
        }
    }
}
