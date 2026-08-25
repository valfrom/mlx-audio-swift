<!--
//
//  README.md
//  MOSSOptimization
//
//  Created by Valerii Ivanov on 26.08.2026.
//  Copyright © 2026 TapMediaLtd. All rights reserved.
//
-->

# MOSS optimization benchmark

This branch benchmarks `MOSS-Transcribe-Diarize` on physical Apple devices and evaluates diarization on the complete VoxConverse dev and test sets.

The selected configuration uses the official prompt, a 4-bit text decoder with the Whisper encoder and adaptor left in BF16, and fills positive gaps of at most 0.75 seconds only between adjacent segments assigned to the same speaker. The boundary policy was selected on VoxConverse dev and frozen before test scoring.

## Quality

`sweep_boundaries.py` selects the boundary policy on dev. `apply_boundary_policy.py` applies the frozen policy and scores overall, dev, test, and speaker-count strata with the VoxSRC 2022 scorer using a 0.25-second collar, included overlap, and 0.01-second step.

```sh
python Benchmarks/Diarization/apply_boundary_policy.py \
  --source RESULT.jsonl \
  --project-root EXPERIMENT_RESULTS \
  --scorer-root VOXSRC2022 \
  --temporary-root /Volumes/XBOX/tmp/MOSSOptimization/main/score \
  --output PROCESSED_RESULT.jsonl
```

## Physical-device benchmark

`VoicesApp` reads the model from `Documents/Models`, reads `Documents/Media/children_vs_adults.m4a`, warms up on ten seconds, processes the full recording, and saves transcript, postprocessed segments, speed, thermal state, and memory counters to `Documents/Results`.

The authoritative configurations are:

- Baseline: original BF16 model, greedy decoding, 30-second chunks.
- Optimized: `vanch007/mlx-MOSS-Transcribe-Diarize-4bit`, official prompt, greedy decoding, 30-second chunks, 0.75-second same-speaker boundary fill.

## Verification

```sh
swift test \
  --scratch-path /Volumes/XBOX/tmp/MOSSOptimization/main/swift-test \
  --filter mossParseSegmentsFillsShortSameSpeakerGaps
```

The committed measurement report and raw outputs are in the sibling `ExperimentResults/results/moss-optimization` directory.
