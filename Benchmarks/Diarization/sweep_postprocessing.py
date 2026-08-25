#
#  sweep_postprocessing.py
#  MOSSOptimization
#
#  Created by Valerii Ivanov on 25.08.2026.
#  Copyright © 2026 TapMediaLtd. All rights reserved.
#

import argparse
import copy
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scorer-root", type=Path, required=True)
    parser.add_argument("--temporary-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_records(path):
    with path.open(encoding="utf-8") as stream:
        return [
            record
            for line in stream
            if line.strip()
            for record in [json.loads(line)]
            if record.get("type") == "diarization_result"
        ]


def adjacent_speaker(segments, index, excluded, totals):
    candidates = []
    if index > 0 and segments[index - 1]["speaker"] not in excluded:
        candidates.append(segments[index - 1]["speaker"])
    if index + 1 < len(segments) and segments[index + 1]["speaker"] not in excluded:
        candidates.append(segments[index + 1]["speaker"])
    if candidates:
        return max(candidates, key=lambda speaker: totals[speaker])
    remaining = [speaker for speaker in totals if speaker not in excluded]
    return max(remaining, key=totals.get) if remaining else segments[index]["speaker"]


def postprocess(record, island_seconds, minimum_speaker_seconds, minimum_speaker_fraction):
    result = copy.deepcopy(record)
    segments = sorted(result["segments"], key=lambda segment: (segment["start"], segment["end"]))
    if island_seconds > 0:
        for index in range(1, len(segments) - 1):
            segment = segments[index]
            if (
                segment["end"] - segment["start"] <= island_seconds
                and segments[index - 1]["speaker"] == segments[index + 1]["speaker"]
            ):
                segment["speaker"] = segments[index - 1]["speaker"]
    totals = {}
    for segment in segments:
        totals[segment["speaker"]] = totals.get(segment["speaker"], 0.0) + segment["end"] - segment["start"]
    total_speech = sum(totals.values())
    threshold = max(minimum_speaker_seconds, total_speech * minimum_speaker_fraction)
    rare = {speaker for speaker, duration in totals.items() if duration < threshold}
    if rare and len(rare) < len(totals):
        for index, segment in enumerate(segments):
            if segment["speaker"] in rare:
                segment["speaker"] = adjacent_speaker(segments, index, rare, totals)
    result["segments"] = segments
    result["predicted_speaker_count"] = len({segment["speaker"] for segment in segments})
    return result


def write_inputs(records, project_root, reference_list, system_rttm):
    with reference_list.open("w", encoding="utf-8") as references:
        for record in records:
            references.write(str(project_root / record["reference_path"]) + "\n")
    with system_rttm.open("w", encoding="utf-8") as system:
        for record in records:
            for segment in record["segments"]:
                duration = segment["end"] - segment["start"]
                if duration > 0:
                    system.write(
                        f"SPEAKER {record['recording_id']} 1 {segment['start']:.6f} {duration:.6f} "
                        f"<NA> <NA> speaker_{segment['speaker']} <NA> <NA>\n"
                    )


def score(records, label, args):
    reference_list = args.temporary_root / f"{label}-references.scp"
    system_rttm = args.temporary_root / f"{label}-system.rttm"
    write_inputs(records, args.project_root, reference_list, system_rttm)
    completed = subprocess.run(
        [
            sys.executable,
            str(args.scorer_root / "compute_diarisation_metrics.py"),
            "-R",
            str(reference_list),
            "-s",
            str(system_rttm),
            "--collar",
            "0.25",
            "--step",
            "0.01",
        ],
        cwd=args.scorer_root,
        check=True,
        capture_output=True,
        text=True,
    )
    der = re.search(r"Primary Metric DER=([0-9.eE+-]+)", completed.stdout)
    jer = re.search(r"JER=([0-9.eE+-]+)", completed.stdout)
    if not der or not jer:
        raise RuntimeError(completed.stdout + completed.stderr)
    return {
        "diarization_error_rate_percent": float(der.group(1)),
        "jaccard_error_rate_percent": float(jer.group(1)),
        "recording_count": len(records),
    }


def main():
    args = arguments()
    args.project_root = args.project_root.resolve()
    args.temporary_root.mkdir(parents=True, exist_ok=True)
    records = load_records(args.source)
    dev = [record for record in records if record["split"] == "dev"]
    candidates = []
    for island, minimum_seconds, minimum_fraction in itertools.product(
        (0.0, 0.25, 0.5, 1.0),
        (0.0, 0.25, 0.5, 1.0, 2.0),
        (0.0, 0.002, 0.005),
    ):
        parameters = {
            "island_seconds": island,
            "minimum_speaker_seconds": minimum_seconds,
            "minimum_speaker_fraction": minimum_fraction,
        }
        transformed = [postprocess(record, island, minimum_seconds, minimum_fraction) for record in dev]
        metrics = score(transformed, f"dev-{len(candidates):03d}", args)
        candidates.append({"parameters": parameters, "dev": metrics})
        print(json.dumps(candidates[-1], separators=(",", ":")), flush=True)
    baseline = candidates[0]
    eligible = [
        candidate
        for candidate in candidates
        if candidate["dev"]["diarization_error_rate_percent"] < baseline["dev"]["diarization_error_rate_percent"]
        and candidate["dev"]["jaccard_error_rate_percent"] < baseline["dev"]["jaccard_error_rate_percent"]
    ]
    selected = min(
        eligible or candidates,
        key=lambda candidate: candidate["dev"]["diarization_error_rate_percent"]
        + candidate["dev"]["jaccard_error_rate_percent"],
    )
    parameters = selected["parameters"]
    processed = [postprocess(record, **parameters) for record in records]
    selected["test"] = score(
        [record for record in processed if record["split"] == "test"],
        "selected-test",
        args,
    )
    selected["overall"] = score(processed, "selected-overall", args)
    report = {"baseline": baseline, "candidate_count": len(candidates), "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
