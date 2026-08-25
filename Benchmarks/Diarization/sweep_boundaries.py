#
#  sweep_boundaries.py
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
            if record.get("type") == "diarization_result" and record.get("split") == "dev"
        ]


def fill_boundaries(record, maximum_gap, assignment, scope):
    result = copy.deepcopy(record)
    segments = sorted(result["segments"], key=lambda segment: (segment["start"], segment["end"]))
    for left, right in zip(segments, segments[1:]):
        gap = right["start"] - left["end"]
        if gap <= 0 or gap > maximum_gap:
            continue
        if scope == "same" and left["speaker"] != right["speaker"]:
            continue
        if assignment == "left":
            left["end"] = right["start"]
        elif assignment == "right":
            right["start"] = left["end"]
        else:
            midpoint = (left["end"] + right["start"]) / 2
            left["end"] = midpoint
            right["start"] = midpoint
    result["segments"] = segments
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
    baseline = {
        "parameters": {"maximum_gap_seconds": 0.0, "assignment": "none", "scope": "none"},
        "dev": score(records, "baseline", args),
    }
    candidates = []
    for maximum_gap, assignment, scope in itertools.product(
        (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0),
        ("left", "midpoint", "right"),
        ("same", "all"),
    ):
        parameters = {
            "maximum_gap_seconds": maximum_gap,
            "assignment": assignment,
            "scope": scope,
        }
        transformed = [fill_boundaries(record, maximum_gap, assignment, scope) for record in records]
        metrics = score(transformed, f"candidate-{len(candidates):03d}", args)
        candidates.append({"parameters": parameters, "dev": metrics})
        print(json.dumps(candidates[-1], separators=(",", ":")), flush=True)
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
    report = {
        "baseline": baseline,
        "candidate_count": len(candidates),
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
