#
#  apply_boundary_policy.py
#  MOSSOptimization
#
#  Created by Valerii Ivanov on 26.08.2026.
#  Copyright © 2026 TapMediaLtd. All rights reserved.
#

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sweep_boundaries import fill_boundaries, score


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scorer-root", type=Path, required=True)
    parser.add_argument("--temporary-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-gap", type=float, default=0.75)
    return parser.parse_args()


def load_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def comparison(moss, sortformer):
    return {
        "moss": moss,
        "sortformer": sortformer,
        "moss_minus_sortformer_der_percentage_points":
            moss["diarization_error_rate_percent"] - sortformer["diarization_error_rate_percent"],
        "moss_minus_sortformer_jer_percentage_points":
            moss["jaccard_error_rate_percent"] - sortformer["jaccard_error_rate_percent"],
    }


def main():
    args = arguments()
    args.project_root = args.project_root.resolve()
    args.temporary_root.mkdir(parents=True, exist_ok=True)
    source = load_jsonl(args.source)
    metadata = next(record for record in source if record["type"] == "benchmark_metadata")
    source_summary = next(record for record in source if record["type"] == "comparison_summary")
    policy = {
        "maximum_gap_seconds": args.maximum_gap,
        "assignment": "right",
        "scope": "same_speaker",
        "selection_split": "dev",
    }
    records = []
    for record in source:
        if record.get("type") != "diarization_result":
            continue
        transformed = fill_boundaries(record, args.maximum_gap, "right", "same")
        transformed["postprocessing"] = policy
        records.append(transformed)

    scopes = {
        "overall": records,
        "dev": [record for record in records if record["split"] == "dev"],
        "test": [record for record in records if record["split"] == "test"],
    }
    strata = {
        label: [record for record in records if record["speaker_bin"] == label]
        for label in ("1-2", "3-4", "5+")
    }
    metrics = {
        label: score(scope, f"processed-{label}", args)
        for label, scope in scopes.items()
    }
    stratum_metrics = {
        label: score(scope, f"processed-speakers-{label.replace('+', 'plus')}", args)
        for label, scope in strata.items()
    }

    processed_metadata = dict(metadata)
    processed_metadata["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    processed_metadata["experiment_id"] += "-same-speaker-gap-0.75s"
    processed_metadata["postprocessing"] = policy
    try:
        processed_metadata["source_result"] = str(args.source.resolve().relative_to(args.project_root))
    except ValueError:
        processed_metadata["source_result"] = str(args.source)

    summary = dict(source_summary)
    summary["completed_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary["postprocessing"] = policy
    summary["raw_moss"] = {
        label: source_summary["matched_sample"][label]["moss"]
        for label in scopes
    }
    summary["matched_sample"] = {
        label: comparison(metrics[label], source_summary["matched_sample"][label]["sortformer"])
        for label in scopes
    }
    summary["speaker_strata"] = {
        label: comparison(stratum_metrics[label], source_summary["speaker_strata"][label]["sortformer"])
        for label in strata
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for record in [processed_metadata, *records, summary]:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
