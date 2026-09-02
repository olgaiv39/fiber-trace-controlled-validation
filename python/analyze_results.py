#!/usr/bin/env python3
"""Compute minimal reference-polyline distances for controlled native traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def nearest_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    start, target = polyline[:-1], polyline[1:]
    delta = target - start
    length2 = np.sum(delta * delta, axis=1)
    distances = np.full(len(points), np.inf, dtype=np.float64)
    for a, d, d2 in zip(start, delta, length2):
        if d2 <= 0.0:
            continue
        t = np.clip(np.sum((points - a) * d, axis=1) / d2, 0.0, 1.0)
        distances = np.minimum(distances, np.linalg.norm(points - (a + t[:, None] * d), axis=1))
    return distances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spans = json.loads(args.spans.read_text(encoding="utf-8"))
    result_by_id = {item["case_id"]: item for item in json.loads(args.results.read_text(encoding="utf-8"))["results"]}
    case_by_id = {item["id"]: item for item in spans["cases"]}
    rows = []
    for case_id, case in case_by_id.items():
        result = result_by_id[case_id]
        path = np.asarray(result["fused_path_xyz"], dtype=np.float64)
        reference = np.asarray(case["reference_polyline_xyz_local"], dtype=np.float64)
        row = {"case_id": case_id, "accepted": result["accepted"]}
        if len(path):
            intended = nearest_polyline_distances(path, reference)
            row.update({"mean_distance_to_intended": float(np.mean(intended)), "max_distance_to_intended": float(np.max(intended))})
            competing_reference = case.get("competing_reference_polyline_xyz_local")
            if competing_reference is not None:
                competing = nearest_polyline_distances(path, np.asarray(competing_reference, dtype=np.float64))
                row.update({
                    "competing_tree_id": case["competing_tree_id"],
                    "mean_distance_to_competing": float(np.mean(competing)),
                    "fraction_points_closer_to_competing": float(np.mean(competing < intended)),
                })
        rows.append(row)
    args.output.write_text(json.dumps({"results": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
