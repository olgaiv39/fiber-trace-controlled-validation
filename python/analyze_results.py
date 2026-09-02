#!/usr/bin/env python3
"""Compute minimal reference distances and prediction-source-mode comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PREDICTION_SOURCE_MODES = ("native", "delegated", "zero_invalid")
COMPARISON_TOLERANCE = 1.0e-12


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


def path_comparison(left: Any, right: Any) -> dict[str, Any]:
    if left == right:
        return {"exact_equal": True, "max_abs_delta": 0.0}
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        return {
            "exact_equal": False,
            "max_abs_delta": None,
            "left_shape": list(left_array.shape),
            "right_shape": list(right_array.shape),
        }
    return {
        "exact_equal": False,
        "max_abs_delta": float(np.max(np.abs(left_array - right_array))) if left_array.size else 0.0,
    }


def scalar_delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    left_value = left.get(key)
    right_value = right.get(key)
    if not isinstance(left_value, (int, float)) or not isinstance(right_value, (int, float)):
        return None
    return float(abs(left_value - right_value))


def scalar_strictly_equivalent(left: dict[str, Any], right: dict[str, Any], key: str) -> bool:
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value is None or right_value is None:
        return left_value is None and right_value is None
    if not isinstance(left_value, (int, float)) or not isinstance(right_value, (int, float)):
        return False
    return abs(left_value - right_value) <= COMPARISON_TOLERANCE


def compare_results(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "meeting_error_base_voxels",
        "meeting_error_ratio",
        "fused_path_length_trace_voxels",
        "forward_endpoint_error_trace_voxels",
        "reverse_endpoint_error_trace_voxels",
        "max_endpoint_error_trace_voxels",
        "max_endpoint_error_base_voxels",
    )
    comparison: dict[str, Any] = {
        "accepted_equal": left.get("accepted") == right.get("accepted"),
        "reason_equal": left.get("reason") == right.get("reason"),
        "forward_reason_equal": left.get("forward_reason") == right.get("forward_reason"),
        "reverse_reason_equal": left.get("reverse_reason") == right.get("reverse_reason"),
        "fused_path": path_comparison(left.get("fused_path_xyz", []), right.get("fused_path_xyz", [])),
        "forward_path": path_comparison(left.get("forward_points_xyz", []), right.get("forward_points_xyz", [])),
        "reverse_path": path_comparison(left.get("reverse_points_xyz", []), right.get("reverse_points_xyz", [])),
    }
    for field in fields:
        comparison[f"{field}_abs_delta"] = scalar_delta(left, right, field)
    comparison["strictly_equivalent"] = (
        comparison["accepted_equal"]
        and comparison["reason_equal"]
        and comparison["forward_reason_equal"]
        and comparison["reverse_reason_equal"]
        and comparison["fused_path"]["max_abs_delta"] is not None
        and comparison["forward_path"]["max_abs_delta"] is not None
        and comparison["reverse_path"]["max_abs_delta"] is not None
        and comparison["fused_path"]["max_abs_delta"] <= COMPARISON_TOLERANCE
        and comparison["forward_path"]["max_abs_delta"] <= COMPARISON_TOLERANCE
        and comparison["reverse_path"]["max_abs_delta"] <= COMPARISON_TOLERANCE
        and all(scalar_strictly_equivalent(left, right, field) for field in fields)
    )
    return comparison


def result_distance_row(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    path = np.asarray(result["fused_path_xyz"], dtype=np.float64)
    reference = np.asarray(case["reference_polyline_xyz_local"], dtype=np.float64)
    row = {
        "case_id": case["id"],
        "prediction_source_mode": result["prediction_source_mode"],
        "accepted": result["accepted"],
    }
    if len(path):
        intended = nearest_polyline_distances(path, reference)
        row.update({
            "mean_distance_to_intended": float(np.mean(intended)),
            "max_distance_to_intended": float(np.max(intended)),
        })
        competing_reference = case.get("competing_reference_polyline_xyz_local")
        if competing_reference is not None:
            competing = nearest_polyline_distances(
                path, np.asarray(competing_reference, dtype=np.float64))
            row.update({
                "competing_tree_id": case["competing_tree_id"],
                "mean_distance_to_competing": float(np.mean(competing)),
                "fraction_points_closer_to_competing": float(np.mean(competing < intended)),
            })
    return row


def analyze(spans: dict[str, Any], native_results: dict[str, Any]) -> dict[str, Any]:
    result_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for result in native_results["results"]:
        key = (result["case_id"], result["prediction_source_mode"])
        if key in result_by_key:
            raise ValueError(f"duplicate native result for case/mode {key}")
        result_by_key[key] = result

    rows = []
    comparisons = []
    cross_case_comparisons = []
    for case in spans["cases"]:
        case_id = case["id"]
        for mode in PREDICTION_SOURCE_MODES:
            result = result_by_key.get((case_id, mode))
            if result is not None:
                rows.append(result_distance_row(case, result))
        native = result_by_key.get((case_id, "native"))
        delegated = result_by_key.get((case_id, "delegated"))
        zero_invalid = result_by_key.get((case_id, "zero_invalid"))
        native_delegated_comparison = None
        if native is not None and delegated is not None:
            native_delegated_comparison = compare_results(native, delegated)
            comparisons.append({
                "case_id": case_id,
                "comparison": "native_vs_delegated",
                **native_delegated_comparison,
            })
        if delegated is not None and zero_invalid is not None:
            zero_invalid_comparison = compare_results(delegated, zero_invalid)
            if native_delegated_comparison is None:
                zero_invalid_comparison["causal_interpretation"] = "CONTROL_MISSING"
            elif native_delegated_comparison["strictly_equivalent"]:
                zero_invalid_comparison["causal_interpretation"] = "UNCONFOUNDED_BY_DELEGATION"
            else:
                zero_invalid_comparison["causal_interpretation"] = "CONFOUNDED_BY_DELEGATION"
            comparisons.append({
                "case_id": case_id,
                "comparison": "delegated_vs_zero_invalid",
                **zero_invalid_comparison,
            })

    for positive_case_id, difficult_case_id in (
        ("positive_343", "difficult_343_with_831"),
        ("positive_831", "difficult_831_with_343"),
    ):
        for mode in PREDICTION_SOURCE_MODES:
            positive = result_by_key.get((positive_case_id, mode))
            difficult = result_by_key.get((difficult_case_id, mode))
            if positive is not None and difficult is not None:
                cross_case_comparisons.append({
                    "comparison": "positive_vs_difficult",
                    "prediction_source_mode": mode,
                    "positive_case_id": positive_case_id,
                    "difficult_case_id": difficult_case_id,
                    **compare_results(positive, difficult),
                })
    return {
        "results": rows,
        "mode_comparisons": comparisons,
        "positive_difficult_comparisons": cross_case_comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spans = json.loads(args.spans.read_text(encoding="utf-8"))
    native_results = json.loads(args.results.read_text(encoding="utf-8"))
    args.output.write_text(
        json.dumps(analyze(spans, native_results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
