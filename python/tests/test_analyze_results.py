from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import analyze_results


def result(mode: str, *, accepted: bool, path: list[list[float]]) -> dict:
    return {
        "case_id": "case",
        "prediction_source_mode": mode,
        "accepted": accepted,
        "reason": "ok" if accepted else "meeting_error_threshold",
        "forward_reason": "ok",
        "reverse_reason": "ok",
        "fused_path_xyz": path,
        "forward_points_xyz": path[:2],
        "reverse_points_xyz": path[1:],
        "meeting_error_base_voxels": 1.0 if accepted else 5.0,
        "meeting_error_ratio": 0.01 if accepted else 0.5,
        "fused_path_length_trace_voxels": 2.0,
        "forward_endpoint_error_trace_voxels": 0.1,
        "reverse_endpoint_error_trace_voxels": 0.2,
        "max_endpoint_error_trace_voxels": 0.2,
        "max_endpoint_error_base_voxels": 0.2,
    }


class AnalyzeResultsTests(unittest.TestCase):
    def test_mode_comparisons_keep_wrapper_control_separate(self) -> None:
        path = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        spans = {
            "cases": [{
                "id": "case",
                "reference_polyline_xyz_local": path,
            }]
        }
        native = result("native", accepted=False, path=path)
        delegated = result("delegated", accepted=False, path=path)
        zero_invalid = result("zero_invalid", accepted=True, path=path)
        zero_invalid["meeting_error_base_voxels"] = 1.0
        zero_invalid["meeting_error_ratio"] = 0.01

        output = analyze_results.analyze(
            spans, {"results": [native, delegated, zero_invalid]})

        self.assertEqual(
            [row["prediction_source_mode"] for row in output["results"]],
            ["native", "delegated", "zero_invalid"],
        )
        comparisons = {item["comparison"]: item for item in output["mode_comparisons"]}
        self.assertTrue(comparisons["native_vs_delegated"]["strictly_equivalent"])
        self.assertFalse(comparisons["delegated_vs_zero_invalid"]["strictly_equivalent"])
        self.assertEqual(
            comparisons["delegated_vs_zero_invalid"]["causal_interpretation"],
            "UNCONFOUNDED_BY_DELEGATION",
        )
        self.assertEqual(
            comparisons["native_vs_delegated"]["forward_path"]["max_abs_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
