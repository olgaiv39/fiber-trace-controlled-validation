from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VILLA_UPSTREAM = Path(
    os.environ.get("VILLA_UPSTREAM_MAIN", ROOT.parent / "villa-upstream-main")
).resolve()
sys.path.insert(0, str(ROOT / "python"))
import generate_controlled_fields as generator


def upstream_normal_encoding():
    path = VILLA_UPSTREAM / "lasagna" / "normal_encoding.py"
    spec = importlib.util.spec_from_file_location("upstream_normal_encoding", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def encode_3x2(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)

    def pair(a: float, b: float) -> tuple[float, float]:
        denominator = a * a + b * b + 1.0e-12
        cos2 = (a * a - b * b) / denominator
        sin2 = 2.0 * a * b / denominator
        return 0.5 + 0.5 * cos2, 0.5 + 0.5 * (cos2 - sin2) / np.sqrt(2.0)

    return np.asarray([*pair(vector[0], vector[1]), *pair(vector[0], vector[2]), *pair(vector[1], vector[2])])


class ControlledFieldTests(unittest.TestCase):
    def test_simple_path_order_and_subpath(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            nml = Path(temp) / "tiny.nml"
            nml.write_text("""<things><thing id=\"a\"><nodes>
                <node id=\"1\" x=\"0\" y=\"0\" z=\"0\"/>
                <node id=\"2\" x=\"1\" y=\"0\" z=\"0\"/>
                <node id=\"3\" x=\"2\" y=\"0\" z=\"0\"/>
                </nodes><edges><edge source=\"2\" target=\"3\"/><edge source=\"1\" target=\"2\"/></edges></thing></things>""")
            tree = generator.load_simple_open_tree(nml, "a")
            self.assertEqual(set(tree.ordered_node_ids), {"1", "2", "3"})
            subpath = generator.ordered_subpath(tree, "1", "3")
            np.testing.assert_array_equal(subpath[:, 0], [0.0, 1.0, 2.0])

    def test_xyz_zyx_round_trip(self) -> None:
        origin = np.array([10.0, 20.0, 30.0])
        xyz = np.array([[12.0, 24.0, 35.0]])
        np.testing.assert_allclose(generator.local_zyx_indices_to_xyz(generator.local_xyz_to_zyx_indices(xyz, origin), origin), xyz)

    def test_compact_axis_matches_upstream_normal_helper(self) -> None:
        helper = upstream_normal_encoding()
        for vector in (np.array([0.3, 0.4, 0.8660254]), np.array([-0.6, 0.2, -0.7745967])):
            local_nx, local_ny = generator.compact_axis_u8(vector)
            encoded = encode_3x2(vector)
            expected_nx, expected_ny = helper.encode_normal_nxny_u8(*encoded)
            self.assertEqual(int(local_nx), int(expected_nx))
            self.assertEqual(int(local_ny), int(expected_ny))

    def test_separated_tubes_do_not_form_support_bridge(self) -> None:
        presence, _, _ = generator.rasterize_prediction(
            (5, 12, 20), np.zeros(3),
            [np.array([[1.0, 3.0, 2.0], [18.0, 3.0, 2.0]]), np.array([[1.0, 8.0, 2.0], [18.0, 8.0, 2.0]])],
            1.25,
        )
        self.assertTrue(np.all(presence[:, 5:7, :] == 0))

    def test_zarr_manifest_and_constant_normal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "field.zarr"
            array = np.full((3, 4, 5), 128, dtype=np.uint8)
            generator.write_zarr_u8(path, array)
            metadata = json.loads((path / ".zarray").read_text())
            self.assertEqual(metadata["dtype"], "|u1")
            self.assertEqual(metadata["shape"], [3, 4, 5])
            self.assertEqual((path / "0.0.0").read_bytes()[:1], bytes([128]))

    def test_generation_is_deterministic(self) -> None:
        nml = VILLA_UPSTREAM / "foundation/datasets/fibers-dataset/fibers_s5_06500z_02000y_04000x_500_v03.nml"
        cases = ROOT / "cases/smoke_spans.json"
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "one", Path(temp) / "two"
            generator.generate(nml, cases, first, 1.25, 32)
            generator.generate(nml, cases, second, 1.25, 32)
            def digest(root: Path) -> str:
                hasher = hashlib.sha256()
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        hasher.update(path.relative_to(root).as_posix().encode())
                        hasher.update(path.read_bytes())
                return hasher.hexdigest()
            self.assertEqual(digest(first), digest(second))
            spans = json.loads((first / "spans_local.json").read_text())
            self.assertEqual(spans["roi_shape_zyx"], [90, 99, 289])
            field_manifest = json.loads(
                (first / "fields" / "difficult_343_with_831" / "fiber.lasagna.json").read_text())
            self.assertEqual(set(field_manifest["groups"]), {"presence", "nx", "ny"})
            for name in ("presence", "nx", "ny"):
                self.assertEqual(
                    json.loads((first / "fields" / "difficult_343_with_831" / f"{name}.zarr" / ".zarray").read_text())["dtype"],
                    "|u1")
            self.assertEqual((first / "normal" / "grad_mag.zarr" / "0.0.0").read_bytes()[0], 255)
            self.assertEqual((first / "normal" / "nx.zarr" / "0.0.0").read_bytes()[0], 128)
            self.assertEqual((first / "normal" / "ny.zarr" / "0.0.0").read_bytes()[0], 128)
            normal_manifest = json.loads((first / "normal" / "normal.lasagna.json").read_text())
            self.assertEqual(normal_manifest["grad_mag_encode_scale"], 255.0)
            self.assertEqual(normal_manifest["grad_mag_factor"], 1.0)
            self.assertEqual(255.0 / (normal_manifest["grad_mag_encode_scale"] /
                                      normal_manifest["grad_mag_factor"]), 1.0)


if __name__ == "__main__":
    unittest.main()
