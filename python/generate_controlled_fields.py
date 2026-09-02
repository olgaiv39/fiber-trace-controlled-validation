#!/usr/bin/env python3
"""Generate a tiny manifest-backed FiberTrace field from selected NML paths.

This is a controlled geometry experiment, not neural inference.  Arrays are
uncompressed Zarr v2 uint8 arrays so the generator needs only Python stdlib and
NumPy; current VC readers consume the resulting Lasagna manifests directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


VILLA_COMMIT = "908aa7f06e326d6df5cf167ebaab1fc733466987"
DEFAULT_TUBE_RADIUS = 1.25
DEFAULT_MARGIN = 32
CHUNK_MAX = 64


@dataclass(frozen=True)
class NmlTree:
    tree_id: str
    positions: dict[str, np.ndarray]
    adjacency: dict[str, list[str]]
    ordered_node_ids: tuple[str, ...]

    @property
    def ordered_points(self) -> np.ndarray:
        return np.stack([self.positions[node_id] for node_id in self.ordered_node_ids])


def load_simple_open_tree(nml_path: Path, tree_id: str) -> NmlTree:
    root = ET.parse(nml_path).getroot()
    thing = next((item for item in root.findall("thing") if item.get("id") == str(tree_id)), None)
    if thing is None:
        raise ValueError(f"NML tree {tree_id!r} not found in {nml_path}")
    positions = {
        node.get("id"): np.array(
            [float(node.get("x")), float(node.get("y")), float(node.get("z"))], dtype=np.float64
        )
        for node in thing.findall("nodes/node")
    }
    if not positions:
        raise ValueError(f"NML tree {tree_id!r} has no nodes")
    adjacency = {node_id: [] for node_id in positions}
    for edge in thing.findall("edges/edge"):
        source, target = edge.get("source"), edge.get("target")
        if source not in adjacency or target not in adjacency:
            raise ValueError(f"NML tree {tree_id!r} edge references an unknown node")
        adjacency[source].append(target)
        adjacency[target].append(source)
    edge_count = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    endpoints = [node_id for node_id, neighbors in adjacency.items() if len(neighbors) == 1]
    if edge_count != len(positions) - 1 or len(endpoints) != 2 or any(
        len(neighbors) not in (1, 2) for neighbors in adjacency.values()
    ):
        raise ValueError(f"NML tree {tree_id!r} is not a connected simple open path")
    ordered: list[str] = []
    previous: str | None = None
    current = endpoints[0]
    while True:
        ordered.append(current)
        next_nodes = [node_id for node_id in adjacency[current] if node_id != previous]
        if not next_nodes:
            break
        if len(next_nodes) != 1:
            raise ValueError(f"NML tree {tree_id!r} branches while ordering")
        previous, current = current, next_nodes[0]
    if len(ordered) != len(positions):
        raise ValueError(f"NML tree {tree_id!r} is disconnected")
    return NmlTree(str(tree_id), positions, adjacency, tuple(ordered))


def ordered_subpath(tree: NmlTree, start_node_id: str, target_node_id: str) -> np.ndarray:
    try:
        start = tree.ordered_node_ids.index(str(start_node_id))
        target = tree.ordered_node_ids.index(str(target_node_id))
    except ValueError as exc:
        raise ValueError(f"requested endpoint is not on tree {tree.tree_id}") from exc
    node_ids = tree.ordered_node_ids[start : target + 1] if start <= target else tree.ordered_node_ids[target : start + 1][::-1]
    if len(node_ids) < 2:
        raise ValueError("requested endpoints must be distinct")
    return np.stack([tree.positions[node_id] for node_id in node_ids])


def local_xyz_to_zyx_indices(xyz: np.ndarray, origin_xyz: np.ndarray) -> np.ndarray:
    return (np.asarray(xyz, dtype=np.float64) - origin_xyz)[..., ::-1]


def local_zyx_indices_to_xyz(zyx: np.ndarray, origin_xyz: np.ndarray) -> np.ndarray:
    return np.asarray(zyx, dtype=np.float64)[..., ::-1] + origin_xyz


def compact_axis_u8(axis_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match current ChannelSampler.cpp compact axis decode convention."""
    axis = np.asarray(axis_xyz, dtype=np.float64)
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    # Background voxels have no owning segment. Their compact values are
    # overwritten with a neutral value by rasterize_prediction.
    axis = axis / np.where(norm > 1.0e-12, norm, 1.0)
    axis = np.where(axis[..., 2:3] < 0.0, -axis, axis)
    nx = np.clip(np.rint(axis[..., 0] * 127.0 + 128.0), 0.0, 255.0).astype(np.uint8)
    ny = np.clip(np.rint(axis[..., 1] * 127.0 + 128.0), 0.0, 255.0).astype(np.uint8)
    return nx, ny


def _nearest_segment_field(
    shape_zyx: tuple[int, int, int],
    origin_xyz: np.ndarray,
    polylines: Iterable[np.ndarray],
    tube_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-segment squared distance and tangent, with stable ownership."""
    z_size, y_size, x_size = shape_zyx
    best_dist2 = np.full(shape_zyx, np.inf, dtype=np.float32)
    best_tangent = np.zeros(shape_zyx + (3,), dtype=np.float32)
    for polyline in polylines:
        for start, target in zip(polyline[:-1], polyline[1:]):
            delta = target - start
            length2 = float(np.dot(delta, delta))
            if length2 <= 0.0:
                continue
            tangent = (delta / math.sqrt(length2)).astype(np.float32)
            low_xyz = np.maximum(
                np.floor(np.minimum(start, target) - tube_radius).astype(np.int64),
                origin_xyz.astype(np.int64),
            )
            high_xyz = np.minimum(
                np.ceil(np.maximum(start, target) + tube_radius).astype(np.int64),
                origin_xyz.astype(np.int64) + np.array([x_size - 1, y_size - 1, z_size - 1]),
            )
            if np.any(high_xyz < low_xyz):
                continue
            x0, y0, z0 = (low_xyz - origin_xyz.astype(np.int64)).astype(int)
            x1, y1, z1 = (high_xyz - origin_xyz.astype(np.int64) + 1).astype(int)
            zz, yy, xx = np.indices((z1 - z0, y1 - y0, x1 - x0), dtype=np.float32)
            grid = np.stack(
                [xx + x0 + origin_xyz[0], yy + y0 + origin_xyz[1], zz + z0 + origin_xyz[2]],
                axis=-1,
            )
            rel = grid - start
            t = np.clip(np.sum(rel * delta, axis=-1) / length2, 0.0, 1.0)
            nearest = start + t[..., None] * delta
            dist2 = np.sum((grid - nearest) ** 2, axis=-1)
            current_dist2 = best_dist2[z0:z1, y0:y1, x0:x1]
            update = dist2 < current_dist2
            current_dist2[update] = dist2[update]
            tangent_region = best_tangent[z0:z1, y0:y1, x0:x1]
            tangent_region[update] = tangent
    return best_dist2, best_tangent


def rasterize_prediction(
    shape_zyx: tuple[int, int, int],
    origin_xyz: np.ndarray,
    polylines: Iterable[np.ndarray],
    tube_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist2, tangent = _nearest_segment_field(shape_zyx, origin_xyz, polylines, tube_radius)
    support = dist2 <= float(tube_radius * tube_radius)
    presence = np.where(support, 255, 0).astype(np.uint8)
    nx, ny = compact_axis_u8(tangent)
    # Direction outside support is never used because presence is zero. Keep a
    # valid neutral compact axis to avoid undefined data in raw artifacts.
    nx = np.where(support, nx, 128).astype(np.uint8)
    ny = np.where(support, ny, 128).astype(np.uint8)
    return presence, nx, ny


def write_zarr_u8(path: Path, array: np.ndarray) -> None:
    """Write a one-level, uncompressed, C-order Zarr v2 uint8 array."""
    if array.dtype != np.uint8 or array.ndim != 3:
        raise ValueError("controlled Zarr writer accepts only 3-D uint8 arrays")
    path.mkdir(parents=True, exist_ok=True)
    chunk = [min(CHUNK_MAX, int(value)) for value in array.shape]
    metadata = {
        "zarr_format": 2,
        "shape": list(map(int, array.shape)),
        "chunks": chunk,
        "dtype": "|u1",
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": ".",
    }
    (path / ".zarray").write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    for z0 in range(0, array.shape[0], chunk[0]):
        for y0 in range(0, array.shape[1], chunk[1]):
            for x0 in range(0, array.shape[2], chunk[2]):
                z1, y1, x1 = min(z0 + chunk[0], array.shape[0]), min(y0 + chunk[1], array.shape[1]), min(x0 + chunk[2], array.shape[2])
                payload = array[z0:z1, y0:y1, x0:x1]
                # Zarr v2 stores complete chunks; pad boundary chunks to the
                # declared chunk shape using the zero fill value.
                full = np.zeros(tuple(chunk), dtype=np.uint8)
                full[: z1 - z0, : y1 - y0, : x1 - x0] = payload
                (path / f"{z0 // chunk[0]}.{y0 // chunk[1]}.{x0 // chunk[2]}").write_bytes(full.tobytes(order="C"))


def write_manifest(
    path: Path,
    shape_zyx: tuple[int, int, int],
    groups: dict[str, tuple[str, list[str]]],
    root_fields: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "version": 2,
        "source_to_base": 1.0,
        "base_shape_zyx": list(map(int, shape_zyx)),
        "groups": {
            name: {"zarr": zarr_path, "scaledown": 0, "channels": channels}
            for name, (zarr_path, channels) in groups.items()
        },
    }
    if root_fields is not None:
        document.update(root_fields)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def generate(nml_path: Path, cases_path: Path, output_dir: Path, tube_radius: float, margin: int) -> dict:
    source_cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    tree_ids = sorted({str(tree_id) for item in source_cases for tree_id in item["field_tree_ids"]})
    trees = {tree_id: load_simple_open_tree(nml_path, tree_id) for tree_id in tree_ids}
    references: dict[str, np.ndarray] = {}
    tree_subpaths: dict[str, np.ndarray] = {}
    for item in source_cases:
        tree_id = str(item["intended_tree_id"])
        references[item["id"]] = ordered_subpath(trees[tree_id], *item["endpoint_node_ids"])
        tree_subpaths.setdefault(tree_id, references[item["id"]])
    all_points = np.concatenate(list(references.values()), axis=0)
    minimum = np.floor(all_points.min(axis=0)).astype(np.int64) - int(margin)
    maximum = np.ceil(all_points.max(axis=0)).astype(np.int64) + int(margin)
    origin_xyz = minimum.astype(np.float64)
    shape_xyz = (maximum - minimum + 1).astype(np.int64)
    shape_zyx = tuple(int(value) for value in shape_xyz[::-1])
    if tube_radius <= 0.0:
        raise ValueError("tube radius must be positive")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    normal_root = output_dir / "normal"
    normal_arrays = {
        "grad_mag": np.full(shape_zyx, 255, dtype=np.uint8),
        "nx": np.full(shape_zyx, 128, dtype=np.uint8),
        "ny": np.full(shape_zyx, 128, dtype=np.uint8),
    }
    for name, array in normal_arrays.items():
        write_zarr_u8(normal_root / f"{name}.zarr", array)
    normal_manifest = normal_root / "normal.lasagna.json"
    write_manifest(normal_manifest, shape_zyx, {
        name: (f"{name}.zarr", [name]) for name in normal_arrays
    }, {"grad_mag_encode_scale": 255.0, "grad_mag_factor": 1.0})

    generated_cases = []
    for item in source_cases:
        case_id = item["id"]
        polylines = [tree_subpaths[str(tree_id)] for tree_id in item["field_tree_ids"]]
        presence, nx, ny = rasterize_prediction(shape_zyx, origin_xyz, polylines, tube_radius)
        field_root = output_dir / "fields" / case_id
        arrays = {"presence": presence, "nx": nx, "ny": ny}
        for name, array in arrays.items():
            write_zarr_u8(field_root / f"{name}.zarr", array)
        field_manifest = field_root / "fiber.lasagna.json"
        write_manifest(field_manifest, shape_zyx, {
            name: (f"{name}.zarr", [name]) for name in arrays
        })
        local_reference = references[case_id] - origin_xyz
        competing_ids = [str(tree_id) for tree_id in item["field_tree_ids"] if str(tree_id) != str(item["intended_tree_id"])]
        generated_cases.append({
            **item,
            "endpoint_xyz_local": local_reference[[0, -1]].tolist(),
            "reference_polyline_xyz_local": local_reference.tolist(),
            "reference_polyline_xyz_nml": references[case_id].tolist(),
            "competing_tree_id": int(competing_ids[0]) if len(competing_ids) == 1 else None,
            "competing_reference_polyline_xyz_local": (tree_subpaths[competing_ids[0]] - origin_xyz).tolist() if len(competing_ids) == 1 else None,
            "fiber_manifest": str(field_manifest.relative_to(output_dir)),
            "normal_manifest": str(normal_manifest.relative_to(output_dir)),
        })

    result = {
        "version": 1,
        "experiment": "controlled_real_nml_geometry_not_neural_inference",
        "villa_commit": VILLA_COMMIT,
        "nml_path": str(nml_path.resolve()),
        "nml_sha256": _path_digest([nml_path]),
        "coordinate_frame": "local XYZ = NML absolute XYZ - roi_origin_xyz_nml; arrays are local ZYX",
        "roi_origin_xyz_nml": origin_xyz.tolist(),
        "roi_shape_zyx": list(shape_zyx),
        "tube_radius_base_voxels": float(tube_radius),
        "presence_encoding": "uint8: support=255, background=0",
        "axis_encoding": "current VC compact sign-invariant upper-hemisphere nx/ny uint8",
        "normal_field": "controlled_constant_planar_normal: +Z, grad_mag=255, nx=128, ny=128",
        "normal_manifest": str(normal_manifest.relative_to(output_dir)),
        "cases": generated_cases,
    }
    (output_dir / "spans_local.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nml", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tube-radius", type=float, default=DEFAULT_TUBE_RADIUS)
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    args = parser.parse_args()
    result = generate(args.nml, args.cases, args.output, args.tube_radius, args.margin)
    print(json.dumps({"roi_origin_xyz_nml": result["roi_origin_xyz_nml"], "roi_shape_zyx": result["roi_shape_zyx"]}, sort_keys=True))


if __name__ == "__main__":
    main()
