#!/usr/bin/env python3
"""Verify metric depth backprojection and coordinate frames on ManiSkill PushCube."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from q4d_wam.config import load_config
from q4d_wam.geometry import (
    backproject_depth_cv,
    camera_cv_to_gl,
    camera_cv_to_world,
    camera_gl_to_world,
    invert_rigid_transform,
    transform_points,
)


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    valid = np.isfinite(xyz).all(axis=-1)
    xyz = xyz[valid]
    rgb = rgb[valid].astype(np.uint8)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(header)
        for point, color in zip(xyz, rgb, strict=True):
            stream.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def _save_plot(
    path: Path, xyz: np.ndarray, rgb: np.ndarray, cube_center: np.ndarray, max_points: int = 5000
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(xyz) > max_points:
        indices = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
        xyz = xyz[indices]
        rgb = rgb[indices]

    figure = plt.figure(figsize=(8, 6), dpi=160)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb / 255.0, s=1, depthshade=False)
    axis.scatter(*cube_center, c="magenta", marker="x", s=80, linewidths=2, label="cube pose")
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_zlabel("world z (m)")
    axis.set_title("PushCube metric RGB-D backprojection")
    axis.legend(loc="upper right")
    axis.set_box_aspect((1.0, 1.0, 0.7))
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _metrics(errors: torch.Tensor) -> dict[str, float]:
    return {
        "mean_m": float(errors.mean()),
        "p95_m": float(torch.quantile(errors, 0.95)),
        "max_m": float(errors.max()),
    }


def run_verification(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    software_icd = Path("/usr/share/vulkan/icd.d/lvp_icd.json")
    if software_icd.exists():
        os.environ.setdefault("VK_ICD_FILENAMES", str(software_icd))

    import mani_skill.envs  # noqa: F401 -- environment registration after Vulkan selection

    env = gym.make(
        config.simulation.env_id,
        num_envs=1,
        obs_mode="rgb+depth+position+segmentation",
        control_mode=config.simulation.control_mode,
        sim_backend="physx_cpu",
        render_backend="sapien_cpu",
        render_mode="sensors",
    )
    try:
        observation, _ = env.reset(seed=config.project.seed)
        camera_name = next(iter(observation["sensor_data"]))
        sensor = observation["sensor_data"][camera_name]
        calibration = observation["sensor_param"][camera_name]

        depth_mm = sensor["depth"]
        rgb = sensor["rgb"]
        position_gl_mm = sensor["position"]
        segmentation = sensor["segmentation"]
        intrinsics = calibration["intrinsic_cv"]
        extrinsic_cv = calibration["extrinsic_cv"]
        cam2world_gl = calibration["cam2world_gl"]

        points_cv, valid = backproject_depth_cv(depth_mm, intrinsics, max_depth_m=10.0)
        points_gl = camera_cv_to_gl(points_cv)
        points_world_cv = camera_cv_to_world(points_cv, extrinsic_cv)
        points_world_gl = camera_gl_to_world(points_gl, cam2world_gl)

        native_points_gl = position_gl_mm[..., :3].to(torch.float32) / 1000.0
        native_points_world = camera_gl_to_world(native_points_gl, cam2world_gl)

        frame_errors = torch.linalg.vector_norm(
            points_world_cv[valid] - points_world_gl[valid], dim=-1
        )
        native_camera_errors = torch.linalg.vector_norm(
            points_gl[valid] - native_points_gl[valid], dim=-1
        )
        native_world_errors = torch.linalg.vector_norm(
            points_world_cv[valid] - native_points_world[valid], dim=-1
        )

        unwrapped = env.unwrapped
        cube_id = int(unwrapped.obj.per_scene_id[0])
        cube_pose = unwrapped.obj.pose.to_transformation_matrix()
        cube_center = unwrapped.obj.pose.p[0]
        cube_mask = valid & (segmentation.squeeze(-1) == cube_id)
        cube_world = points_world_cv[cube_mask]
        if len(cube_world) == 0:
            raise RuntimeError(f"no depth pixels found for cube segmentation id {cube_id}")
        cube_local = transform_points(cube_world, invert_rigid_transform(cube_pose[0]))
        half_size = float(unwrapped.cube_half_size)
        absolute_local = cube_local.abs()
        aabb_violation = torch.relu(absolute_local - half_size).max(dim=-1).values
        nearest_face_distance = (absolute_local - half_size).abs().min(dim=-1).values

        rotation = extrinsic_cv[..., :3, :3]
        rotation_orthogonality = rotation @ rotation.transpose(-1, -2)
        identity = torch.eye(3, device=rotation.device).expand_as(rotation_orthogonality)

        report: dict[str, Any] = {
            "environment": config.simulation.env_id,
            "seed": config.project.seed,
            "camera": camera_name,
            "depth": {
                "raw_dtype": str(depth_mm.dtype),
                "unit": "millimetres",
                "image_shape": list(depth_mm.shape),
                "valid_pixels": int(valid.sum()),
                "minimum_m": float((depth_mm[valid[..., None]].to(torch.float32) / 1000).min()),
                "maximum_m": float((depth_mm[valid[..., None]].to(torch.float32) / 1000).max()),
            },
            "calibration": {
                "intrinsic_cv": _tensor_to_numpy(intrinsics[0]).tolist(),
                "extrinsic_cv_world_to_camera": _tensor_to_numpy(extrinsic_cv[0]).tolist(),
                "cam2world_gl": _tensor_to_numpy(cam2world_gl[0]).tolist(),
                "rotation_orthogonality_max_error": float(
                    (rotation_orthogonality - identity).abs().max()
                ),
            },
            "cv_vs_gl_world_route": _metrics(frame_errors),
            "depth_vs_native_position_camera": _metrics(native_camera_errors),
            "depth_vs_native_position_world": _metrics(native_world_errors),
            "cube_geometry": {
                "segmentation_id": cube_id,
                "visible_pixels": int(cube_mask.sum()),
                "half_size_m": half_size,
                "pose_p_world_m": _tensor_to_numpy(cube_center).tolist(),
                "aabb_violation_max_m": float(aabb_violation.max()),
                "surface_distance_mean_m": float(nearest_face_distance.mean()),
                "surface_distance_p95_m": float(torch.quantile(nearest_face_distance, 0.95)),
            },
        }

        checks = {
            "valid_depth_pixels": int(valid.sum()) > 1000,
            "calibration_rotation": report["calibration"]["rotation_orthogonality_max_error"]
            < 1e-5,
            "cv_gl_routes_agree": report["cv_vs_gl_world_route"]["max_m"] < 1e-5,
            "native_position_agrees": report["depth_vs_native_position_world"]["p95_m"]
            < 0.003,
            "cube_is_visible": report["cube_geometry"]["visible_pixels"] > 5,
            "cube_points_inside_bounds": report["cube_geometry"]["aabb_violation_max_m"]
            < 0.003,
            "cube_points_on_surface": report["cube_geometry"]["surface_distance_p95_m"]
            < 0.004,
        }
        report["checks"] = checks
        report["passed"] = all(checks.values())

        valid_world = _tensor_to_numpy(points_world_cv[valid])
        valid_rgb = _tensor_to_numpy(rgb[valid])
        visualization_mask = valid & (depth_mm.squeeze(-1) <= 1500)
        visualization_world = _tensor_to_numpy(points_world_cv[visualization_mask])
        visualization_rgb = _tensor_to_numpy(rgb[visualization_mask])
        report["artifacts"] = {
            "full_cloud_points": len(valid_world),
            "workspace_cloud_points": len(visualization_world),
            "workspace_depth_limit_m": 1.5,
        }
        np.savez_compressed(
            output_dir / "backprojection.npz",
            xyz_world_m=valid_world,
            rgb=valid_rgb,
            intrinsic_cv=_tensor_to_numpy(intrinsics[0]),
            extrinsic_cv=_tensor_to_numpy(extrinsic_cv[0]),
            cam2world_gl=_tensor_to_numpy(cam2world_gl[0]),
            cube_pose_world=_tensor_to_numpy(cube_pose[0]),
        )
        _write_ply(
            output_dir / "backprojection_workspace.ply",
            visualization_world,
            visualization_rgb,
        )
        _save_plot(
            output_dir / "backprojection_world.png",
            visualization_world,
            visualization_rgb,
            _tensor_to_numpy(cube_center),
        )
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))

        if not report["passed"]:
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(f"backprojection verification failed: {', '.join(failed)}")
        return report
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.toml"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/geometry/backprojection")
    )
    args = parser.parse_args()
    run_verification(args.config, args.output_dir)


if __name__ == "__main__":
    main()
