# Metric backprojection and coordinate frames

Verified on ManiSkill 3.0.1 and SAPIEN 3.0.3 using `PushCube-v1`, seed 7.

## Depth convention

ManiSkill RGB-D depth is a signed 16-bit axial depth image in millimetres. Pixel rays are
evaluated at pixel centres. For pixel row `v`, column `u`, calibration
`(fx, fy, cx, cy)`, and raw depth `d`:

```text
z = d / 1000
x = (u + 0.5 - cx) * z / fx
y = (v + 0.5 - cy) * z / fy
```

This produces metres in the OpenCV camera frame:

- `+x`: image right
- `+y`: image down
- `+z`: camera forward

Raw depths less than or equal to zero are invalid and their returned XYZ values are zeroed.

## World transforms

`extrinsic_cv` is a 3x4 world-to-OpenCV-camera rigid transform, so camera points are moved
to world coordinates with its inverse.

`cam2world_gl` instead accepts OpenGL-camera coordinates. OpenCV camera points convert to
OpenGL with `(x, y, z) -> (x, -y, -z)`. Both routes are tested against each other for every
valid pixel.

## Simulator-backed verification

The verification requests RGB, depth, position, and segmentation in one render. It checks:

1. OpenCV and OpenGL calibration routes produce the same world point.
2. Reconstructed points agree with SAPIEN's native position texture.
3. Pixels carrying the cube's actor segmentation ID lie within the simulator's 4 cm cube.
4. Visible cube pixels lie on a cube face within the depth quantization tolerance.

Observed results:

| Check | Result |
| --- | ---: |
| Valid depth pixels | 16,384 |
| OpenCV/OpenGL route maximum difference | 0.000000060 m |
| Native position agreement, mean | 0.000574 m |
| Native position agreement, 95th percentile | 0.001038 m |
| Visible cube pixels | 20 |
| Cube AABB maximum violation | 0.000551 m |
| Cube surface distance, 95th percentile | 0.000551 m |

The approximately 1 mm native-position discrepancy is expected from integer-millimetre
texture quantization. The full numerical report is generated at
`artifacts/geometry/backprojection/report.json`.

