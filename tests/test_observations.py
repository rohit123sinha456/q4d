import numpy as np

from q4d_wam.observations import find_leaf, leaf_summary


def test_nested_observations_are_discovered_without_camera_name_assumptions() -> None:
    rgb = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    depth = np.ones((2, 8, 8, 1), dtype=np.float32)
    observation = {"sensor_data": {"base_camera": {"rgb": rgb, "depth": depth}}}

    assert find_leaf(observation, ".rgb") is rgb
    assert find_leaf(observation, ".depth") is depth
    assert leaf_summary(observation)["obs.sensor_data.base_camera.rgb"]["shape"] == [2, 8, 8, 3]

