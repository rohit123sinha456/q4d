from pathlib import Path

from q4d_wam.config import load_config


def test_smoke_config_is_small_enough_for_eight_gb_gpu() -> None:
    config = load_config(Path("configs/smoke.toml"))

    assert config.simulation.env_id == "PushCube-v1"
    assert config.model.n_scene_points == 512
    assert config.model.n_query_points <= 32
    assert config.model.horizon <= 4
    assert config.model.width <= 128

