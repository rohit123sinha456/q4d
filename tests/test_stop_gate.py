from q4d_wam.evaluation.decision import evaluate_stop_gate


def _prediction(ade: float, *, cache_speedup: float = 2.0) -> dict:
    return {
        "passed": True,
        "test": {
            "groups": {
                group: {"ade_m": ade}
                for group in ("all", "moving", "contact", "object")
            }
        },
        "candidate_benchmark": {"cached_speedup": cache_speedup},
    }


def _mpc() -> dict:
    return {
        "passed": True,
        "conditions": [
            {
                "model": "q4d",
                "episodes": 2,
                "success_rate": 0.5,
                "mean_final_cube_goal_distance_m": 0.1,
            },
            {
                "model": "no_action",
                "episodes": 2,
                "success_rate": 0.0,
                "mean_final_cube_goal_distance_m": 0.2,
            },
        ],
        "episodes": [
            {"model": model, "seed": seed}
            for model in ("q4d", "no_action")
            for seed in (1, 2)
        ],
    }


def test_stop_gate_continues_only_when_all_hypotheses_pass() -> None:
    thresholds = {
        "cache_min_speedup": 1.0,
        "dense_overall_ade_ratio_max": 1.25,
        "dense_task_ade_ratio_max": 1.10,
        "mpc_min_success_rate_margin": 0.2,
    }
    cache = {
        "passed": True,
        "rows": [
            {
                "cache_speedup": 1.5,
                "maximum_output_difference_m": 0.0001,
            }
        ],
        "checks": {"cache_matches_reencoding": True},
    }
    passing = evaluate_stop_gate(
        no_action=_prediction(0.02),
        q4d=_prediction(0.01, cache_speedup=0.8),
        dense=_prediction(0.0095),
        cache_grid=cache,
        mpc=_mpc(),
        thresholds=thresholds,
    )
    assert passing["decision"] == "continue"
    assert all(passing["gates"].values())

    cache["rows"][0]["cache_speedup"] = 0.9
    failing = evaluate_stop_gate(
        no_action=_prediction(0.02),
        q4d=_prediction(0.01),
        dense=_prediction(0.0095),
        cache_grid=cache,
        mpc=_mpc(),
        thresholds=thresholds,
    )
    assert failing["decision"] == "stop_and_diagnose"
    assert not failing["gates"]["scene_caching_improves_throughput"]
