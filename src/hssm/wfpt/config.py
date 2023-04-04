"""
Add specification
"""


default_model_config = {
    "ddm": {
        "loglik_kind": "analytical",
        "list_params": ["v", "sv", "a", "z", "t"],
        "backend": "pytensor",
        "formula": "c(rt,response)  ~ 1",
        "default": {
            "v": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "sv": {"name": "Uniform", "lower": 0.0, "upper": 1.0},
            "a": {"name": "Uniform", "lower": 0.30, "upper": 2.5},
            "z": {"name": "Uniform", "lower": 0.10, "upper": 0.9},
            "t": {"name": "Uniform", "lower": 0.0, "upper": 2.0},
        },
        "default_boundaries": [[-3.0, 0.0, 0.3, 0.1, 0.0], [3.0, 1.0, 2.5, 0.9, 2.0]],
    },
    "angle": {
        "loglik_kind": "approx_differentiable",
        "loglik_path": "test.onnx",
        "list_params": ["v", "a", "z", "t", "theta"],
        "backend": "jax",
        "formula": "c(rt,response)  ~ 1",
        "default": {
            "v": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
            "a": {"name": "Uniform", "lower": 0.0, "upper": 1.2},
            "z": {"name": "Uniform", "lower": 0.0, "upper": 2.01},
            "t": {"name": "Uniform", "lower": 0.0, "upper": 0.90},
            "theta": {"name": "Uniform", "lower": 0.0, "upper": 2.01},
        },
        "default_boundaries": [
            [-3.0, 0.3, 0.1, 0.001, -0.1],
            [3.0, 3.0, 0.9, 2.0, 1.3],
        ],
    },
}
