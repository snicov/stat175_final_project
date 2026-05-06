#!/usr/bin/env python3
"""Lightweight validation checks for the simulation pipeline."""

from __future__ import annotations

import numpy as np

from dgp import generate_dataset
from exposure_mapping import estimate_radius
from sim_utils import erdos_renyi_adjacency


def main() -> None:
    rng = np.random.default_rng(123)
    adjacency = erdos_renyi_adjacency(50, 0.2, rng)
    assert adjacency.shape == (50, 50)
    assert np.all(adjacency == adjacency.T)
    assert np.all(np.diag(adjacency) == 0)

    config = {
        "scenario_name": "validation",
        "seeds": [1],
        "n": 80,
        "r_true": 20.0,
        "spillover_type": "friend_average",
        "distance": {"type": "uniform_square", "map_size": 80.0},
        "network": {"type": "erdos_renyi", "p": 0.2},
        "treatment": {"type": "bernoulli", "pi": 0.5},
        "outcome": {"alpha_true": 0.0, "tau_true": 1.0, "gamma_true": 20.0, "sigma2": 1.0, "beta_true": {}},
        "estimator": {
            "radius_grid": [10.0, 20.0, 30.0],
            "band_edges": [10.0, 20.0, 30.0],
            "B": 5,
            "psi_dictionary": "annulus_treated_share",
        },
    }
    generated = generate_dataset(config, seed=999)
    assert {"Y", "W", "S_true", "qualifying_friend_count"}.issubset(generated.unit_df.columns)
    assert generated.distance_true.shape == (80, 80)
    assert generated.distance_observed.shape == (80, 80)

    result = estimate_radius(
        unit_df=generated.unit_df,
        distance_observed=generated.distance_observed,
        config=config,
        rng=np.random.default_rng(456),
        save_outputs=False,
    )
    assert result.selection["B"] == 5
    assert result.selection["phi"] == "identity: phi_m(Y_i)=Y_i"
    assert result.selection["r_hat"] in {10.0, 20.0, 30.0}
    print("Validation checks passed.")


if __name__ == "__main__":
    main()

