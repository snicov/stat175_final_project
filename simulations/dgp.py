#!/usr/bin/env python3
"""Data-generating processes for the spillover simulation study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sim_utils import (
    apply_distance_noise,
    bernoulli_treatment,
    central_nodes_adjacency,
    compute_distance_matrix,
    covariate_homophily_adjacency,
    distance_decay_adjacency,
    erdos_renyi_adjacency,
    graph_diagnostics,
    sbm_adjacency,
    sigmoid,
)


@dataclass
class GeneratedData:
    """Container for one generated dataset."""

    unit_df: pd.DataFrame
    distance_true: np.ndarray
    distance_observed: np.ndarray
    adjacency: np.ndarray
    metadata: dict[str, Any]


def generate_covariates(n_units: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a fixed population with correlated observed covariates."""
    age = np.clip(rng.normal(40.0, 13.0, size=n_units), 18.0, 80.0)
    gender = rng.binomial(1, 0.5, size=n_units)
    education_levels = np.array(["high_school", "some_college", "college", "graduate"])
    education = rng.choice(education_levels, size=n_units, p=[0.30, 0.25, 0.30, 0.15])
    race_levels = np.array(["A", "B", "C", "D"])
    race = rng.choice(race_levels, size=n_units, p=[0.50, 0.25, 0.15, 0.10])

    education_income = {
        "high_school": 0.0,
        "some_college": 0.18,
        "college": 0.40,
        "graduate": 0.65,
    }
    race_income = {"A": 0.05, "B": -0.05, "C": 0.0, "D": -0.02}
    log_income = (
        10.0
        + 0.012 * (age - 40.0)
        + np.array([education_income[e] for e in education])
        + np.array([race_income[r] for r in race])
        + 0.05 * gender
        + rng.normal(0.0, 0.35, size=n_units)
    )
    income = np.exp(log_income)

    return pd.DataFrame(
        {
            "unit_id": np.arange(n_units, dtype=int),
            "age": age,
            "gender": gender.astype(int),
            "education": education.astype(str),
            "race": race.astype(str),
            "log_income": log_income,
            "income": income,
        }
    )


def generate_locations(n_units: int, config: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Generate observed locations and return x/y columns."""
    kind = config.get("type", "uniform_square")
    map_size = float(config.get("map_size", 150.0))
    if kind == "uniform_square":
        x = rng.uniform(0.0, map_size, size=n_units)
        y = rng.uniform(0.0, map_size, size=n_units)
    elif kind == "two_cluster_square":
        spread = float(config.get("cluster_sd", 12.0))
        left_n = n_units // 2
        right_n = n_units - left_n
        left = np.column_stack(
            [
                rng.normal(0.30 * map_size, spread, size=left_n),
                rng.normal(0.50 * map_size, spread, size=left_n),
            ]
        )
        right = np.column_stack(
            [
                rng.normal(0.70 * map_size, spread, size=right_n),
                rng.normal(0.50 * map_size, spread, size=right_n),
            ]
        )
        coords = np.vstack([left, right])
        coords = np.clip(coords, 0.0, map_size)
        x, y = coords[:, 0], coords[:, 1]
    else:
        raise ValueError(f"Unknown distance specification: {kind}")
    return pd.DataFrame({"x": x, "y": y})


def generate_adjacency(
    network_config: dict[str, Any],
    covariates: pd.DataFrame,
    distance_matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a latent adjacency matrix from a scenario config."""
    kind = network_config.get("type", "erdos_renyi")
    if kind == "erdos_renyi":
        adjacency = erdos_renyi_adjacency(len(covariates), float(network_config.get("p", 0.2)), rng)
        extra = {"network_type": kind, "p": float(network_config.get("p", 0.2))}
    elif kind == "covariate_homophily":
        adjacency = covariate_homophily_adjacency(
            covariates,
            intercept=float(network_config.get("intercept", -3.0)),
            similarity_scale=float(network_config.get("similarity_scale", 0.8)),
            rng=rng,
        )
        extra = {"network_type": kind}
    elif kind == "distance_decay":
        adjacency = distance_decay_adjacency(
            distance_matrix,
            intercept=float(network_config.get("intercept", 0.0)),
            decay=float(network_config.get("decay", 3.0)),
            rng=rng,
        )
        extra = {"network_type": kind}
    elif kind == "central_nodes":
        adjacency = central_nodes_adjacency(
            len(covariates),
            p=float(network_config.get("p", 0.05)),
            k=int(network_config.get("k", 5)),
            rng=rng,
        )
        extra = {"network_type": kind, "p": float(network_config.get("p", 0.05)), "k": int(network_config.get("k", 5))}
    elif kind == "sbm":
        adjacency, blocks = sbm_adjacency(
            len(covariates),
            n_blocks=int(network_config.get("n_blocks", 2)),
            p_in=float(network_config.get("p_in", 0.2)),
            p_out=float(network_config.get("p_out", 0.05)),
            rng=rng,
        )
        extra = {"network_type": kind, "blocks": blocks.tolist()}
    else:
        raise ValueError(f"Unknown network type: {kind}")
    return adjacency, extra


def compute_spillover(
    adjacency: np.ndarray,
    distance_matrix: np.ndarray,
    treatment: np.ndarray,
    r_true: float,
    spillover_type: str,
    expected_edge_probability: float | None = None,
    expected_edge_probability_matrix: np.ndarray | None = None,
    saturation_lambda: float = 0.10,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute latent spillover under the chosen specification."""
    within = (distance_matrix < float(r_true)).astype(int)
    np.fill_diagonal(within, 0)
    qualifying = adjacency * within
    treated_friend_count = (qualifying * treatment[None, :]).sum(axis=1)
    qualifying_friend_count = qualifying.sum(axis=1)
    treated_geo_count = (within * treatment[None, :]).sum(axis=1)
    if spillover_type == "friend_average":
        degree = adjacency.sum(axis=1)
        spillover = np.divide(
            treated_friend_count,
            degree,
            out=np.zeros_like(treated_friend_count, dtype=float),
            where=degree > 0,
        )
    elif spillover_type == "friend_sum":
        spillover = treated_friend_count.astype(float)
    elif spillover_type == "expected_er_friend_sum":
        if expected_edge_probability is None:
            raise ValueError("expected_er_friend_sum requires expected_edge_probability.")
        spillover = float(expected_edge_probability) * treated_geo_count.astype(float)
    elif spillover_type == "expected_er_saturated_friend_sum":
        if expected_edge_probability is None:
            raise ValueError("expected_er_saturated_friend_sum requires expected_edge_probability.")
        expected_count = float(expected_edge_probability) * treated_geo_count.astype(float)
        spillover = 1.0 - np.exp(-float(saturation_lambda) * expected_count)
    elif spillover_type == "expected_distance_decay_saturated_friend_sum":
        if expected_edge_probability_matrix is None:
            raise ValueError("expected_distance_decay_saturated_friend_sum requires expected_edge_probability_matrix.")
        expected_count = (expected_edge_probability_matrix * within * treatment[None, :]).sum(axis=1)
        spillover = 1.0 - np.exp(-float(saturation_lambda) * expected_count)
    elif spillover_type == "expected_homophily_saturated_friend_sum":
        if expected_edge_probability_matrix is None:
            raise ValueError("expected_homophily_saturated_friend_sum requires expected_edge_probability_matrix.")
        expected_count = (expected_edge_probability_matrix * within * treatment[None, :]).sum(axis=1)
        spillover = 1.0 - np.exp(-float(saturation_lambda) * expected_count)
    elif spillover_type == "expected_sbm_saturated_friend_sum":
        if expected_edge_probability_matrix is None:
            raise ValueError("expected_sbm_saturated_friend_sum requires expected_edge_probability_matrix.")
        expected_count = (expected_edge_probability_matrix * within * treatment[None, :]).sum(axis=1)
        spillover = 1.0 - np.exp(-float(saturation_lambda) * expected_count)
    elif spillover_type == "expected_central_hub_saturated_friend_sum":
        if expected_edge_probability_matrix is None:
            raise ValueError("expected_central_hub_saturated_friend_sum requires expected_edge_probability_matrix.")
        expected_count = (expected_edge_probability_matrix * within * treatment[None, :]).sum(axis=1)
        spillover = 1.0 - np.exp(-float(saturation_lambda) * expected_count)
    else:
        raise ValueError(f"Unknown spillover type: {spillover_type}")
    pieces = {
        "treated_friend_count": treated_friend_count.astype(int),
        "qualifying_friend_count": qualifying_friend_count.astype(int),
        "treated_geo_count_within_r_true": treated_geo_count.astype(int),
        "latent_degree": adjacency.sum(axis=1).astype(int),
    }
    return spillover, pieces


def build_design_matrix(covariates: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build an observed covariate design matrix for the DGP."""
    age_scaled = (covariates["age"].to_numpy(dtype=float) - 40.0) / 10.0
    gender = covariates["gender"].to_numpy(dtype=float)
    log_income_scaled = covariates["log_income"].to_numpy(dtype=float) - covariates["log_income"].mean()
    education = pd.get_dummies(covariates["education"], prefix="educ", drop_first=True)
    race = pd.get_dummies(covariates["race"], prefix="race", drop_first=True)
    design = pd.concat(
        [
            pd.Series(age_scaled, name="age_scaled"),
            pd.Series(gender, name="gender"),
            pd.Series(log_income_scaled, name="log_income_centered"),
            education.reset_index(drop=True),
            race.reset_index(drop=True),
        ],
        axis=1,
    )
    return design.to_numpy(dtype=float), list(design.columns)


def beta_vector(names: list[str], beta_config: dict[str, float]) -> np.ndarray:
    """Map beta config to design-matrix column order."""
    defaults = {
        "age_scaled": 0.10,
        "gender": 0.05,
        "log_income_centered": 0.20,
        "educ_some_college": 0.05,
        "educ_college": 0.10,
        "educ_graduate": 0.15,
        "race_B": -0.05,
        "race_C": 0.03,
        "race_D": -0.02,
    }
    defaults.update(beta_config or {})
    return np.array([float(defaults.get(name, 0.0)) for name in names], dtype=float)


def generate_dataset(config: dict[str, Any], seed: int) -> GeneratedData:
    """Generate one full dataset from a scenario/grid config and seed."""
    rng = np.random.default_rng(seed)
    n_units = int(config.get("n", config.get("n_units", 1000)))
    r_true = float(config.get("r_true", 30.0))

    covariates = generate_covariates(n_units, rng)
    locations = generate_locations(n_units, config.get("distance", {"type": "uniform_square"}), rng)
    unit_df = pd.concat([covariates, locations], axis=1)

    distance_true = compute_distance_matrix(unit_df["x"].to_numpy(), unit_df["y"].to_numpy())
    distance_noise_sd = float(config.get("distance_noise_sd", 0.0))
    distance_observed = apply_distance_noise(distance_true, distance_noise_sd, rng)

    adjacency, network_extra = generate_adjacency(config.get("network", {"type": "erdos_renyi", "p": 0.2}), unit_df, distance_true, rng)
    treatment_design = config.get("treatment", {"type": "bernoulli", "pi": 0.5})
    treatment = bernoulli_treatment(n_units, float(treatment_design.get("pi", 0.5)), rng)
    network_config = config.get("network", {})
    expected_edge_probability_matrix = None
    if network_config.get("type") == "distance_decay":
        i_idx, j_idx = np.triu_indices(n_units, k=1)
        scale = np.median(distance_true[i_idx, j_idx])
        scale = float(scale) if np.isfinite(scale) and scale > 0 else 1.0
        score = float(network_config.get("intercept", 0.0)) - float(network_config.get("decay", 3.0)) * distance_true / scale
        expected_edge_probability_matrix = sigmoid(score)
        np.fill_diagonal(expected_edge_probability_matrix, 0.0)
    elif network_config.get("type") == "covariate_homophily":
        age = unit_df["age"].to_numpy(dtype=float)
        log_income = np.log(np.maximum(unit_df["income"].to_numpy(dtype=float), 1.0))
        gender = unit_df["gender"].to_numpy()
        education = unit_df["education"].to_numpy()
        race = unit_df["race"].to_numpy()
        age_sim = -np.abs(age[:, None] - age[None, :]) / 20.0
        inc_sim = -np.abs(log_income[:, None] - log_income[None, :])
        same_gender = (gender[:, None] == gender[None, :]).astype(float)
        same_educ = (education[:, None] == education[None, :]).astype(float)
        same_race = (race[:, None] == race[None, :]).astype(float)
        score = float(network_config.get("intercept", -3.0)) + float(network_config.get("similarity_scale", 0.8)) * (
            age_sim + inc_sim + same_gender + same_educ + same_race
        )
        expected_edge_probability_matrix = sigmoid(score)
        np.fill_diagonal(expected_edge_probability_matrix, 0.0)
    elif network_config.get("type") == "sbm":
        blocks = np.asarray(network_extra.get("blocks"), dtype=int)
        same_block = blocks[:, None] == blocks[None, :]
        expected_edge_probability_matrix = np.where(
            same_block,
            float(network_config.get("p_in", 0.2)),
            float(network_config.get("p_out", 0.05)),
        ).astype(float)
        np.fill_diagonal(expected_edge_probability_matrix, 0.0)
    elif network_config.get("type") == "central_nodes":
        hub_mask = adjacency.sum(axis=1) >= n_units - 1
        expected_edge_probability_matrix = np.full((n_units, n_units), float(network_config.get("p", 0.02)), dtype=float)
        expected_edge_probability_matrix[hub_mask, :] = 1.0
        expected_edge_probability_matrix[:, hub_mask] = 1.0
        np.fill_diagonal(expected_edge_probability_matrix, 0.0)

    spillover, spill_pieces = compute_spillover(
        adjacency=adjacency,
        distance_matrix=distance_true,
        treatment=treatment,
        r_true=r_true,
        spillover_type=config.get("spillover_type", "friend_average"),
        expected_edge_probability=config.get("network", {}).get("p"),
        expected_edge_probability_matrix=expected_edge_probability_matrix,
        saturation_lambda=float(config.get("saturation_lambda", 0.10)),
    )

    x_design, beta_names = build_design_matrix(unit_df)
    params = config.get("outcome", {})
    beta = beta_vector(beta_names, params.get("beta_true", {}))
    alpha = float(params.get("alpha_true", 0.0))
    tau = float(params.get("tau_true", 1.0))
    gamma = float(params.get("gamma_true", 100.0))
    sigma2 = float(params.get("sigma2", 1.0))
    epsilon = rng.normal(0.0, np.sqrt(max(sigma2, 0.0)), size=n_units)
    y_mean = alpha + x_design @ beta + tau * treatment + gamma * spillover
    y = y_mean + epsilon

    unit_df["W"] = treatment.astype(int)
    unit_df["S_true"] = spillover
    unit_df["Y_mean"] = y_mean
    unit_df["epsilon"] = epsilon
    unit_df["Y"] = y
    for name, values in spill_pieces.items():
        unit_df[name] = values

    metadata = {
        "seed": int(seed),
        "n_units": n_units,
        "r_true": r_true,
        "spillover_type": config.get("spillover_type", "friend_average"),
        "treatment_design": treatment_design,
        "outcome": {
            "alpha_true": alpha,
            "tau_true": tau,
            "gamma_true": gamma,
            "sigma2": sigma2,
            "beta_names": beta_names,
            "beta_values": beta.tolist(),
        },
        "distance": config.get("distance", {"type": "uniform_square"}),
        "distance_noise_sd": distance_noise_sd,
        "network": {**config.get("network", {"type": "erdos_renyi", "p": 0.2}), **network_extra},
        "graph_diagnostics": graph_diagnostics(adjacency, spill_pieces["qualifying_friend_count"]),
        "corr_spillover_outcome": float(np.corrcoef(spillover, y)[0, 1]) if np.std(spillover) > 0 else None,
    }
    return GeneratedData(unit_df, distance_true, distance_observed, adjacency, metadata)

