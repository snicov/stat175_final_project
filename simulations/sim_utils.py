#!/usr/bin/env python3
"""Shared utilities for the spillover simulation study."""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def timestamp() -> str:
    """Return a filesystem-safe timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON-compatible YAML config file.

    The project configs are written as JSON-compatible YAML, so they can be
    parsed without adding a PyYAML dependency.
    """
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON with NumPy-safe conversions."""

    def convert(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(inner) for key, inner in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    Path(path).write_text(json.dumps(convert(payload), indent=2), encoding="utf-8")


def copy_config(config_path: str | Path, output_dir: str | Path) -> None:
    """Copy a config file into a run output folder."""
    shutil.copy2(config_path, Path(output_dir) / "config.yaml")


def radius_grid(min_radius: float, max_radius: float, step: float) -> np.ndarray:
    """Build a sorted positive radius grid."""
    values = np.arange(float(min_radius), float(max_radius) + 0.5 * float(step), float(step))
    values = np.unique(np.round(values, 10))
    values = values[values > 0]
    if values.size == 0:
        raise ValueError("Radius grid is empty.")
    return values


def compute_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.sqrt(dx * dx + dy * dy)


def distance_matrix_to_long(distance_matrix: np.ndarray) -> pd.DataFrame:
    """Convert a dense distance matrix to upper-triangle long form."""
    i_idx, j_idx = np.triu_indices(distance_matrix.shape[0], k=1)
    return pd.DataFrame(
        {
            "unit_i": i_idx.astype(int),
            "unit_j": j_idx.astype(int),
            "distance": distance_matrix[i_idx, j_idx].astype(float),
        }
    )


def apply_distance_noise(
    distance_matrix: np.ndarray,
    noise_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add symmetric Gaussian noise to an observed distance matrix."""
    if noise_sd <= 0:
        return distance_matrix.copy()
    n = distance_matrix.shape[0]
    noise = rng.normal(0.0, noise_sd, size=(n, n))
    noise = (noise + noise.T) / 2.0
    noisy = np.maximum(distance_matrix + noise, 0.0)
    np.fill_diagonal(noisy, 0.0)
    return noisy


def bernoulli_treatment(n_units: int, pi: float, rng: np.random.Generator) -> np.ndarray:
    """Draw independent Bernoulli treatment assignment."""
    if not (0.0 <= pi <= 1.0):
        raise ValueError("Treatment probability pi must be in [0, 1].")
    return rng.binomial(1, pi, size=n_units).astype(int)


def placebo_assignments(
    n_draws: int,
    n_units: int,
    design: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw placebo assignments from the known experimental design."""
    design_type = design.get("type", "bernoulli")
    if design_type != "bernoulli":
        raise NotImplementedError(f"Unsupported treatment design: {design_type}")
    pi = float(design.get("pi", 0.5))
    return rng.binomial(1, pi, size=(int(n_draws), int(n_units))).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def make_symmetric_adjacency_from_upper(upper_values: np.ndarray, n_units: int) -> np.ndarray:
    """Build symmetric zero-diagonal adjacency from upper-triangle values."""
    adjacency = np.zeros((n_units, n_units), dtype=int)
    i_idx, j_idx = np.triu_indices(n_units, k=1)
    adjacency[i_idx, j_idx] = upper_values.astype(int)
    adjacency[j_idx, i_idx] = upper_values.astype(int)
    return adjacency


def erdos_renyi_adjacency(n_units: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Generate an undirected ER graph with independent upper-triangle edges."""
    if not (0.0 <= p <= 1.0):
        raise ValueError("ER probability p must be in [0, 1].")
    i_idx, _ = np.triu_indices(n_units, k=1)
    upper = rng.binomial(1, p, size=i_idx.size)
    return make_symmetric_adjacency_from_upper(upper, n_units)


def covariate_homophily_adjacency(
    covariates: pd.DataFrame,
    intercept: float,
    similarity_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate latent friendships with probability increasing in covariate similarity."""
    n = len(covariates)
    age = covariates["age"].to_numpy(dtype=float)
    log_income = np.log(np.maximum(covariates["income"].to_numpy(dtype=float), 1.0))
    gender = covariates["gender"].to_numpy()
    education = covariates["education"].to_numpy()
    race = covariates["race"].to_numpy()
    i_idx, j_idx = np.triu_indices(n, k=1)
    age_sim = -np.abs(age[i_idx] - age[j_idx]) / 20.0
    inc_sim = -np.abs(log_income[i_idx] - log_income[j_idx])
    same_gender = (gender[i_idx] == gender[j_idx]).astype(float)
    same_educ = (education[i_idx] == education[j_idx]).astype(float)
    same_race = (race[i_idx] == race[j_idx]).astype(float)
    score = intercept + similarity_scale * (age_sim + inc_sim + same_gender + same_educ + same_race)
    upper = rng.binomial(1, sigmoid(score))
    return make_symmetric_adjacency_from_upper(upper, n)


def distance_decay_adjacency(
    distance_matrix: np.ndarray,
    intercept: float,
    decay: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate latent friendships with probability decreasing in distance."""
    n = distance_matrix.shape[0]
    i_idx, j_idx = np.triu_indices(n, k=1)
    scale = np.median(distance_matrix[i_idx, j_idx])
    scale = float(scale) if np.isfinite(scale) and scale > 0 else 1.0
    score = intercept - decay * distance_matrix[i_idx, j_idx] / scale
    upper = rng.binomial(1, sigmoid(score))
    return make_symmetric_adjacency_from_upper(upper, n)


def central_nodes_adjacency(n_units: int, p: float, k: int, rng: np.random.Generator) -> np.ndarray:
    """Generate ER graph with k random hubs connected to everyone."""
    adjacency = erdos_renyi_adjacency(n_units, p, rng)
    k = min(max(int(k), 0), n_units)
    if k > 0:
        hubs = rng.choice(n_units, size=k, replace=False)
        adjacency[hubs, :] = 1
        adjacency[:, hubs] = 1
        np.fill_diagonal(adjacency, 0)
    return adjacency


def sbm_adjacency(
    n_units: int,
    n_blocks: int,
    p_in: float,
    p_out: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple stochastic block model graph."""
    blocks = np.arange(n_units) % int(n_blocks)
    rng.shuffle(blocks)
    i_idx, j_idx = np.triu_indices(n_units, k=1)
    probs = np.where(blocks[i_idx] == blocks[j_idx], p_in, p_out)
    upper = rng.binomial(1, probs)
    return make_symmetric_adjacency_from_upper(upper, n_units), blocks


def graph_diagnostics(adjacency: np.ndarray, qualifying_friend_count: np.ndarray | None = None) -> dict[str, Any]:
    """Compute graph diagnostics."""
    degrees = adjacency.sum(axis=1)
    out: dict[str, Any] = {
        "edge_count": int(np.triu(adjacency, k=1).sum()),
        "mean_degree": float(degrees.mean()),
        "median_degree": float(np.median(degrees)),
        "min_degree": int(degrees.min()) if degrees.size else 0,
        "max_degree": int(degrees.max()) if degrees.size else 0,
    }
    if qualifying_friend_count is not None:
        out.update(
            {
                "mean_friends_within_r_true": float(np.mean(qualifying_friend_count)),
                "median_friends_within_r_true": float(np.median(qualifying_friend_count)),
                "share_zero_friends_within_r_true": float(np.mean(qualifying_friend_count == 0)),
            }
        )
    return out


def plot_base_diagnostics(base_dir: str | Path, r_true: float) -> None:
    """Save pairwise distance and friend-count diagnostic plots."""
    base_dir = ensure_dir(base_dir)
    pair_path = base_dir / "pair_distances.csv"
    unit_path = base_dir / "unit_data.csv"
    if pair_path.exists():
        distances = pd.read_csv(pair_path, usecols=["distance"])["distance"].to_numpy(dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(distances, bins=80, color="steelblue", edgecolor="white")
        axes[0].axvline(float(np.mean(distances)), color="darkred", linestyle="--", label=f"mean={np.mean(distances):.1f}")
        axes[0].axvline(float(np.median(distances)), color="orange", linestyle=":", label=f"median={np.median(distances):.1f}")
        axes[0].set_xlabel("Distance")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Pairwise distances")
        axes[0].legend()
        axes[1].hist(distances, bins=80, density=True, color="steelblue", edgecolor="white")
        axes[1].set_xlabel("Distance")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Pairwise distance density")
        fig.tight_layout()
        fig.savefig(base_dir / "pairwise_distance_distribution.png", dpi=160)
        plt.close(fig)
    if unit_path.exists():
        unit_df = pd.read_csv(unit_path)
        if "qualifying_friend_count" in unit_df:
            counts = unit_df["qualifying_friend_count"].to_numpy(dtype=int)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            bins = np.arange(-0.5, max(int(counts.max()) if counts.size else 1, 1) + 1.5)
            ax.hist(counts, bins=bins, color="seagreen", edgecolor="white")
            ax.axvline(float(np.mean(counts)), color="darkred", linestyle="--", label=f"mean={np.mean(counts):.2f}")
            ax.axvline(float(np.median(counts)), color="orange", linestyle=":", label=f"median={np.median(counts):.0f}")
            ax.set_xlabel(f"Friends within r_true={r_true:g}")
            ax.set_ylabel("Units")
            ax.set_title("Friends within true radius")
            ax.legend()
            fig.tight_layout()
            fig.savefig(base_dir / "friends_within_r_true_distribution.png", dpi=160)
            plt.close(fig)


def finite_or_none(value: float) -> float | None:
    """Convert non-finite values to None for JSON outputs."""
    return float(value) if math.isfinite(float(value)) else None

