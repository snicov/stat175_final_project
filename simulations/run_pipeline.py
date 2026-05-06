#!/usr/bin/env python3
"""Generic config-driven runner for spillover simulation scenarios."""

from __future__ import annotations

import argparse
import copy
import json
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dgp import generate_dataset
from exposure_mapping import estimate_radius
from sim_utils import (
    copy_config,
    distance_matrix_to_long,
    ensure_dir,
    plot_base_diagnostics,
    read_config,
    timestamp,
    write_json,
)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Return a recursively updated copy of base."""
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def set_by_path(config: dict[str, Any], dotted_path: str, value: Any) -> dict[str, Any]:
    """Set a nested config value by dot-separated path."""
    out = copy.deepcopy(config)
    cursor = out
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
    return out


def grid_configs(config: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Expand scenario sweep grid into concrete configs and labels."""
    sweep = config.get("sweep", [])
    if not sweep:
        return [(config, {})]
    paths = [item["path"] for item in sweep]
    values = [item["values"] for item in sweep]
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for combo in product(*values):
        cfg = dict(config)
        label: dict[str, Any] = {}
        for path, value in zip(paths, combo):
            cfg = set_by_path(cfg, path, value)
            label[path] = value
        out.append((cfg, label))
    return out


def add_estimator_grids(config: dict[str, Any]) -> dict[str, Any]:
    """Compute radius and band grids in a config."""
    est = dict(config.get("estimator", {}))
    radius_min = float(est.get("radius_min", 4.0))
    radius_max = float(est.get("radius_max", 50.0))
    radius_step = float(est.get("radius_step", 2.0))
    band_min = float(est.get("band_radius_min", 2.0))
    band_max = float(est.get("band_radius_max", radius_max))
    band_step = float(est.get("band_radius_step", 2.0))
    est["radius_grid"] = list(np.arange(radius_min, radius_max + 0.5 * radius_step, radius_step))
    est["band_edges"] = list(np.arange(band_min, band_max + 0.5 * band_step, band_step))
    est.setdefault("B", 50)
    est.setdefault("psi_dictionary", "annulus_treated_share")
    cfg = dict(config)
    cfg["estimator"] = est
    return cfg


def safe_label(label: dict[str, Any]) -> str:
    """Create a filesystem-safe grid label."""
    if not label:
        return "base"
    pieces = []
    for key, value in label.items():
        short = key.replace(".", "_")
        val = f"{value:g}" if isinstance(value, (float, int)) else str(value)
        pieces.append(f"{short}_{val}".replace(".", "p"))
    return "__".join(pieces)


def write_dataset_outputs(run_dir: Path, generated, config: dict[str, Any]) -> None:
    """Write generated data and diagnostics for one seed."""
    data_dir = ensure_dir(run_dir / "data")
    generated.unit_df.to_csv(data_dir / "unit_data.csv", index=False)
    distance_matrix_to_long(generated.distance_true).to_csv(data_dir / "pair_distances_true.csv", index=False)
    distance_matrix_to_long(generated.distance_observed).to_csv(data_dir / "pair_distances.csv", index=False)
    edge_i, edge_j = np.triu_indices(generated.adjacency.shape[0], k=1)
    edge_mask = generated.adjacency[edge_i, edge_j] == 1
    pd.DataFrame({"unit_i": edge_i[edge_mask], "unit_j": edge_j[edge_mask]}).to_csv(
        data_dir / "latent_friend_edges.csv", index=False
    )
    write_json(data_dir / "metadata.json", generated.metadata)
    plot_base_diagnostics(data_dir, r_true=float(config["r_true"]))


def summarize_group(rows: pd.DataFrame, sweep_columns: list[str]) -> pd.DataFrame:
    """Aggregate per-seed results by sweep point."""
    group_cols = sweep_columns or ["scenario_name"]
    grouped = rows.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        mean_r_hat=("r_hat", "mean"),
        median_r_hat=("r_hat", "median"),
        empirical_se_r_hat=("r_hat", "std"),
        mean_analytic_se=("r_hat_se", "mean"),
        mean_abs_error=("abs_error_r_hat", "mean"),
        median_abs_error=("abs_error_r_hat", "median"),
        mean_corr_spill_y=("corr_true_spill_Y", "mean"),
        n_runs=("r_hat", "count"),
    ).reset_index()
    return summary


def plot_aggregate(summary: pd.DataFrame, output_dir: Path, sweep_columns: list[str], r_true: float) -> None:
    """Write aggregate plots for one scenario."""
    plots_dir = ensure_dir(output_dir / "plots")
    if len(sweep_columns) != 1:
        return
    xcol = sweep_columns[0]
    x = summary[xcol]

    fig, ax = plt.subplots(figsize=(8, 5))
    yerr = 1.96 * summary["empirical_se_r_hat"].fillna(0.0)
    ax.errorbar(x, summary["mean_r_hat"], yerr=yerr, marker="o", capsize=3)
    ax.axhline(r_true, color="tab:green", linestyle="--", label="r_true")
    ax.set_xlabel(xcol)
    ax.set_ylabel("Mean r_hat")
    ax.set_title("Mean estimated radius across seeds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "mean_r_hat_vs_sweep.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, summary["mean_abs_error"], marker="o")
    ax.set_xlabel(xcol)
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Mean absolute radius error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "mean_abs_error_vs_sweep.png", dpi=160)
    plt.close(fig)


def run_config(config_path: str | Path, output_dir: str | Path | None = None, max_seeds: int | None = None) -> Path:
    """Run one scenario config and return the output directory."""
    config_path = Path(config_path)
    config = read_config(config_path)
    scenario_name = config["scenario_name"]
    scenario_dir = config_path.parent
    run_dir = ensure_dir(output_dir or (scenario_dir / "output" / f"run_{timestamp()}"))
    copy_config(config_path, run_dir)
    write_json(run_dir / "resolved_config.json", config)

    seeds = list(config.get("seeds", list(range(20))))
    if max_seeds is not None:
        seeds = seeds[: int(max_seeds)]
    concrete = grid_configs(config)
    sweep_columns = [item["path"] for item in config.get("sweep", [])]
    all_rows: list[dict[str, Any]] = []

    for grid_idx, (raw_grid_cfg, label) in enumerate(concrete):
        grid_cfg = add_estimator_grids(raw_grid_cfg)
        grid_label = safe_label(label)
        for seed in seeds:
            seed_dir = ensure_dir(run_dir / "grid" / grid_label / f"seed_{int(seed):04d}")
            selection_path = seed_dir / "estimation" / "radius_selection.json"
            metadata_path = seed_dir / "data" / "metadata.json"
            if selection_path.exists() and metadata_path.exists():
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                row = {
                    "scenario_name": scenario_name,
                    "grid_index": grid_idx,
                    "seed": int(seed),
                    "r_true": float(raw_grid_cfg["r_true"]),
                    "r_hat": float(selection["r_hat"]),
                    "r_hat_se": float(selection["r_hat_se"]),
                    "wald_ci_low": selection["wald_ci_95"][0],
                    "wald_ci_high": selection["wald_ci_95"][1],
                    "abs_error_r_hat": abs(float(selection["r_hat"]) - float(raw_grid_cfg["r_true"])),
                    "second_step_criterion_min": float(selection["second_step_criterion_min"]),
                    "first_step_criterion_min": float(selection["first_step_criterion_min"]),
                    "selected_radius_j_pvalue": float(selection["selected_radius_j_pvalue"]),
                    "corr_true_spill_Y": metadata.get("corr_spillover_outcome"),
                    "B": int(raw_grid_cfg.get("estimator", {}).get("B", 50)),
                }
                for key, value in label.items():
                    row[key] = value
                all_rows.append(row)
                continue
            generated = generate_dataset(grid_cfg, int(seed))
            write_dataset_outputs(seed_dir, generated, grid_cfg)
            est_dir = seed_dir / "estimation"
            result = estimate_radius(
                unit_df=generated.unit_df,
                distance_observed=generated.distance_observed,
                config=grid_cfg,
                rng=np.random.default_rng(int(seed) + 100_000),
                output_dir=est_dir,
                save_outputs=True,
            )
            row = {
                "scenario_name": scenario_name,
                "grid_index": grid_idx,
                "seed": int(seed),
                "r_true": float(grid_cfg["r_true"]),
                "r_hat": float(result.selection["r_hat"]),
                "r_hat_se": float(result.selection["r_hat_se"]),
                "wald_ci_low": result.selection["wald_ci_95"][0],
                "wald_ci_high": result.selection["wald_ci_95"][1],
                "abs_error_r_hat": abs(float(result.selection["r_hat"]) - float(grid_cfg["r_true"])),
                "second_step_criterion_min": float(result.selection["second_step_criterion_min"]),
                "first_step_criterion_min": float(result.selection["first_step_criterion_min"]),
                "selected_radius_j_pvalue": float(result.selection["selected_radius_j_pvalue"]),
                "corr_true_spill_Y": generated.metadata.get("corr_spillover_outcome"),
                "B": int(grid_cfg["estimator"].get("B", 50)),
            }
            for key, value in label.items():
                row[key] = value
            all_rows.append(row)

    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(run_dir / "per_seed_results.csv", index=False)
    summary = summarize_group(per_seed, sweep_columns)
    summary.to_csv(run_dir / "aggregate_results.csv", index=False)
    write_json(run_dir / "aggregate_results.json", summary.to_dict(orient="records"))
    plot_aggregate(summary, run_dir, sweep_columns, r_true=float(config.get("r_true", 30.0)))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a config-driven spillover simulation scenario.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_seeds", type=int, default=None, help="Optional limit for smoke runs.")
    args = parser.parse_args()
    out = run_config(args.config, args.output_dir, args.max_seeds)
    print(f"Scenario complete. Output directory: {out}")


if __name__ == "__main__":
    main()

