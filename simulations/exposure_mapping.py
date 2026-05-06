#!/usr/bin/env python3
"""Paper-aligned exposure-mapping estimator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from sim_utils import ensure_dir, placebo_assignments, write_json


@dataclass
class EstimationResult:
    """Result bundle for one radius-estimation run."""

    objective_df: pd.DataFrame
    moment_df: pd.DataFrame
    selection: dict[str, Any]


def exposure_from_masks(assignments: np.ndarray, masks: list[np.ndarray], mode: str = "share") -> np.ndarray:
    """Compute exposure for assignments and boolean masks.

    assignments can be shape (n,) or (B, n). Return shape is (n,) or (B, n).
    `mode="share"` returns treated share; `mode="count"` returns treated count.
    """
    if mode not in {"share", "count"}:
        raise ValueError("mode must be 'share' or 'count'.")
    w = np.asarray(assignments, dtype=float)
    was_1d = w.ndim == 1
    if was_1d:
        w = w[None, :]
    out = np.zeros((w.shape[0], len(masks)), dtype=float)
    for i, mask in enumerate(masks):
        if mode == "count":
            out[:, i] = w[:, mask].sum(axis=1)
            continue
        denom = int(mask.sum())
        if denom > 0:
            out[:, i] = w[:, mask].mean(axis=1)
    return out[0] if was_1d else out


def radius_masks(distance_matrix: np.ndarray, radii: np.ndarray) -> dict[float, list[np.ndarray]]:
    """Build neighbor masks for cumulative radius exposure maps."""
    n = distance_matrix.shape[0]
    idx = np.arange(n)
    return {
        float(r): [(distance_matrix[i] <= float(r)) & (idx != i) for i in range(n)]
        for r in radii
    }


def outside_radius_masks(distance_matrix: np.ndarray, radii: np.ndarray) -> dict[float, list[np.ndarray]]:
    """Build neighbor masks for outside-radius exposure maps."""
    n = distance_matrix.shape[0]
    idx = np.arange(n)
    return {
        float(r): [(distance_matrix[i] > float(r)) & (idx != i) for i in range(n)]
        for r in radii
    }


def local_outer_shell_masks(distance_matrix: np.ndarray, radius: float, shell_width: float) -> tuple[list[str], list[list[np.ndarray]]]:
    """Build a one-shell psi dictionary for (r, r + shell_width]."""
    n = distance_matrix.shape[0]
    idx = np.arange(n)
    upper = float(radius) + float(shell_width)
    masks = [
        (distance_matrix[i] > float(radius)) & (distance_matrix[i] <= upper) & (idx != i)
        for i in range(n)
    ]
    return [f"outer_shell_{float(radius):g}_{upper:g}"], [masks]


def annulus_masks(distance_matrix: np.ndarray, band_edges: np.ndarray) -> tuple[list[str], list[list[np.ndarray]]]:
    """Build annulus masks for psi dictionary."""
    n = distance_matrix.shape[0]
    idx = np.arange(n)
    names: list[str] = []
    masks_by_band: list[list[np.ndarray]] = []
    lower = 0.0
    for upper in band_edges:
        names.append(f"band_{lower:g}_{float(upper):g}")
        masks_by_band.append(
            [
                (distance_matrix[i] > lower) & (distance_matrix[i] <= float(upper)) & (idx != i)
                for i in range(n)
            ]
        )
        lower = float(upper)
    return names, masks_by_band


def cumulative_psi_masks(distance_matrix: np.ndarray, psi_radii: np.ndarray) -> tuple[list[str], list[list[np.ndarray]]]:
    """Build cumulative treated-count psi masks for a grid of radii."""
    masks_by_radius = radius_masks(distance_matrix, psi_radii)
    names = [f"cumulative_0_{float(r):g}" for r in psi_radii]
    return names, [masks_by_radius[float(r)] for r in psi_radii]


def compute_psi_matrix(assignments: np.ndarray, band_masks: list[list[np.ndarray]], mode: str = "share") -> np.ndarray:
    """Compute psi_m(W) for each annulus band.

    Returns (n, m) for one assignment or (B, n, m) for placebo assignments.
    """
    w = np.asarray(assignments, dtype=float)
    was_1d = w.ndim == 1
    if was_1d:
        w = w[None, :]
    psi = np.zeros((w.shape[0], len(band_masks[0]), len(band_masks)), dtype=float)
    for m, masks in enumerate(band_masks):
        psi[:, :, m] = exposure_from_masks(w, masks, mode=mode)
    return psi[0] if was_1d else psi


def exact_bernoulli_count_conditional_mean(
    g_obs: np.ndarray,
    inside_masks: list[np.ndarray],
    band_masks: list[list[np.ndarray]],
    pi: float,
) -> np.ndarray:
    """Compute exact E[annulus count | inside count] under Bernoulli assignment.

    For unit i, candidate inside set I_i(r), and band set M_i:

        E[C_M | C_I = g] = g * |M cap I| / |I| + pi * |M \\ I|.

    This is exact for independent Bernoulli assignment because, conditional on
    the total treated count inside I, treated units are uniformly distributed
    within I and treatment outside I remains independent Bernoulli(pi).
    """
    n_units = len(inside_masks)
    n_moments = len(band_masks)
    out = np.zeros((n_units, n_moments), dtype=float)
    for i, inside in enumerate(inside_masks):
        inside_count = int(inside.sum())
        g_i = float(g_obs[i])
        for m, masks in enumerate(band_masks):
            band = masks[i]
            overlap = int(np.logical_and(band, inside).sum())
            outside_band = int(np.logical_and(band, ~inside).sum())
            inside_component = g_i * overlap / inside_count if inside_count > 0 else 0.0
            out[i, m] = inside_component + float(pi) * outside_band
    return out


def silverman_bandwidth(values: np.ndarray) -> float:
    """Robust Silverman bandwidth for one-dimensional smoothing."""
    if values.size <= 1:
        return 1e-3
    std = float(np.std(values, ddof=1))
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    scale = min(std, iqr / 1.34) if iqr > 0 else std
    if not np.isfinite(scale) or scale <= 0:
        return 1e-3
    return max(0.9 * scale * values.size ** (-1.0 / 5.0), 1e-3)


def estimate_conditional_psi(
    w_obs: np.ndarray,
    g_obs: np.ndarray,
    w_placebo: np.ndarray,
    g_placebo: np.ndarray,
    psi_placebo: np.ndarray,
) -> np.ndarray:
    """Estimate E[psi | W_i, g_i] using placebo draws only."""
    if g_obs.ndim == 1:
        g_obs_work = g_obs[:, None]
    else:
        g_obs_work = g_obs
    if g_placebo.ndim == 2:
        g_placebo_work = g_placebo[:, :, None]
    else:
        g_placebo_work = g_placebo
    b_count, n_units, g_dim = g_placebo_work.shape
    n_moments = psi_placebo.shape[2]
    out = np.zeros((n_units, n_moments), dtype=float)
    for i in range(n_units):
        own_mask = w_placebo[:, i] == w_obs[i]
        if not np.any(own_mask):
            own_mask = np.ones(b_count, dtype=bool)
        g_samples = g_placebo_work[own_mask, i, :]
        psi_samples = psi_placebo[own_mask, i, :]
        exact = np.all(np.isclose(g_samples, g_obs_work[i, :], atol=1e-12, rtol=0.0), axis=1)
        if np.any(exact):
            out[i, :] = psi_samples[exact].mean(axis=0)
            continue
        bandwidths = np.array([silverman_bandwidth(g_samples[:, dim]) for dim in range(g_dim)])
        z = (g_samples - g_obs_work[i, :]) / bandwidths
        weights = np.exp(-0.5 * np.sum(z * z, axis=1))
        if float(weights.sum()) <= 1e-12:
            distances = np.sum(z * z, axis=1)
            out[i, :] = psi_samples[int(np.argmin(distances)), :]
        else:
            out[i, :] = (weights / weights.sum()) @ psi_samples
    return out


def conley_covariance(scores: np.ndarray, distance_matrix: np.ndarray, bandwidth: float) -> np.ndarray:
    """Simple spatial HAC covariance with triangular kernel."""
    n, m = scores.shape
    omega = np.zeros((m, m), dtype=float)
    for i in range(n):
        weights = np.maximum(0.0, 1.0 - distance_matrix[i] / bandwidth)
        active = np.flatnonzero(weights > 0)
        for j in active:
            omega += weights[j] * np.outer(scores[i], scores[j])
    return omega / n


def ridge_inverse(matrix: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Compute a stable pseudoinverse with a ridge."""
    return np.linalg.pinv(matrix + ridge * np.eye(matrix.shape[0]))


def estimate_radius(
    unit_df: pd.DataFrame,
    distance_observed: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
    output_dir: str | Path | None = None,
    save_outputs: bool = True,
) -> EstimationResult:
    """Estimate the spillover radius using design-based exposure mapping moments."""
    estimator_cfg = config.get("estimator", {})
    if estimator_cfg.get("estimator_type") == "dose_response_sufficiency":
        return estimate_radius_dose_response(unit_df, distance_observed, config, output_dir, save_outputs)
    radii = np.asarray(estimator_cfg.get("radius_grid"), dtype=float)
    band_edges = np.asarray(estimator_cfg.get("band_edges", radii), dtype=float)
    b_placebo = int(estimator_cfg.get("B", 50))
    treatment_design = config.get("treatment", {"type": "bernoulli", "pi": 0.5})
    exposure_map = estimator_cfg.get("exposure_map", "treated_share")
    psi_dictionary = estimator_cfg.get("psi_dictionary", "annulus_treated_share")
    conditional_method = estimator_cfg.get("conditional_expectation", "placebo")
    exposure_mode = "count" if exposure_map == "treated_count" else "share"
    psi_mode = "count" if psi_dictionary in {"annulus_treated_count", "cumulative_treated_count"} else "share"
    shell_width = float(estimator_cfg.get("shell_width", estimator_cfg.get("band_radius_step", 2.0)))

    y = unit_df["Y"].to_numpy(dtype=float)  # phi_m(Y_i)=Y_i
    w_obs = unit_df["W"].to_numpy(dtype=float)
    n_units = len(unit_df)

    cumulative_masks = radius_masks(distance_observed, radii)
    outside_masks = outside_radius_masks(distance_observed, radii)
    if psi_dictionary == "cumulative_treated_count":
        fixed_band_names, fixed_band_masks = cumulative_psi_masks(distance_observed, band_edges)
    else:
        fixed_band_names, fixed_band_masks = annulus_masks(distance_observed, band_edges)
    use_local_shell = psi_dictionary == "local_outer_shell_treated_count"
    use_exact_bernoulli_count = (
        conditional_method == "exact_bernoulli_count"
        and exposure_map == "treated_count"
        and psi_dictionary in {"annulus_treated_count", "cumulative_treated_count"}
        and treatment_design.get("type", "bernoulli") == "bernoulli"
    )
    if use_local_shell:
        band_names = ["outer_shell"]
        psi_obs_fixed = None
    else:
        band_names = fixed_band_names
        psi_obs_fixed = compute_psi_matrix(w_obs, fixed_band_masks, mode=psi_mode)

    if use_exact_bernoulli_count:
        w_placebo = np.empty((0, n_units), dtype=float)
        psi_placebo_fixed = None
    else:
        w_placebo = placebo_assignments(b_placebo, n_units, treatment_design, rng)
        psi_placebo_fixed = None if use_local_shell else compute_psi_matrix(w_placebo, fixed_band_masks, mode=psi_mode)

    radius_results: dict[float, dict[str, Any]] = {}
    covariance_bandwidth = float(estimator_cfg.get("covariance_bandwidth", np.max(radii)))
    for r in radii:
        if exposure_map == "inside_outside_treated_count":
            inside_obs = exposure_from_masks(w_obs, cumulative_masks[float(r)], mode="count")
            outside_obs = exposure_from_masks(w_obs, outside_masks[float(r)], mode="count")
            inside_placebo = exposure_from_masks(w_placebo, cumulative_masks[float(r)], mode="count")
            outside_placebo = exposure_from_masks(w_placebo, outside_masks[float(r)], mode="count")
            g_obs = np.column_stack([inside_obs, outside_obs])
            g_placebo = np.stack([inside_placebo, outside_placebo], axis=2)
        else:
            g_obs = exposure_from_masks(w_obs, cumulative_masks[float(r)], mode=exposure_mode)
            g_placebo = (
                np.empty((0, n_units), dtype=float)
                if use_exact_bernoulli_count
                else exposure_from_masks(w_placebo, cumulative_masks[float(r)], mode=exposure_mode)
            )
        if use_local_shell:
            _, shell_masks = local_outer_shell_masks(distance_observed, float(r), shell_width)
            psi_obs = compute_psi_matrix(w_obs, shell_masks, mode="count")
            psi_placebo = compute_psi_matrix(w_placebo, shell_masks, mode="count")
        else:
            psi_obs = psi_obs_fixed
            psi_placebo = psi_placebo_fixed
        if use_exact_bernoulli_count:
            cond_mean = exact_bernoulli_count_conditional_mean(
                g_obs=g_obs,
                inside_masks=cumulative_masks[float(r)],
                band_masks=fixed_band_masks,
                pi=float(treatment_design.get("pi", 0.5)),
            )
        else:
            cond_mean = estimate_conditional_psi(w_obs, g_obs, w_placebo, g_placebo, psi_placebo)
        residuals = psi_obs - cond_mean
        scores = y[:, None] * residuals
        moment_mean = scores.mean(axis=0)
        omega = conley_covariance(scores, distance_observed, covariance_bandwidth)
        radius_results[float(r)] = {
            "moment_mean": moment_mean,
            "omega": omega,
            "omega_diag": np.diag(omega),
            "scores": scores,
        }

    identity = np.eye(len(band_names))
    first_rows = []
    for r in radii:
        m = radius_results[float(r)]["moment_mean"]
        first_rows.append({"radius": float(r), "first_step_criterion": float(n_units * m.T @ identity @ m)})
    first_df = pd.DataFrame(first_rows)
    r_first = float(first_df.loc[int(first_df["first_step_criterion"].idxmin()), "radius"])
    optimal_weight = ridge_inverse(radius_results[r_first]["omega"])

    objective_rows = []
    moment_rows = []
    for r in radii:
        result = radius_results[float(r)]
        m = result["moment_mean"]
        q = float(n_units * m.T @ optimal_weight @ m)
        objective_rows.append(
            {
                "radius": float(r),
                "first_step_criterion": float(first_df.loc[np.isclose(first_df["radius"], float(r)), "first_step_criterion"].iloc[0]),
                "second_step_criterion": q,
                "j_df": max(len(band_names) - 1, 1),
                "j_pvalue": float(1.0 - chi2.cdf(q, df=max(len(band_names) - 1, 1))),
            }
        )
        row: dict[str, Any] = {"radius": float(r)}
        for name, value, var in zip(band_names, m, result["omega_diag"]):
            row[f"moment_{name}"] = float(value)
            row[f"omega_diag_{name}"] = float(var)
        moment_rows.append(row)

    objective_df = pd.DataFrame(objective_rows).sort_values("radius").reset_index(drop=True)
    moment_df = pd.DataFrame(moment_rows).sort_values("radius").reset_index(drop=True)
    min_idx = int(objective_df["second_step_criterion"].idxmin())
    r_hat = float(objective_df.loc[min_idx, "radius"])

    moment_cols = [f"moment_{name}" for name in band_names]
    selected = int(np.where(np.isclose(moment_df["radius"].to_numpy(dtype=float), r_hat))[0][0])
    if selected == 0:
        left, right = 0, 1
    elif selected == len(moment_df) - 1:
        left, right = len(moment_df) - 2, len(moment_df) - 1
    else:
        left, right = selected - 1, selected + 1
    delta = float(moment_df.loc[right, "radius"] - moment_df.loc[left, "radius"])
    gradient = (
        (moment_df.loc[right, moment_cols].to_numpy(dtype=float) - moment_df.loc[left, moment_cols].to_numpy(dtype=float)) / delta
        if delta > 0
        else np.zeros(len(moment_cols))
    )
    information = float(gradient.T @ optimal_weight @ gradient)
    r_hat_se = float(np.sqrt(1.0 / (n_units * information))) if information > 0 else float("nan")
    wald = [float(r_hat - 1.96 * r_hat_se), float(r_hat + 1.96 * r_hat_se)] if np.isfinite(r_hat_se) else [None, None]

    selection = {
        "r_true": config.get("r_true"),
        "r_first_step": r_first,
        "r_hat": r_hat,
        "r_hat_se": r_hat_se,
        "wald_ci_95": wald,
        "second_step_criterion_min": float(objective_df.loc[min_idx, "second_step_criterion"]),
        "first_step_criterion_min": float(first_df["first_step_criterion"].min()),
        "selected_radius_j_pvalue": float(objective_df.loc[min_idx, "j_pvalue"]),
        "B": b_placebo,
        "phi": "identity: phi_m(Y_i)=Y_i",
        "conditional_expectation": (
            "exact Bernoulli E[annulus treated count | inside treated count]"
            if use_exact_bernoulli_count
            else "placebo estimate of E[psi_m(W) | W_i, g_i(W;r)]"
        ),
        "exposure_map": exposure_map,
        "psi_dictionary": psi_dictionary,
        "band_names": band_names,
    }

    if save_outputs:
        if output_dir is None:
            raise ValueError("output_dir is required when save_outputs=True")
        output_dir = ensure_dir(output_dir)
        objective_df.to_csv(output_dir / "radius_objective.csv", index=False)
        moment_df.to_csv(output_dir / "moment_diagnostics.csv", index=False)
        write_json(output_dir / "radius_selection.json", selection)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(objective_df["radius"], objective_df["second_step_criterion"], marker="o")
        ax.axvline(r_hat, color="tab:red", linestyle="--", label=f"r_hat={r_hat:g}")
        if config.get("r_true") is not None:
            ax.axvline(float(config["r_true"]), color="tab:green", linestyle="--", label=f"r_true={float(config['r_true']):g}")
        ax.set_xlabel("Radius")
        ax.set_ylabel("Second-step criterion")
        ax.set_title("Exposure-mapping radius objective")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "objective_plot.png", dpi=160)
        plt.close(fig)

        lines = [
            "Exposure-mapping radius estimation summary",
            f"r_true: {config.get('r_true')}",
            f"r_first_step: {r_first}",
            f"r_hat: {r_hat}",
            f"r_hat_se: {r_hat_se}",
            f"Wald 95% CI: {wald}",
            "phi_m(Y_i)=Y_i",
            f"Exposure map: {exposure_map}",
            f"Psi dictionary: {psi_dictionary}",
            f"B placebo redraws: {b_placebo}",
            f"Conditional expectation: {selection['conditional_expectation']}",
        ]
        (output_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return EstimationResult(objective_df=objective_df, moment_df=moment_df, selection=selection)


def estimate_radius_dose_response(
    unit_df: pd.DataFrame,
    distance_observed: np.ndarray,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    save_outputs: bool = True,
) -> EstimationResult:
    """Select radius by outcome sufficiency: minimize residual MSE from Y on W and g_i(r)."""
    estimator_cfg = config.get("estimator", {})
    radii = np.asarray(estimator_cfg.get("radius_grid"), dtype=float)
    exposure_map = estimator_cfg.get("exposure_map", "treated_count")
    exposure_mode = "count" if exposure_map in {"treated_count", "inside_outside_treated_count"} else "share"
    y = unit_df["Y"].to_numpy(dtype=float)
    w_obs = unit_df["W"].to_numpy(dtype=float)
    n_units = len(unit_df)
    cumulative_masks = radius_masks(distance_observed, radii)

    rows = []
    coef_rows = []
    for r in radii:
        g = exposure_from_masks(w_obs, cumulative_masks[float(r)], mode=exposure_mode)
        design = np.column_stack([np.ones(n_units), w_obs, g])
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        mse = float(np.mean(residual * residual))
        rows.append(
            {
                "radius": float(r),
                "first_step_criterion": mse,
                "second_step_criterion": mse,
                "j_df": 0,
                "j_pvalue": np.nan,
            }
        )
        coef_rows.append(
            {
                "radius": float(r),
                "coef_intercept": float(beta[0]),
                "coef_treatment": float(beta[1]),
                "coef_exposure": float(beta[2]),
                "mse": mse,
            }
        )

    objective_df = pd.DataFrame(rows).sort_values("radius").reset_index(drop=True)
    moment_df = pd.DataFrame(coef_rows).sort_values("radius").reset_index(drop=True)
    min_idx = int(objective_df["second_step_criterion"].idxmin())
    r_hat = float(objective_df.loc[min_idx, "radius"])

    # Curvature-based heuristic SE for the discrete objective, mostly for plotting metadata.
    radii_arr = objective_df["radius"].to_numpy(dtype=float)
    q_arr = objective_df["second_step_criterion"].to_numpy(dtype=float)
    idx = min_idx
    if 0 < idx < len(radii_arr) - 1:
        h = float(radii_arr[idx + 1] - radii_arr[idx])
        curvature = float((q_arr[idx + 1] - 2 * q_arr[idx] + q_arr[idx - 1]) / (h * h))
        sigma2_hat = float(q_arr[idx])
        r_hat_se = float(np.sqrt(max(sigma2_hat, 0.0) / (n_units * curvature))) if curvature > 0 else float("nan")
    else:
        r_hat_se = float("nan")
    wald = [float(r_hat - 1.96 * r_hat_se), float(r_hat + 1.96 * r_hat_se)] if np.isfinite(r_hat_se) else [None, None]

    selection = {
        "r_true": config.get("r_true"),
        "r_first_step": r_hat,
        "r_hat": r_hat,
        "r_hat_se": r_hat_se,
        "wald_ci_95": wald,
        "second_step_criterion_min": float(objective_df.loc[min_idx, "second_step_criterion"]),
        "first_step_criterion_min": float(objective_df.loc[min_idx, "first_step_criterion"]),
        "selected_radius_j_pvalue": np.nan,
        "B": 0,
        "phi": "identity: phi_m(Y_i)=Y_i",
        "conditional_expectation": "not used; dose-response sufficiency criterion",
        "estimator_type": "dose_response_sufficiency",
        "exposure_map": exposure_map,
        "dose_response": "linear: Y_i ~ 1 + W_i + g_i(r)",
    }

    if save_outputs:
        if output_dir is None:
            raise ValueError("output_dir is required when save_outputs=True")
        output_dir = ensure_dir(output_dir)
        objective_df.to_csv(output_dir / "radius_objective.csv", index=False)
        moment_df.to_csv(output_dir / "moment_diagnostics.csv", index=False)
        write_json(output_dir / "radius_selection.json", selection)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(objective_df["radius"], objective_df["second_step_criterion"], marker="o")
        ax.axvline(r_hat, color="tab:red", linestyle="--", label=f"r_hat={r_hat:g}")
        if config.get("r_true") is not None:
            ax.axvline(float(config["r_true"]), color="tab:green", linestyle="--", label=f"r_true={float(config['r_true']):g}")
        ax.set_xlabel("Radius")
        ax.set_ylabel("Residual MSE")
        ax.set_title("Dose-response sufficiency radius objective")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "objective_plot.png", dpi=160)
        plt.close(fig)
        lines = [
            "Dose-response sufficiency radius estimation summary",
            f"r_true: {config.get('r_true')}",
            f"r_hat: {r_hat}",
            f"minimum MSE: {selection['second_step_criterion_min']}",
            "Model: Y_i ~ 1 + W_i + g_i(r)",
        ]
        (output_dir / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return EstimationResult(objective_df=objective_df, moment_df=moment_df, selection=selection)

