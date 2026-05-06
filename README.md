# STAT 175 Final Project

This repository contains the simulation pipeline and final report for a STAT 175 project on recovering spillover radii when interference is mediated by latent networks but the researcher observes geography.

## Repository Structure

- `simulations/`: core simulation code, DGPs, estimators, and scenario runner.
- `simulations/scenarios/`: one folder per scenario, including reproducible config files.
- `requirements.txt`: Python dependencies for running the simulations.

Large generated simulation outputs are intentionally not tracked in GitHub. They can be regenerated from the scenario configs.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Validate the Pipeline

```bash
python3 simulations/validate_pipeline.py
```

## Run One Scenario

Each scenario is controlled by a `config.yaml` file. For example:

```bash
python3 simulations/run_pipeline.py \
  --config simulations/scenarios/scenario_21_sigma2_sweep_saturated_expected_er_dose_response_gamma1/config.yaml
```

Outputs are written under that scenario's `output/` folder. These outputs are ignored by Git because they can become very large.

## Reproduce Main Report Results

The final report is based on scenarios 20 through 31. To regenerate the main simulation outputs, run:

```bash
for config in \
  simulations/scenarios/scenario_20_sigma2_sweep_expected_er_dose_response_gamma1/config.yaml \
  simulations/scenarios/scenario_21_sigma2_sweep_saturated_expected_er_dose_response_gamma1/config.yaml \
  simulations/scenarios/scenario_22_er_p_sweep_saturated_expected_er_dose_response_gamma1/config.yaml \
  simulations/scenarios/scenario_23_r_true_sweep_saturated_expected_er_dose_response_gamma1/config.yaml \
  simulations/scenarios/scenario_24_n_sweep_saturated_expected_er_dose_response_gamma1/config.yaml \
  simulations/scenarios/scenario_25_distance_noise_sweep_saturated_expected_er_dose_response_gamma1/config.yaml \
  simulations/scenarios/scenario_26_distance_decay_sweep_saturated_dose_response_gamma1/config.yaml \
  simulations/scenarios/scenario_27_homophily_sweep_saturated_dose_response_gamma1/config.yaml \
  simulations/scenarios/scenario_28_sbm_community_sweep_saturated_dose_response_gamma1/config.yaml \
  simulations/scenarios/scenario_29_central_hub_sweep_saturated_dose_response_gamma1/config.yaml \
  simulations/scenarios/scenario_30_two_cluster_geography_sweep_saturated_dose_response_gamma1/config.yaml \
  simulations/scenarios/scenario_31_treatment_probability_sweep_saturated_dose_response_gamma1/config.yaml
do
  python3 simulations/run_pipeline.py --config "$config"
done
```

Each run uses the seeds specified in its config file and writes aggregate and per-seed summaries to the scenario output directory.

## Notes on Outputs

The full simulation outputs include per-seed datasets, pairwise distance files, and latent edge lists. These are large and are excluded from GitHub by `.gitignore`. Re-running the scenario configs regenerates them.
