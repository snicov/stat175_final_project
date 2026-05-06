# Simulation Pipeline

This directory contains the redesigned simulation pipeline for the final project.

## Main files

- `sim_utils.py`: shared utilities for distances, treatment redraws, network generation, plotting, and IO.
- `dgp.py`: fixed-population covariates, location generation, latent network generation, spillover construction, and outcomes.
- `exposure_mapping.py`: paper-aligned exposure-mapping radius estimator.
- `run_pipeline.py`: generic config-driven runner for scenario grids and multi-seed aggregation.

## Scenario layout

Each scenario has its own folder:

```text
simulations/scenarios/scenario_<NN>_<name>/
  config.yaml
  output/
    run_<timestamp>/
```

The `config.yaml` files are JSON-compatible YAML so they can be parsed without extra dependencies.

## Estimator alignment

The estimator uses

```math
\eta_{m,i}(r)=Y_i\left\{\psi_m(W)-\widehat{E}[\psi_m(W)\mid W_i,g_i(W;r)]\right\}.
```

- `phi_m(Y_i)=Y_i`.
- Placebo assignments follow the known experimental design.
- Default placebo redraw count is `B=50`, configurable per scenario.
- Initial scenarios use annulus treated-share bands as the `psi_m` dictionary.

## Run one scenario

```bash
python3 simulations/run_pipeline.py \
  --config simulations/scenarios/scenario_01_er_p_sweep/config.yaml
```

For a quick smoke run:

```bash
python3 simulations/run_pipeline.py \
  --config simulations/scenarios/scenario_01_er_p_sweep/config.yaml \
  --max_seeds 1
```

