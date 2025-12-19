# Reformatting Utilities for GenX Plotting

This package groups together the scripts that turn raw GenX scenario exports into tidy CSVs that drive the plotting notebooks and dashboards. Each module is importable as part of `reformat_data_for_plot`.

## Module overview

| Module | Primary functions | Purpose |
| --- | --- | --- |
| `build_generators_data.py` | `build_generators_data`, `compare_columns` | Merge resource-level inputs (thermal, storage, VRE, etc.) into a single `Generators_data.csv` aligned with GenX expectations. |
| `check_tech_map_coverage_in_generators_data.py` | `check_tech_map_coverage` | Validate that every generator resource maps to a technology label using the site’s tech-map helper. |
| `create_dispatch_summary.py` | `create_dispatch_summary` | Convert hourly `power.csv` results into a long-form dispatch table with resource, zone, tech type, and planning year metadata. |
| `create_emissions_summary.py` | `create_emissions_summary` | Extract annual zone-level emissions totals from `results/emissions.csv`. |
| `create_generation_summary.py` | `create_generation_summary` | Summarize annual generation by resource and technology from `capacityfactor.csv`. |
| `create_netrevenue_summary.py` | `create_netrevenue_summary` | Reshape `results/NetRevenue.csv` into a tidy per-resource breakdown for plotting. |
| `create_resource_capacity.py` | `create_resource_capacity`, `map_capacity_to_resources` | Combine GenX input metadata with `capacity.csv` and `capacityfactor.csv` to track start/end capacity, new build flags, and capacity factors. |
| `create_time_weights.py` | `create_time_weights` | Rebuild `time_weights.csv` when it is missing using representative period metadata. |
| `format_genx_result_for_plotting.py` | `create_annual_demand_csv`, `export_genx_for_plotting`, `export_all_genx_scenarios` | Orchestrate the full export pipeline that copies inputs/results, regenerates derived tables, and aggregates summaries for every scenario. |
| `gz_to_csv.py` | `gz_to_csv` | Ad-hoc helper to decompress `.csv.gz` outputs (e.g., dispatch summaries) when needed. |

## Expected input structure

The utilities assume the standard GenX export layout where each scenario folder contains:

- `system/` with core inputs such as `Demand_data.csv`, `Fuels_data.csv`, `Network.csv`, `Period_map.csv`, and `Representative_Period.csv`.
- `resources/` with resource attribute CSVs (`Thermal.csv`, `Storage.csv`, `Vre.csv`, `Must_run.csv`, `Hydro.csv`, `Flex_demand.csv`) and, optionally, `policy_assignments/` overrides.
- `results/` with simulation outputs including `power.csv`, `capacityfactor.csv`, `capacity.csv`, `costs.csv`, `emissions.csv`, `nse.csv`, and `NetRevenue.csv`.
- Optional multi-period subfolders named `Inputs_p1`, `Inputs_p2`, etc., each mirroring the structure above.

When running the full export pipeline, the code expects to find or will construct:

- `<scenario>_op_inputs/Inputs/Inputs_px/Generators_data.csv` – consolidated generator metadata built from the source `resources` directory.
- `<scenario>_results_summary/` – destination for derived CSVs (`annual_demand.csv`, `dispatch.csv`, `emissions.csv`, `generation.csv`, `resource_capacity.csv`).

## Output datasets

The summaries produced by this package are CSV files ready for visualization notebooks:

- `annual_demand.csv`: columns `zone`, `annual_demand`, `planning_year`.
- `dispatch.csv`: columns `hour`, `resource_name`, `tech_type`, `value`, `zone`, `planning_year`, `model`, `case` (and compressed `.csv.gz`).
- `emissions.csv`: columns `model`, `zone`, `planning_year`, `case`, `unit`, `value`.
- `generation.csv`: columns `model`, `zone`, `resource_name`, `tech_type`, `planning_year`, `case`, `timestep`, `new_build`, `existing`, `unit`, `value`.
- `resource_capacity.csv`: columns `model`, `zone`, `resource_name`, `tech_type`, `planning_year`, `case`, `unit`, `start_value`, `end_value`, `new_build`, `existing`, and optional `capacity_factor`.
- `netrevenue.csv`: columns `model`, `planning_year`, `case`, `resource_name`, optional `zone/region/cluster/r_id`, `netrevenue_component`, `unit`, `value`.
- `Generators_data.csv`: wide table keyed by `Resource` with the union of attributes drawn from input resource files.

Refer to individual docstrings for additional details on optional parameters and logging behaviour.
