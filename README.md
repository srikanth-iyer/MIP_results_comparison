# GenX Scenario Data Export

## Quick start

Minimal steps to get started locally (PowerShell):

```powershell
# Create conda environment from the provided spec
conda env create -f environment.yml
conda activate mip_figures

# Or using pip in an existing Python 3.10+ environment
pip install -r requirements.txt

# Run the full one-click workflow (scenario export + Quarto render)
python run_full_workflow.py --verbose
```

### One-click workflow overview

`run_full_workflow.py` wraps the export pipeline and Quarto rendering so you can refresh
the dashboard end-to-end in a single command. Key options:

- `--input-folder PATH` and `--output-folder PATH` to override the default
  `genx_scenarios_inputs/` and `genx-scenarios/` locations.
- `--render-target TARGET` to point Quarto at a specific QMD file or to `.` for the
  entire project.
- `--render-arg ARG` (repeatable) to forward extra flags to `quarto render`, e.g.
  `--render-arg "--execute"`.
- `--skip-quarto` if you only need to regenerate the CSV exports.
- `--verbose` for detailed logging and `--debug-overwrites` to allow overwriting
  generated files.

Example:

```powershell
python run_full_workflow.py --verbose --render-arg "--execute"
```

## Scenario reformatting

The scripts under `reformat_data_for_plot/` convert raw GenX outputs into the tidy CSVs
expected by the dashboards. Run `python reformat_data_for_plot/format_genx_result_for_plotting.py`
for direct control over input/output locations or use `run_full_workflow.py` from the
quick start section to combine reformatting with Quarto rendering.

```powershell
# Export GenX scenario results with default paths (default input: (genx_scenarios_inputs); default output:(genx-scenarios))
python reformat_data_for_plot/format_genx_result_for_plotting.py 

# Or specify custom input and output folders
python reformat_data_for_plot/format_genx_result_for_plotting.py --input path/to/scenarios --output-folder path/to/output

# Enable verbose logging and debug overwrites
python reformat_data_for_plot/format_genx_result_for_plotting.py --verbose --debug-overwrites
```

## Quarto dashboard

The interactive dashboard under `docs/genx-notebooks/genx-results-scenarios.html` is generated from the Quarto source `genx-notebooks/genx-results-scenarios.qmd` using the project configuration in `_quarto.yml`.
Add or modify the source notebook to change the dashboard content.

```powershell
# render the full dashboard site (outputs to docs/)
quarto render

# or render only the a specific dashboard page
quarto render genx-notebooks/genx-results-scenarios.qmd
```

After rendering, open `docs/genx-notebooks/genx-results-scenarios.html` in a browser to preview locally. Quarto already targets the `docs/` folder, so pushing those artifacts keeps GitHub Pages in sync.

## Push to GitHub Pages

To publish the dashboard to GitHub Pages, push the rendered `docs/` folder to the `main` branch of the repository. GitHub Pages will automatically serve the content from there.

## Repository structure (top-level)

- `genx_scenarios_inputs/` - Source GenX run folders (default input)
- `genx-scenarios/` - Processed GenX inputs and outputs (default output)
- `genx-notebooks/` - Notebook helpers and plotting config
- `reformat_data_for_plot/` - Python package containing data export and summary utilities
  - `format_genx_result_for_plotting.py` - Main CLI script to export GenX scenario data
  - `build_generators_data.py` - Utility that compiles Generators_Data.csv
  - `create_*_summary.py` - Utilities to create summary CSVs for plotting/dashboard
- `_quarto.yml` - Quarto dashboard configuration
- `docs/` - Rendered Quarto dashboard output
- `run_full_workflow.py` - One-click helper that runs both the export pipeline and Quarto render
