# GenX Scenario Data Export

## Quick start

Minimal steps to get started locally (PowerShell):

```powershell
# Create conda environment from the provided spec
conda env create -f environment.yml
conda activate mip_figures

# Or using pip in an existing Python 3.10+ environment
pip install -r requirements.txt

# Export GenX scenario results with default paths
python reformat_data_for_plot/format_genx_result_for_plotting.py

# Or specify custom input and output folders
python reformat_data_for_plot/format_genx_result_for_plotting.py --input path/to/scenarios --output-folder path/to/output

# Enable verbose logging and debug overwrites
python reformat_data_for_plot/format_genx_result_for_plotting.py --verbose --debug-overwrites
```

## Quarto dashboard

The interactive dashboard under `docs/genx-notebooks/genx-results-scenarios.html` is generated from the Quarto source `genx-notebooks/genx-results-scenarios.qmd` using the project configuration in `_quarto.yml`.

```powershell
# render the full dashboard site (outputs to docs/)
quarto render

# or render only the main dashboard page
quarto render genx-notebooks/genx-results-scenarios.qmd
```

After rendering, open `docs/genx-notebooks/genx-results-scenarios.html` in a browser to preview locally. Quarto already targets the `docs/` folder, so pushing those artifacts keeps GitHub Pages in sync.

## Repository structure (top-level)

- `genx_scenarios_results/` - Source GenX run folders (default input)
- `genx-scenarios/` - Processed GenX inputs and outputs (default output)
- `genx-notebooks/` - Notebook helpers and plotting config
- `reformat_data_for_plot/` - Python package containing data export and summary utilities
  - `format_genx_result_for_plotting.py` - Main CLI script to export GenX scenario data
  - `build_generators_data.py` - Utility that compiles Generators_Data.csv
  - `create_*_summary.py` - Utilities to create summary CSVs for plotting/dashboard
- `_quarto.yml` - Quarto dashboard configuration
- `docs/` - Rendered Quarto dashboard output
