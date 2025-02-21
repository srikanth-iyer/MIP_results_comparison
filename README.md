# Model Intercomparison Project Results Comparison

Automatically create comparison figures for MIP runs from GenX, Switch, TEMOA and USENSYS results. Upload a folder with your results summary files to the appropriate top-level folder (e.g. `full-base-200`) and a GitHub Action will create comparison figures.

## Case name convention

The folders with model results for each case either start with `full` (52-weeks of hourly data) or `20-week`. Folders with `base` represent a net-zero CO2 emission scenario, and folders with `current-policies` represent current US policies (RPS/CES standards, IRA tax credits, etc). Additional configuration options include the CO2 slack price in net-zero scenarios (`full-base-200` has a $200/tonne slack price), economic retirement, transmission expansion constraints, the inclusion of unit commitment constraints, and fuel price sensitivities.

## Data description

Data folders contains summary results for a capacity expansion run by each model (`<model>_results_summary`), the translation of those results into GenX operational model inputs (`<model>_op_inputs`), and selected outputs of the GenX operational model.

## PowerGenome settings and inputs

PowerGenome is used to create all model inputs. The settings for PowerGenome are in `case_settings/26-zone/settings-atb2023`. Data inputs used by PowerGenome in this study are available at doi:10.5281/zenodo.14906951.

## Dashboard results

The results are compiled as dashboards. A static version with results for each case and a series of comparisons can be accessed at <https://gschivley.github.io/MIP_results_comparison>. An interactive version that lets the user select cases for comparison and modify figures is available at <https://mip-results.shinyapps.io/mip-results>.

If the interactive shinyapps website does not work, an experimental version is also available at <https://gschivley.github.io/test-shinylive>.
