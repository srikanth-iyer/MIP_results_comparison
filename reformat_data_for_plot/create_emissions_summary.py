from pathlib import Path
from typing import Union

import pandas as pd


def create_emissions_summary(
    genx_scenario_results_path: Union[str, Path],
    scenario_name: str,
    output_folder_path: Union[str, Path],
    planning_year: int,
    case: str = "Results_p1",
    unit: str = "tons",
) -> Path:
    """
    Build an emissions summary CSV for a single GenX scenario.

    Inputs:
    - genx_scenario_results_path: Path or str to the specific scenario folder containing a "results/emissions.csv" file
    - scenario_name: Scenario name to include in the output (also used as the model name)
    - output_folder_path: Path or str to the root folder where "<scenario_name>_results_summary/emissions.csv" will be written
    - planning_year: Planning year to record in the summary
    - case: Case label for the summary rows (default: "Results_p1")
    - unit: Unit for the emissions values (default: "tons")

    Returns:
    - Path to the written emissions summary CSV.
    """

    # Normalize to Path objects if strings are provided
    genx_scenario_results_path = Path(genx_scenario_results_path)
    output_folder_path = Path(output_folder_path)

    input_emissions_filepath = genx_scenario_results_path / "results" / "emissions.csv"
    if not input_emissions_filepath.exists():
        raise FileNotFoundError(f"Missing emissions.csv at: {input_emissions_filepath}")

    df = pd.read_csv(input_emissions_filepath)
    if "Zone" not in df.columns:
        raise ValueError("Expected a 'Zone' column in emissions.csv")

    # Identify zone columns (digits) and find the 'AnnualSum' row
    zones = [c for c in df.columns if c not in ["Zone", "Total"] and str(c).isdigit()]
    if not zones:
        raise ValueError("No numeric zone columns found in emissions.csv")

    matches = df[df["Zone"] == "AnnualSum"]
    if matches.empty:
        raise ValueError("Could not find a row where Zone == 'AnnualSum' in emissions.csv")

    annual_sum_row = matches.iloc[0]
    annual_sum_values = annual_sum_row[zones]

    emissions_summary_df = pd.DataFrame(
        {
            "model": [scenario_name] * len(zones),
            "zone": zones,
            "planning_year": [planning_year] * len(zones),
            "case": [case] * len(zones),
            "unit": [unit] * len(zones),
            "value": annual_sum_values.values,
        }
    )

    results_summary_folder_path = output_folder_path / f"{scenario_name}_results_summary"
    results_summary_folder_path.mkdir(parents=True, exist_ok=True)
    output_path = results_summary_folder_path / "emissions.csv"
    emissions_summary_df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    # Sample execution showing how to use this function directly
    scenario = "p4_Mod_Elect_Low_RE"
    out = create_emissions_summary(
        genx_scenario_results_path=Path(r"C:\Users\Sriki\MIP_results_comparison-1\genx_results") / scenario,
        scenario_name=scenario,
        output_folder_path=Path(r"C:\Users\Sriki\MIP_results_comparison-1\20-week-genx"),
        planning_year=2030,
        # case defaults to "Results_p1"
    )
    print(f"Wrote emissions summary to: {out}")
