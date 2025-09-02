"""
Create a resource capacity summary CSV from GenX inputs and results.

This module exposes a function `create_resource_capacity` and a sample usage in the
`__main__` block that uses the parameters previously hardcoded in this script.
"""

from pathlib import Path
from typing import Union

import pandas as pd


def create_resource_capacity(
    model_name: str,
    case_name: str,
    scenario_folder_path: Union[str, Path],
    genx_scenario_results_path: Union[str, Path],
    results_summary_folder_path: Union[str, Path],
    planning_year: int = 2030,
    unit: str = "MW",
    generators_filename: str = "Generators_data.csv",
    capacity_filename: str = "capacity.csv",
    output_filename: str = "resource_capacity.csv",
) -> Path:
    """
    Build a resource capacity dataframe and write it to CSV.

    Parameters
    - model_name: Model identifier (e.g., "GenX").
    - case_name: Case identifier (e.g., "p1").
    - scenario_folder_path: Path to GenX Inputs folder that contains generators.csv.
    - genx_scenario_results_path: Path to the scenario folder that contains results/capacity.csv.
    - results_summary_folder_path: Output directory where the CSV will be written.
    - planning_year: Planning year to stamp on the output rows. Default 2030.
    - unit: Capacity unit. Default "MW".
    - generators_filename: Name of the generators file. Default "generators.csv".
    - capacity_filename: Name of the capacity results file. Default "capacity.csv".
    - output_filename: Name of the output CSV file. Default "resource_capacity.csv".

    Returns
    - Path to the written CSV file.
    """

    scenario_folder_path = Path(scenario_folder_path)
    genx_scenario_results_path = Path(genx_scenario_results_path)
    results_summary_folder_path = Path(results_summary_folder_path)

    # Create a DataFrame with the specified columns
    columns = [
        "model",
        "zone",
        "resource_name",
        "tech_type",
        "planning_year",
        "case",
        "unit",
        "start_value",
        "end_value",
    ]
    df = pd.DataFrame(columns=columns)

    # Read inputs
    generators_df = pd.read_csv(scenario_folder_path / generators_filename)

    # Map columns from generators
    df["resource_name"] = generators_df["Resource"]
    df["zone"] = generators_df["region"]
    df["model"] = model_name
    df["case"] = case_name
    df["planning_year"] = int(planning_year)
    df["tech_type"] = generators_df["technology"]
    df["unit"] = unit

    # Read capacity results
    capacity_df = pd.read_csv(genx_scenario_results_path / "results" / capacity_filename)
    df["start_value"] = capacity_df["StartCap"]
    df["end_value"] = capacity_df["EndCap"]

    # Ensure output directory exists
    results_summary_folder_path.mkdir(parents=True, exist_ok=True)
    output_path = results_summary_folder_path / output_filename
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    # Sample usage with the same parameters previously defined in this file
    model_name = "GenX"
    case_name = "p1"

    scenario_folder_path = Path(
        r"C:\Users\Sriki\MIP_results_comparison-1\20-week-genx_simulations\GenX_op_inputs\Inputs\Inputs_p1"
    )
    results_summary_folder_path = Path(
        r"C:\Users\Sriki\MIP_results_comparison-1\20-week-genx_simulations\GenX_results_summary"
    )
    genx_scenario_results_path = Path(
        r"C:\Users\Sriki\MIP_results_comparison-1\20-week-genx_simulations\p1_High_Elect_Mid_RE"
    )

    output = create_resource_capacity(
        model_name=model_name,
        case_name=case_name,
        scenario_folder_path=scenario_folder_path,
        genx_scenario_results_path=genx_scenario_results_path,
        results_summary_folder_path=results_summary_folder_path,
        planning_year=2030,
        unit="MW",
    )
    print(f"Wrote resource capacity CSV to: {output}")

