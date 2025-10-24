from pathlib import Path
from typing import Union

import pandas as pd


def create_generation_summary(
    genx_scenario_results_path: Union[str, Path],
    scenario_name: str,
    output_folder_path: Union[str, Path],
    planning_year: int,
    case: str = "Results_p1",
    unit: str = "MWh",
) -> Path:
    """
    Create generations.csv with columns:
    model, zone, resource_name, tech_type, planning_year, case, timesept, unit, value.

    Data sources:
    - capacityfactor.csv from: <genx_scenario_results_path>/results/capacityfactor.csv
      Uses columns: Resource, Zone, AnnualSum (for value).
    - Generators_data.csv from: <output_folder_path>/<scenario_name>_op_inputs/Inputs/Inputs_p1/Generators_data.csv
      Uses columns: Resource, technology (maps to tech_type).

    Returns:
    - Path to the written generations.csv in <output_folder_path>/<scenario_name>_results_summary
    """

    genx_scenario_results_path = Path(genx_scenario_results_path)
    output_folder_path = Path(output_folder_path)

    cap_path = genx_scenario_results_path / "results" / "capacityfactor.csv"
    if not cap_path.exists():
        raise FileNotFoundError(f"Missing capacityfactor.csv at: {cap_path}")

    cap = pd.read_csv(cap_path)

    # Validate expected columns or provide helpful errors/fallbacks
    # Expect at minimum Resource, Zone, and an AnnualSum value column.
    required_cols = {"Resource", "Zone", "AnnualSum"}
    if not required_cols.issubset(set(cap.columns)):
        raise ValueError(
            f"capacityfactor.csv must contain columns {required_cols}, found: {list(cap.columns)}"
        )

    # Load technology mapping from Generators_data.csv
    gen_data_path = (
        output_folder_path
        / f"{scenario_name}_op_inputs"
        / "Inputs"
        / "Inputs_p1"
        / "Generators_data.csv"
    )

    if not gen_data_path.exists():
        raise FileNotFoundError(
            "Generators_data.csv not found. Ensure export_genx_for_plotting created it before calling this function: "
            f"{gen_data_path}"
        )

    gen_df = pd.read_csv(gen_data_path)
    # Be tolerant to capitalization of 'technology'
    tech_col = "technology" if "technology" in gen_df.columns else ("Technology" if "Technology" in gen_df.columns else None)
    if tech_col is None or "Resource" not in gen_df.columns:
        raise ValueError(
            "Generators_data.csv must contain 'Resource' and 'technology' (or 'Technology') columns."
        )

    tech_map = gen_df.set_index("Resource")[tech_col]
    # Map new_build values from gen_df to cap resources
    new_build_map = gen_df.set_index("Resource")["New_Build"]
    # Map existing values (inverse of new_build) from gen_df to cap resources
    existing_map = gen_df.set_index("Resource")["New_Build"].apply(lambda x: 0 if x == 1 else 1)
    # Build summary
    summary = pd.DataFrame({
        "model": scenario_name,
        "zone": cap["Zone"],
        "resource_name": cap["Resource"],
        "tech_type": cap["Resource"].map(tech_map),
        "planning_year": planning_year,
        "case": case,
        "timestep": "all",
        "new_build": cap["Resource"].map(new_build_map),
        "existing": cap["Resource"].map(existing_map),
        "unit": unit,
        "value": cap["AnnualSum"],
    })

    # Write output
    results_summary_folder_path = output_folder_path / f"{scenario_name}_results_summary"
    results_summary_folder_path.mkdir(parents=True, exist_ok=True)
    out_path = results_summary_folder_path / "generation.csv"
    summary.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    # Example direct run (adjust scenario as needed)
    scenario = "p4_Mod_Elect_Low_RE"
    out = create_generation_summary(
        genx_scenario_results_path=Path(r"C:\Users\Sriki\MIP_results_comparison-1\genx_results") / scenario,
        scenario_name=scenario,
        output_folder_path=Path(r"C:\Users\Sriki\MIP_results_comparison-1\20-week-genx"),
        planning_year=2030,
        # case defaults to "Results_p1"
        # unit defaults to "MWh"
    )
    print(f"Wrote generations summary to: {out}")
