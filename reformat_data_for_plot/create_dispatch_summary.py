from pathlib import Path
from typing import Union

import pandas as pd

from .create_time_weights import create_time_weights


def create_dispatch_summary(
    genx_scenario_results_path: Union[str, Path],
    scenario_name: str,
    output_folder_path: Union[str, Path],
    planning_year: int,
    case: str = "Results_p1",
    weight_value: float = 1.0,
    verbose: bool = False,
) -> Path:
    """
    Create dispatch.csv with columns:
    hour, resource_name, tech_type, value, zone, planning_year, model, case

    Parsing rules (tailored to GenX power.csv structure):
    - Read power.csv from <genx_scenario_results_path>/results/power.csv
      The first row is the header. First column header is 'Resource' and subsequent
      headers are resource names. The second data row has first cell 'Zone' and contains
      the zone id for the corresponding resource columns.
      Remaining rows have first column like 't1', 't2', ... representing time periods.

    - Read time_weights.csv from <genx_scenario_results_path>/results/time_weights.csv
      and select time periods where Weight == weight_value (default 1.0). The selected
      time period integers are printed for debug visibility.

                Output path:
                - <output_folder_path>/<scenario_name>_results_summary/dispatch.csv

                tech_type mapping:
                - tech_type is mapped from Generators_data.csv located at
                    <output_folder_path>/<scenario_name>_op_inputs/Inputs/Inputs_p1/Generators_data.csv
                    using the Resource -> technology mapping. If the file or required columns
                    are missing, an informative error is raised.
    """

    import warnings

    genx_scenario_results_path = Path(genx_scenario_results_path)
    output_folder_path = Path(output_folder_path)

    results_dir = genx_scenario_results_path / "results"
    power_path = results_dir / "power.csv"
    if not power_path.exists():
        raise FileNotFoundError(f"Missing power.csv at: {power_path}")

    weights_path = results_dir / "time_weights.csv"
    if not weights_path.exists():
        weights_path = create_time_weights(
            genx_scenario_results_path, output_path=weights_path, verbose=verbose
        )

    # Read inputs
    power_df = pd.read_csv(power_path)
    weights_df = pd.read_csv(weights_path)

    # Identify label column (expected to be 'Resource')
    label_col = power_df.columns[0]
    if str(label_col).lower() != "resource":
        warnings.warn(
            "First column in power.csv is '{label_col}', expected 'Resource'. Proceeding but results may be incorrect.".format(
                label_col=label_col
            )
        )

    # Extract resource columns (exclude the label column and 'Total' aggregate column if present)
    resource_cols = [c for c in power_df.columns if c not in (label_col, "Total")]
    if not resource_cols:
        raise ValueError("No resource columns found in power.csv")
    
    # Zone mapping from the row where first column == 'Zone'
    zone_row = power_df[power_df[label_col].astype(str) == "Zone"]
    if zone_row.empty:
        raise ValueError("Could not find a row starting with 'Zone' in power.csv to map zones")
    zone_row = zone_row.iloc[0]
    zones_map = {res: int(zone_row[res]) for res in resource_cols}

    # Time rows (first column like 't<integer>')
    time_rows = power_df[power_df[label_col].astype(str).str.match(r"^t\d+$", na=False)].copy()
    if time_rows.empty:
        raise ValueError("No time rows like 't1', 't2', ... found in power.csv")
    time_rows["hour"] = time_rows[label_col].astype(str).str.replace("t", "", regex=False).astype(int)

    # Select hours by weight
    if not {"Time", "Weight"}.issubset(set(weights_df.columns)):
        raise ValueError("time_weights.csv must contain columns 'Time' and 'Weight'")
    selected_hours = weights_df.loc[weights_df["Weight"] == weight_value, "Time"].astype(int).tolist()

    # Filter time rows to only selected hours
    time_rows = time_rows[time_rows["hour"].isin(selected_hours)]
    if time_rows.empty:
        raise ValueError("After filtering by time weights, no matching time rows remain in power.csv")

    # Round float values to integers for resource columns
    for col in resource_cols:
        time_rows[col] = time_rows[col].round().astype(int)

    # Melt to long format: one row per hour-resource
    long_df = time_rows.melt(
        id_vars=[label_col, "hour"],
        value_vars=resource_cols,
        var_name="resource_name",
        value_name="value",
    )

    # Map zone per resource from the 'Zone' row
    long_df["zone"] = long_df["resource_name"].map(zones_map)

    # Build final schema
    long_df["planning_year"] = planning_year
    long_df["model"] = scenario_name
    long_df["case"] = case

    # Map tech_type from Generators_data.csv under op_inputs
    gen_data_path = (
        Path(output_folder_path)
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
    tech_col = (
        "technology"
        if "technology" in gen_df.columns
        else ("Technology" if "Technology" in gen_df.columns else None)
    )
    if tech_col is None or "Resource" not in gen_df.columns:
        raise ValueError(
            "Generators_data.csv must contain 'Resource' and 'technology' (or 'Technology') columns."
        )

    tech_map = gen_df.set_index("Resource")[tech_col]
    long_df["tech_type"] = long_df["resource_name"].map(tech_map)

    dispatch = long_df[["hour", "resource_name", "tech_type", "value", "zone", "planning_year", "model", "case"]].copy()

    # Write output
    results_summary_folder_path = output_folder_path / f"{scenario_name}_results_summary"
    results_summary_folder_path.mkdir(parents=True, exist_ok=True)
    out_path = results_summary_folder_path / "dispatch.csv"
    dispatch.to_csv(out_path, index=False)
    # Also save as compressed gz file
    gz_out_path = out_path.with_suffix('.csv.gz')
    dispatch.to_csv(gz_out_path, index=False, compression='gzip')
    return out_path


if __name__ == "__main__":
    # Example direct run (adjust scenario as needed)
    scenario = "p4_Mod_Elect_Low_RE"
    out = create_dispatch_summary(
        genx_scenario_results_path=Path(r"C:\Users\Sriki\MIP_results_comparison-1\genx_results") / scenario,
        scenario_name=scenario,
        output_folder_path=Path(r"C:\Users\Sriki\MIP_results_comparison-1\20-week-genx"),
        planning_year=2030,
        case="Results_p1",
        weight_value=1.0,
    )
    print(f"Wrote dispatch summary to: {out}")
