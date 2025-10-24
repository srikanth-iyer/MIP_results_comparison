"""
Create a resource capacity summary CSV from GenX inputs and results.

This module exposes a function `create_resource_capacity` and a sample usage in the
`__main__` block that uses the parameters previously hardcoded in this script.
"""

from pathlib import Path
from typing import Union, Iterable, Tuple

import pandas as pd
import warnings
import traceback


def map_capacity_to_resources(
    resource_names: Iterable[str],
    capacity_df: pd.DataFrame,
    start_col: str = "StartCap",
    end_col: str = "EndCap",
    resource_col: str | None = "Resource",
    infer_resource_col: bool = True,
    error_on_duplicates: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """Return start and end capacity Series aligned to ``resource_names``."""

    resource_series = pd.Series(list(resource_names), name="resource_name")

    if resource_col is not None and resource_col not in capacity_df.columns:
        if infer_resource_col:
            lower_map = {c.lower(): c for c in capacity_df.columns}
            candidates_priority = [
                "resource",
                "resource_name",
                "generator",
                "gen",
                "name",
            ]
            inferred = next((lower_map[c] for c in candidates_priority if c in lower_map), None)
            if inferred is None:
                raise ValueError(
                    "Could not locate a resource column in capacity_df. Available columns: "
                    f"{list(capacity_df.columns)}"
                )
            resource_col = inferred
        else:
            raise ValueError(
                f"Specified resource_col '{resource_col}' not in capacity_df columns: {list(capacity_df.columns)}"
            )

    if resource_col is None:
        raise ValueError("resource_col could not be resolved.")

    missing_cols = {start_col, end_col} - set(capacity_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required capacity columns {missing_cols}. Available: {list(capacity_df.columns)}"
        )

    if error_on_duplicates:
        dup = capacity_df[resource_col][capacity_df[resource_col].duplicated(keep=False)].unique()
        if len(dup) > 0:
            raise ValueError(
                "Duplicate resource entries in capacity_df for: " + ", ".join(map(str, dup))
            )

    indexed = capacity_df.set_index(resource_col)
    start_mapped = resource_series.map(indexed[start_col])  # type: ignore[index]
    end_mapped = resource_series.map(indexed[end_col])  # type: ignore[index]

    unmapped = resource_series[start_mapped.isna() | end_mapped.isna()].unique()
    if len(unmapped) > 0:
        warnings.warn(
            "Unmapped resources (NaN capacity values): " + ", ".join(map(str, unmapped)),
            RuntimeWarning,
        )

    start_mapped.name = start_col
    end_mapped.name = end_col
    return start_mapped, end_mapped


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
    capacity_factor_filename: str = "capacityfactor.csv",
    output_filename: str = "resource_capacity.csv",
) -> Path:
    """Build a resource capacity dataframe and write it to CSV."""

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
    
    df["new_build"] = generators_df["New_Build"]
    df["existing"] = (df["new_build"] == 0).astype(int)

    # Wrap capacity reading, mapping and merging in try/except to surface debug info
    try:
        capacity_df = pd.read_csv(genx_scenario_results_path / "results" / capacity_filename)

        start_series, end_series = map_capacity_to_resources(
            df["resource_name"], capacity_df, start_col="StartCap", end_col="EndCap", resource_col="Resource"
        )
        df["start_value"] = start_series.values
        df["end_value"] = end_series.values

        # Read capacity factor and merge
        capacity_factor_df = pd.read_csv(genx_scenario_results_path / "results" / capacity_factor_filename)

        # Ensure capacity_factor_df has Resource column, infer if needed
        if "Resource" not in capacity_factor_df.columns:
            candidate_cols = [c for c in capacity_factor_df.columns if c.lower() in {"resource", "generator", "name", "resource_name"}]
            if candidate_cols:
                capacity_factor_df = capacity_factor_df.rename(columns={candidate_cols[0]: "Resource"})
            else:
                warnings.warn("Could not find Resource column in capacity_factor file, setting capacity_factor to NaN")
                df["capacity_factor"] = float('nan')

        if "Resource" in capacity_factor_df.columns:
            # Check for capacity factor column (common names)
            cf_col = None
            for col in ["capacity_factor", "CapacityFactor", "CF", "cf"]:
                if col in capacity_factor_df.columns:
                    cf_col = col
                    break

            if cf_col is None:
                warnings.warn("Could not find capacity factor column in capacity_factor file, setting to NaN")
                df["capacity_factor"] = float('nan')
            else:
                # Use pandas merge to join capacity factor data
                df = df.merge(
                    capacity_factor_df[["Resource", cf_col]].rename(columns={cf_col: "capacity_factor"}),
                    left_on="resource_name",
                    right_on="Resource",
                    how="left"
                ).drop(columns=["Resource"])

                # Warn about unmapped capacity factors
                unmapped_cf = df[df["capacity_factor"].isna()]["resource_name"].unique()
                if len(unmapped_cf) > 0:
                    warnings.warn(
                        f"Resources without capacity factor data: {', '.join(map(str, unmapped_cf))}",
                        RuntimeWarning
                    )
    except Exception as e:
        print("[ERROR] Exception during capacity read/map/merge:")
        print("[ERROR] ", str(e))
        print("[ERROR] Traceback:\n", traceback.format_exc())
        raise

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
        r"C:\Users\Sriki\MIP_results_comparison-1\20-week-genx\GenX_op_inputs\Inputs\Inputs_p1"
    )
    results_summary_folder_path = Path(
        r"C:\Users\Sriki\MIP_results_comparison-1\20-week-genx\GenX_results_summary"
    )
    genx_scenario_results_path = Path(
        r"C:\Users\Sriki\MIP_results_comparison-1\20-week-genx\p1_High_Elect_Mid_RE"
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
