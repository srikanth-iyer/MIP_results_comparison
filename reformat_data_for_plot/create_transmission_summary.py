"""Create a transmission summary CSV from GenX network inputs and expansion results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd
import warnings


_DEFAULT_NETWORK_FILENAME = "Network.csv"
_DEFAULT_NETWORK_EXPANSION_FILENAME = "network_expansion.csv"
_DEFAULT_OUTPUT_FILENAME = "transmission.csv"


def _resolve_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    """Return the matching column name (case-insensitive) for the provided candidates."""

    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match
    if required:
        raise ValueError(
            f"Could not locate any of the expected columns {list(candidates)} in {list(df.columns)}"
        )
    return None


def _find_network_file(root: Path, filename: str) -> Path:
    """Return the resolved Network.csv path, accepting either root/Network.csv or root/system/Network.csv."""

    candidates = [root / filename, root / "system" / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find '{filename}' in {root} or {root / 'system'}"
    )


def create_transmission_summary(
    genx_scenario_results_path: Union[str, Path],
    scenario_name: str,
    output_folder_path: Union[str, Path],
    planning_year: int,
    case: str,
    unit: str = "MW",
    *,
    network_filename: str = _DEFAULT_NETWORK_FILENAME,
    network_expansion_filename: str = _DEFAULT_NETWORK_EXPANSION_FILENAME,
    output_filename: str = _DEFAULT_OUTPUT_FILENAME,
    atol: float = 1e-6,
) -> Path:
    """Build ``transmission.csv`` summarising start and end transmission capacities."""

    genx_scenario_results_path = Path(genx_scenario_results_path)
    output_folder_path = Path(output_folder_path)

    network_path = _find_network_file(genx_scenario_results_path, network_filename)
    network_expansion_path = genx_scenario_results_path / "results" / network_expansion_filename

    if not network_expansion_path.exists():
        raise FileNotFoundError(f"Missing network expansion file at: {network_expansion_path}")

    network_df = pd.read_csv(network_path)
    expansion_df = pd.read_csv(network_expansion_path)

    line_col = _resolve_column(network_df, ["Network_Lines", "Line", "line"])
    zone_col = _resolve_column(network_df, ["Network_zones", "network_zones", "zone"])
    path_name_col = _resolve_column(network_df, ["transmission_path_name", "Transmission_Path_Name"])
    max_flow_col = _resolve_column(network_df, ["Line_Max_Flow_MW", "Line_Max_Flow_mw", "Max_Flow"])
    min_flow_col = _resolve_column(
        network_df,
        ["Line_Min_Flow_MW", "Line_Min_Flow_mw", "Min_Flow"],
        required=False,
    )

    line_expansion_col = _resolve_column(expansion_df, ["Line", "line", "Network_Lines"])
    new_cap_col = _resolve_column(expansion_df, ["New_Trans_Capacity", "new_trans_capacity"])
    cost_col = _resolve_column(
        expansion_df,
        ["Cost_Trans_Capacity", "cost_trans_capacity", "Cost"],
    )

    summary = network_df[[line_col, zone_col, path_name_col, max_flow_col]].copy()
    summary.columns = ["Network_Lines", "Network_zones", "transmission_path_name", "start_value"]

    summary["Network_Lines"] = pd.to_numeric(summary["Network_Lines"], errors="coerce")
    if summary["Network_Lines"].isna().any():
        warnings.warn(
            "Encountered rows in Network.csv with non-numeric Network_Lines; they will be dropped.",
            RuntimeWarning,
        )
        summary = summary.dropna(subset=["Network_Lines"])

    if min_flow_col is not None and min_flow_col in network_df.columns:
        compare_series = pd.to_numeric(network_df[min_flow_col], errors="coerce")
        start_series = pd.to_numeric(summary["start_value"], errors="coerce")
        mismatch_mask = ~(
            np.isclose(start_series, compare_series, atol=atol)
            | (start_series.isna() & compare_series.isna())
        )
        if mismatch_mask.any():
            mismatched_lines = summary.loc[mismatch_mask, "Network_Lines"].tolist()
            warnings.warn(
                "Line_Max_Flow_MW and Line_Min_Flow_MW differ for lines: "
                + ", ".join(map(str, mismatched_lines)),
                RuntimeWarning,
            )

    summary["start_value"] = pd.to_numeric(summary["start_value"], errors="coerce")

    expansion_df = expansion_df[[line_expansion_col, new_cap_col, cost_col]].copy()
    expansion_df.columns = ["Network_Lines", "New_Trans_Capacity", "Cost_Trans_Capacity"]
    expansion_df["Network_Lines"] = pd.to_numeric(expansion_df["Network_Lines"], errors="coerce")

    merged = summary.merge(expansion_df, on="Network_Lines", how="left")
    merged["New_Trans_Capacity"] = pd.to_numeric(
        merged["New_Trans_Capacity"], errors="coerce"
    ).fillna(0.0)
    merged["Cost_Trans_Capacity"] = pd.to_numeric(merged["Cost_Trans_Capacity"], errors="coerce")
    merged["end_value"] = merged["start_value"].fillna(0.0) + merged["New_Trans_Capacity"]

    merged["planning_year"] = int(planning_year)
    merged["case"] = case
    merged["model"] = scenario_name
    merged["unit"] = unit

    columns_order = [
        "model",
        "case",
        "planning_year",
        "unit",
        "Network_zones",
        "Network_Lines",
        "transmission_path_name",
        "start_value",
        "New_Trans_Capacity",
        "Cost_Trans_Capacity",
        "end_value",
    ]
    merged = merged[columns_order]

    results_summary_folder_path = output_folder_path / f"{scenario_name}_results_summary"
    results_summary_folder_path.mkdir(parents=True, exist_ok=True)
    output_path = results_summary_folder_path / output_filename
    merged.to_csv(output_path, index=False)
    return output_path


__all__ = ["create_transmission_summary"]
