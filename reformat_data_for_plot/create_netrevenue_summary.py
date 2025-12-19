"""Create tidy net revenue summaries for GenX scenario periods."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import pandas as pd


def _pick_column(df: pd.DataFrame, candidates: Iterable[str], *, required: bool = False, label: str | None = None) -> str | None:
	"""Return the first matching column from ``candidates`` or raise if required."""

	for col in candidates:
		if col in df.columns:
			return col
	if required:
		display = label or "/".join(candidates)
		raise ValueError(f"NetRevenue.csv missing expected column(s): {display}")
	return None


def create_netrevenue_summary(
	genx_scenario_results_path: Union[str, Path],
	scenario_name: str,
	planning_year: int,
	*,
	case: str = "Results_p1",
	unit: str = "USD",
) -> pd.DataFrame:
	"""Build a long-form net revenue summary for a single planning year.

	Parameters
	----------
	genx_scenario_results_path:
		Path to the period directory that contains a ``results/NetRevenue.csv`` file.
	scenario_name:
		Scenario identifier stored in the ``model`` column of the summary output.
	planning_year:
		Planning year associated with the current period (e.g., 2030).
	case:
		Label identifying the period/case (default ``Results_p1``).
	unit:
		Revenue/profit unit to store in the summary (default ``USD``).

	Returns
	-------
	pd.DataFrame
		Long-form summary dataframe with per-resource net revenue components.
	"""

	results_path = Path(genx_scenario_results_path)
	net_revenue_path = results_path / "results" / "NetRevenue.csv"
	if not net_revenue_path.exists():
		raise FileNotFoundError(f"Missing NetRevenue.csv at: {net_revenue_path}")

	df = pd.read_csv(net_revenue_path)
	if df.empty:
		raise ValueError(f"NetRevenue.csv at {net_revenue_path} is empty")

	resource_col = _pick_column(df, ("Resource", "resource"), required=True, label="Resource")
	zone_col = _pick_column(df, ("zone", "Zone"))
	region_col = _pick_column(df, ("region", "Region"))
	cluster_col = _pick_column(df, ("Cluster", "cluster"))
	rid_col = _pick_column(df, ("R_ID", "r_id", "rid"))

	rename_map = {resource_col: "resource_name"}
	if zone_col:
		rename_map[zone_col] = "zone"
	if region_col:
		rename_map[region_col] = "region"
	if cluster_col:
		rename_map[cluster_col] = "cluster"
	if rid_col:
		rename_map[rid_col] = "r_id"

	renamed = df.rename(columns=rename_map)

	id_vars: list[str] = ["resource_name"]
	for optional in ["zone", "region", "cluster", "r_id"]:
		if optional in renamed.columns:
			id_vars.append(optional)

	value_cols = [c for c in renamed.columns if c not in id_vars]
	if not value_cols:
		raise ValueError("NetRevenue.csv does not contain any value columns to summarize")

	long_df = renamed.melt(
		id_vars=id_vars,
		value_vars=value_cols,
		var_name="netrevenue_component",
		value_name="value",
	)

	long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
	long_df = long_df.dropna(subset=["value"])

	long_df.insert(0, "model", scenario_name)
	long_df.insert(1, "planning_year", int(planning_year))
	long_df.insert(2, "case", case)
	long_df["unit"] = unit

	ordered_cols = ["model", "planning_year", "case", "resource_name"]
	for optional in ["zone", "region", "cluster", "r_id"]:
		if optional in long_df.columns:
			ordered_cols.append(optional)
	ordered_cols.extend(["netrevenue_component", "unit", "value"])

	return long_df[ordered_cols]


__all__ = ["create_netrevenue_summary"]
