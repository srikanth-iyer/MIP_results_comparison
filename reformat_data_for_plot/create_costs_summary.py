"""Create tidy cost summaries for GenX scenario periods."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def create_costs_summary(
	genx_scenario_results_path: Union[str, Path],
	scenario_name: str,
	planning_year: int,
	*,
	case: str = "Results_p1",
	unit: str = "USD",
) -> pd.DataFrame:
	"""Build a long-form costs summary for a single planning year.

	Parameters
	----------
	genx_scenario_results_path:
		Path to the period directory that contains a ``results/costs.csv`` file.
	scenario_name:
		Scenario identifier stored in the ``model`` column of the summary output.
	planning_year:
		Planning year associated with the current period (e.g., 2030).
	case:
		Label identifying the period/case (default ``Results_p1``).
	unit:
		Cost unit to store in the summary (default ``USD``).

	Returns
	-------
	pd.DataFrame
		Long-form summary dataframe with cost breakdown by zone.
	"""

	results_path = Path(genx_scenario_results_path)
	source_costs_path = results_path / "results" / "costs.csv"
	if not source_costs_path.exists():
		raise FileNotFoundError(f"Missing costs.csv at: {source_costs_path}")

	df = pd.read_csv(source_costs_path)
	if df.empty:
		raise ValueError(f"costs.csv at {source_costs_path} is empty")

	identifier_col = df.columns[0]
	if identifier_col.lower() not in {"costs", "cost", "category"}:
		# Continue but rename to a standard column so downstream logic is predictable.
		pass

	renamed = df.rename(columns={identifier_col: "cost_type"})
	long_df = renamed.melt(id_vars=["cost_type"], var_name="zone", value_name="value")

	long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
	long_df = long_df.dropna(subset=["value"])  # Drop placeholder '-' or blank entries

	long_df.insert(0, "model", scenario_name)
	long_df.insert(1, "planning_year", int(planning_year))
	long_df.insert(2, "case", case)
	long_df["unit"] = unit

	summary_columns = ["model", "planning_year", "case", "cost_type", "zone", "unit", "value"]
	summary_df = long_df[summary_columns]

	return summary_df


__all__ = ["create_costs_summary"]

