"""Tests for the create_costs_summary utility."""

from pathlib import Path

import pandas as pd

from reformat_data_for_plot.create_costs_summary import create_costs_summary


def test_create_costs_summary_generates_outputs(real_sample_scenario_1, temp_dir):
    scenario_name = Path(real_sample_scenario_1).name
    period_dir = Path(real_sample_scenario_1) / "inputs" / "inputs_p1"

    summary_df = create_costs_summary(
        genx_scenario_results_path=period_dir,
        scenario_name=scenario_name,
        planning_year=2030,
        case="Results_p1",
        unit="USD",
    )

    expected_columns = {
        "model",
        "planning_year",
        "case",
        "cost_type",
        "zone",
        "unit",
        "value",
    }
    assert set(summary_df.columns) == expected_columns
    assert not summary_df.empty
    assert summary_df["model"].unique().tolist() == [scenario_name]
    assert summary_df["planning_year"].unique().tolist() == [2030]
    assert summary_df["case"].unique().tolist() == ["Results_p1"]
    assert pd.api.types.is_numeric_dtype(summary_df["value"])
