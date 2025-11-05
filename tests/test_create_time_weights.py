from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reformat_data_for_plot.create_time_weights import (
    _repeat_weights,
    create_time_weights,
)


def _build_scenario(
    tmp_path: Path,
    *,
    period_map: pd.DataFrame | None = None,
    representative_periods: pd.DataFrame | None = None,
    generators_variability: pd.DataFrame | None = None,
    power: pd.DataFrame | None = None,
) -> tuple[Path, Path, Path]:
    """Create a minimal GenX scenario folder structure for testing."""

    scenario_path = tmp_path / "scenario"
    system_dir = scenario_path / "system"
    results_dir = scenario_path / "results"

    system_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if period_map is not None:
        period_map.to_csv(system_dir / "Period_map.csv", index=False)

    if representative_periods is not None:
        representative_periods.to_csv(
            system_dir / "Representative_Period.csv", index=False
        )

    if generators_variability is not None:
        generators_variability.to_csv(
            system_dir / "Generators_variability.csv", index=False
        )

    if power is not None:
        power.to_csv(results_dir / "power.csv", index=False)

    return scenario_path, system_dir, results_dir


def test_create_time_weights_with_generators_variability(temp_dir: Path) -> None:
    period_map = pd.DataFrame({"Rep_Period_Index": [1, 1, 2, 3]})
    representative_periods = pd.DataFrame({"Dummy": [10, 20, 30]})
    generators_variability = pd.DataFrame(
        {"Time_Index": [1, 2, 3, 4, 5, 6], "Output": [0.0] * 6}
    )

    scenario_path, _, results_dir = _build_scenario(
        temp_dir,
        period_map=period_map,
        representative_periods=representative_periods,
        generators_variability=generators_variability,
    )

    output_path = create_time_weights(scenario_path)

    assert output_path == results_dir / "time_weights.csv"

    weights_df = pd.read_csv(output_path)
    assert weights_df["Time"].tolist() == [1, 2, 3, 4, 5, 6]
    assert weights_df["Weight"].tolist() == [2.0, 2.0, 1.0, 1.0, 1.0, 1.0]


def test_create_time_weights_includes_missing_representative_periods(temp_dir: Path) -> None:
    period_map = pd.DataFrame({"Rep_Period_Index": [1, 1, 2]})
    representative_periods = pd.DataFrame({"Dummy": [10, 20, 30]})
    generators_variability = pd.DataFrame(
        {"Time_Index": [1, 2, 3, 4, 5, 6], "Output": [0.0] * 6}
    )

    scenario_path, _, _ = _build_scenario(
        temp_dir,
        period_map=period_map,
        representative_periods=representative_periods,
        generators_variability=generators_variability,
    )

    output_path = create_time_weights(scenario_path)
    weights_df = pd.read_csv(output_path)

    assert weights_df["Weight"].tolist() == [2.0, 2.0, 1.0, 1.0, 0.0, 0.0]


def test_create_time_weights_uses_power_fallback(temp_dir: Path) -> None:
    period_map = pd.DataFrame({"Rep_Period_Index": [1, 2, 2]})
    representative_periods = pd.DataFrame({"Dummy": [1, 2, 3]})
    power = pd.DataFrame({"Label": ["t1", "t2", "t3"], "R1": [0.0, 0.0, 0.0]})

    scenario_path, _, results_dir = _build_scenario(
        temp_dir,
        period_map=period_map,
        representative_periods=representative_periods,
        power=power,
    )

    output_path = create_time_weights(scenario_path)

    assert output_path == results_dir / "time_weights.csv"

    weights_df = pd.read_csv(output_path)
    assert weights_df["Time"].tolist() == [1, 2, 3]
    assert weights_df["Weight"].tolist() == [1.0, 2.0, 0.0]


def test_create_time_weights_respects_no_overwrite(temp_dir: Path) -> None:
    scenario_path, _, _ = _build_scenario(temp_dir)

    output_path = temp_dir / "custom" / "weights.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("original")

    returned_path = create_time_weights(
        scenario_path,
        output_path=output_path,
        overwrite=False,
    )

    assert returned_path == output_path
    assert output_path.read_text() == "original"


def test_create_time_weights_missing_period_map(temp_dir: Path) -> None:
    representative_periods = pd.DataFrame({"Dummy": [1]})

    scenario_path, _, _ = _build_scenario(
        temp_dir, representative_periods=representative_periods
    )
    (scenario_path / "system" / "Period_map.csv").unlink(missing_ok=True)

    with pytest.raises(FileNotFoundError, match="Period_map.csv"):
        create_time_weights(scenario_path)


def test_create_time_weights_missing_representative_period(temp_dir: Path) -> None:
    period_map = pd.DataFrame({"Rep_Period_Index": [1]})
    scenario_path, _, _ = _build_scenario(temp_dir, period_map=period_map)
    (scenario_path / "system" / "Representative_Period.csv").unlink(missing_ok=True)

    with pytest.raises(FileNotFoundError, match="Representative_Period.csv"):
        create_time_weights(scenario_path)


def test_create_time_weights_empty_representative_period(temp_dir: Path) -> None:
    period_map = pd.DataFrame({"Rep_Period_Index": [1]})
    representative_periods = pd.DataFrame(columns=["Dummy"])

    scenario_path, _, _ = _build_scenario(
        temp_dir,
        period_map=period_map,
        representative_periods=representative_periods,
    )

    with pytest.raises(ValueError, match="Representative_Period.csv is empty"):
        create_time_weights(scenario_path)


def test_create_time_weights_non_divisible_time_slices(temp_dir: Path) -> None:
    period_map = pd.DataFrame({"Rep_Period_Index": [1, 2]})
    representative_periods = pd.DataFrame({"Dummy": [1, 2]})
    generators_variability = pd.DataFrame(
        {"Time_Index": [1, 2, 3, 4, 5], "Output": [0.0] * 5}
    )

    scenario_path, _, _ = _build_scenario(
        temp_dir,
        period_map=period_map,
        representative_periods=representative_periods,
        generators_variability=generators_variability,
    )

    with pytest.raises(ValueError, match="not divisible"):
        create_time_weights(scenario_path)


def test_create_time_weights_generators_variability_missing_time_index(temp_dir: Path) -> None:
    period_map = pd.DataFrame({"Rep_Period_Index": [1]})
    representative_periods = pd.DataFrame({"Dummy": [1]})
    generators_variability = pd.DataFrame({"Bad": [1, 2]})

    scenario_path, _, _ = _build_scenario(
        temp_dir,
        period_map=period_map,
        representative_periods=representative_periods,
        generators_variability=generators_variability,
    )

    with pytest.raises(ValueError, match="Time_Index"):
        create_time_weights(scenario_path)


def test_create_time_weights_power_missing_time_rows(temp_dir: Path) -> None:
    period_map = pd.DataFrame({"Rep_Period_Index": [1, 1]})
    representative_periods = pd.DataFrame({"Dummy": [1, 2]})
    power = pd.DataFrame({"Label": ["summary", "average"], "R1": [0.0, 0.0]})

    scenario_path, _, _ = _build_scenario(
        temp_dir,
        period_map=period_map,
        representative_periods=representative_periods,
        power=power,
    )

    with pytest.raises(ValueError, match="t<index>"):
        create_time_weights(scenario_path)


def test_create_time_weights_missing_results_directory(temp_dir: Path) -> None:
    scenario_path = temp_dir / "scenario"
    system_dir = scenario_path / "system"
    system_dir.mkdir(parents=True)

    period_map = pd.DataFrame({"Rep_Period_Index": [1]})
    period_map.to_csv(system_dir / "Period_map.csv", index=False)
    representative_periods = pd.DataFrame({"Dummy": [1]})
    representative_periods.to_csv(
        system_dir / "Representative_Period.csv", index=False
    )

    with pytest.raises(FileNotFoundError, match="results directory"):
        create_time_weights(scenario_path)


def test_create_time_weights_missing_scenario_path(temp_dir: Path) -> None:
    missing_path = temp_dir / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="Scenario path"):
        create_time_weights(missing_path)


def test_repeat_weights() -> None:
    assert _repeat_weights([1.0, 2.0], 3) == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
