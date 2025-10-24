from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import pandas as pd

PathLike = Union[str, Path]


def _load_period_map(system_dir: Path) -> pd.DataFrame:
    period_map_path = system_dir / "Period_map.csv"
    if not period_map_path.exists():
        raise FileNotFoundError(
            "Unable to rebuild time weights because Period_map.csv was not found at "
            f"{period_map_path}"
        )

    period_map = pd.read_csv(period_map_path)
    required_columns = {"Rep_Period_Index"}
    missing_columns = required_columns.difference(period_map.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Period_map.csv must contain the following columns to rebuild time weights: "
            f"{missing}"
        )
    return period_map


def _load_representative_periods(system_dir: Path) -> pd.DataFrame:
    rep_period_path = system_dir / "Representative_Period.csv"
    if not rep_period_path.exists():
        raise FileNotFoundError(
            "Unable to rebuild time weights because Representative_Period.csv was not found at "
            f"{rep_period_path}"
        )

    rep_periods = pd.read_csv(rep_period_path)
    if rep_periods.empty:
        raise ValueError(
            "Representative_Period.csv is empty; cannot infer representative periods"
        )

    rep_periods = rep_periods.reset_index(drop=True)
    rep_periods.index = rep_periods.index + 1  # 1-indexed to match Rep_Period_Index
    rep_periods.index.name = "Rep_Period_Index"
    return rep_periods


def _infer_total_time_slices(system_dir: Path, results_dir: Path) -> int:
    """Infer the total number of time slices present in the scenario results."""

    generators_variability_path = system_dir / "Generators_variability.csv"
    if generators_variability_path.exists():
        variability = pd.read_csv(generators_variability_path)
        if "Time_Index" not in variability.columns:
            raise ValueError(
                "Generators_variability.csv exists but does not contain a 'Time_Index' column"
            )
        return int(variability["Time_Index"].nunique())

    power_path = results_dir / "power.csv"
    if power_path.exists():
        power_df = pd.read_csv(power_path)
        if power_df.empty:
            raise ValueError("power.csv is empty; cannot infer time slices")
        first_col = power_df.columns[0]
        mask = power_df[first_col].astype(str).str.match(r"^t\d+$", na=False)
        time_rows = power_df.loc[mask]
        if time_rows.empty:
            raise ValueError(
                "power.csv does not contain rows labelled t<index>; cannot infer time slices"
            )
        return int(len(time_rows))

    raise FileNotFoundError(
        "Unable to infer the total number of time slices. Neither Generators_variability.csv "
        "nor power.csv are available."
    )


def _repeat_weights(weights: Iterable[float], repeats: int) -> list[float]:
    values: list[float] = []
    for weight in weights:
        values.extend([float(weight)] * repeats)
    return values


def create_time_weights(
    genx_scenario_results_path: PathLike,
    *,
    output_path: PathLike | None = None,
    overwrite: bool = True,
    verbose: bool = False,
) -> Path:
    """Rebuild the GenX ``time_weights.csv`` file from representative-period data."""

    scenario_path = Path(genx_scenario_results_path)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario path does not exist: {scenario_path}")

    system_dir = scenario_path / "system"
    results_dir = scenario_path / "results"
    if not system_dir.is_dir():
        raise FileNotFoundError(f"Scenario system directory not found: {system_dir}")
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Scenario results directory not found: {results_dir}")

    if output_path is None:
        output_path = results_dir / "time_weights.csv"
    else:
        output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        if verbose:
            print(f"time_weights.csv already exists; leaving in place: {output_path}")
        return output_path

    period_map = _load_period_map(system_dir)
    rep_periods = _load_representative_periods(system_dir)

    counts = period_map["Rep_Period_Index"].value_counts()
    # Ensure deterministic ordering and include missing representative periods
    counts = counts.sort_index()

    total_time_slices = _infer_total_time_slices(system_dir, results_dir)
    number_of_representative_periods = len(rep_periods)
    if number_of_representative_periods == 0:
        raise ValueError("No representative periods discovered; cannot build time weights")

    if total_time_slices % number_of_representative_periods != 0:
        raise ValueError(
            "Total number of time slices is not divisible by the number of representative "
            "periods. Cannot evenly allocate weights."
        )

    time_slices_per_rep_period = total_time_slices // number_of_representative_periods

    weights_per_rep = [
        float(counts.get(rep_index, 0.0))
        for rep_index in range(1, number_of_representative_periods + 1)
    ]

    weights = _repeat_weights(weights_per_rep, time_slices_per_rep_period)
    if len(weights) != total_time_slices:
        raise AssertionError(
            "Internal error while constructing time weights: generated "
            f"{len(weights)} entries, expected {total_time_slices}."
        )

    weights_df = pd.DataFrame(
        {
            "Time": range(1, total_time_slices + 1),
            "Weight": weights,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights_df.to_csv(output_path, index=False)

    if verbose:
        print(f"Wrote time weights to: {output_path}")

    return output_path


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebuild time_weights.csv from representative-period metadata."
    )
    parser.add_argument(
        "scenario_path",
        type=Path,
        help="Path to the GenX scenario directory containing system/ and results/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional explicit output path for the generated time_weights.csv.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite an existing output file; simply return its path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output.",
    )

    args = parser.parse_args()
    create_time_weights(
        args.scenario_path,
        output_path=args.output,
        overwrite=not args.no_overwrite,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    _main()
