"""Technology-map coverage validation for GenX Generators_data files."""

from pathlib import Path
import sys
from typing import Callable, Iterable, Tuple

import pandas as pd


_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parent
_CONFIG_DIR = _REPO_ROOT / "genx-notebooks"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

from fig_functions_genx_only import tech_to_type  # type: ignore  # noqa: E402


GeneratorsEntry = Tuple[Path, str | None, str | None, str | None]


def check_tech_map_coverage(
    generators_entries: Iterable[GeneratorsEntry],
    *,
    output_csv: Path | None = None,
    warning_callback: Callable[[str, str | None, str | None, str | None], None] | None = None,
) -> pd.DataFrame | None:
    """Check whether every resource in Generators_data files maps to a technology.

    Args:
        generators_entries: Iterable of tuples containing
            (generators_csv_path, scenario_name, period_name, period_key).
        output_csv: Optional path where uncovered-resource rows are written.
        warning_callback: Optional callable receiving (message, scenario, period_name, period_key).

    Returns:
        A DataFrame of uncovered rows when any are found; otherwise, ``None``.
    """

    uncovered_frames: list[pd.DataFrame] = []

    for generators_path, scenario, period_name, period_key in generators_entries:
        if not generators_path.exists():
            if warning_callback:
                warning_callback(
                    f"Generators data file not found at {generators_path}",
                    scenario,
                    period_name,
                    period_key,
                )
            continue

        try:
            generators_df = pd.read_csv(generators_path)
        except Exception as read_err:  # pragma: no cover - defensive
            if warning_callback:
                warning_callback(
                    f"Tech map coverage check skipped due to read error: {read_err}",
                    scenario,
                    period_name,
                    period_key,
                )
            continue

        try:
            tech_df = tech_to_type(generators_df)
        except Exception as mapping_err:  # pragma: no cover - defensive
            if warning_callback:
                warning_callback(
                    f"Tech map coverage check failed: {mapping_err}",
                    scenario,
                    period_name,
                    period_key,
                )
            continue

        if "tech_type" not in tech_df.columns or "resource_name" not in tech_df.columns:
            if warning_callback:
                warning_callback(
                    "Tech map coverage check requires 'resource_name' and 'tech_type' columns.",
                    scenario,
                    period_name,
                    period_key,
                )
            continue

        missing = tech_df.loc[tech_df["tech_type"] == "Not Specified", ["resource_name", "tech_type"]].copy()
        if missing.empty:
            continue

        missing.insert(0, "scenario", scenario)
        missing["period_name"] = period_name
        missing["period_key"] = period_key
        missing["generators_data_csv"] = str(generators_path)
        uncovered_frames.append(missing)

    if not uncovered_frames:
        if output_csv and output_csv.exists():
            try:
                output_csv.unlink()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        return None

    uncovered_df = pd.concat(uncovered_frames, ignore_index=True)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        uncovered_df.to_csv(output_csv, index=False)

    if warning_callback:
        for row in uncovered_df[["scenario", "period_name", "period_key"]].drop_duplicates().itertuples(index=False):
            scenario_name, period_name, period_key = row
            remaining = uncovered_df[
                (uncovered_df["scenario"] == scenario_name)
                & (uncovered_df["period_name"] == period_name)
                & (uncovered_df["period_key"] == period_key)
            ]
            warning_callback(
                f"Technology map missing for {len(remaining)} resource(s)",
                scenario_name,
                period_name,
                period_key,
            )

    return uncovered_df


def _iter_generators_files_for_workspace() -> list[GeneratorsEntry]:
    root = _REPO_ROOT / "genx-scenarios"
    scenario_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and p.name.endswith("_op_inputs")
    )

    entries: list[GeneratorsEntry] = []
    for scenario_path in scenario_dirs:
        scenario_name = scenario_path.name
        for period_key in ["p1", "p2", "p3", "p4"]:
            generators_path = (
                scenario_path
                / "Inputs"
                / f"inputs_{period_key}"
                / "Generators_data.csv"
            )
            period_name = f"Inputs_{period_key}" if generators_path.parent.exists() else None
            entries.append((generators_path, scenario_name, period_name, period_key))
    return entries


def _default_warning(message: str, scenario: str | None, period_name: str | None, period_key: str | None) -> None:
    parts = ["Tech map coverage"]
    if scenario:
        parts.append(str(scenario))
    if period_name:
        parts.append(str(period_name))
    elif period_key:
        parts.append(str(period_key))
    context = " / ".join(parts)
    print(f"Warning: {context}: {message}")


def main() -> None:
    entries = _iter_generators_files_for_workspace()
    if not entries:
        print("No generator data files discovered for tech-map coverage check.")
        return

    output_csv = _MODULE_DIR / "incomplete_tech_mapping.csv"
    uncovered = check_tech_map_coverage(
        entries,
        output_csv=output_csv,
        warning_callback=_default_warning,
    )

    if uncovered is None:
        print("Tech map covers all resources across scanned scenarios.")
    else:
        print(f"Wrote {len(uncovered)} uncovered resource row(s) to {output_csv}")


if __name__ == "__main__":
    main()
