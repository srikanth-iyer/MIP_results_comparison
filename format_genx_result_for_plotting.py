from pathlib import Path
import shutil
from build_generators_data import build_generators_data
from create_resource_capacity import create_resource_capacity
from create_emissions_summary import create_emissions_summary
from create_generation_summary import create_generation_summary   
from create_dispatch_summary import create_dispatch_summary
import pandas as pd



def create_annual_demand_csv(
    scenario_path: Path,
    output_path: Path,
    planning_year: int,
    *,
    verbose: bool = False,
) -> None:
    """
    Create an annual_demand.csv file from the Demand_data.csv in the GenX scenario.

    Args:
        scenario_path: Path to the GenX scenario folder containing 'system/Demand_data.csv'.
        output_path: Path where the annual_demand.csv will be saved.
        planning_year: Planning year to annotate in the output rows.
        verbose: When True, log a message after writing the CSV (defaults to False).
    """
    demand_data_file = scenario_path / "system" / "Demand_data.csv"
    if not demand_data_file.exists():
        raise FileNotFoundError(f"Demand data file not found: {demand_data_file}")

    demand_df = pd.read_csv(demand_data_file)
    zone_cols = [c for c in demand_df.columns if c.startswith("Demand_MW_z")]
    if not zone_cols:
        raise ValueError("No demand zone columns found matching prefix 'Demand_MW_z'")

    rows: list[dict] = []
    for col in zone_cols:
        zone = col.replace("Demand_MW_", "")
        total_demand_mwh = float(demand_df[col].sum())
        rows.append({
            "zone": zone,
            "annual_demand": total_demand_mwh,
            "planning_year": planning_year,
        })

    annual_demand_df = pd.DataFrame(rows, columns=["zone", "annual_demand", "planning_year"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annual_demand_df.to_csv(output_path, index=False)
    if verbose:
        print(f"Wrote annual demand CSV to: {output_path}")


def export_genx_for_plotting(
    scenario_data_path: Path,
    scenario_name: str,
    output_folder_path: Path,
    scenario_to_year_map: dict[str, int] | None = None,
    *,
    debug_overwrites: bool = False,
    verbose: bool = False,
) -> tuple[Path, list[dict[str, str | None]]]:
    """
    Prepare GenX scenario outputs for plotting by copying inputs/results and building summaries.

    Args:
        scenario_data_path: Path to the root folder containing all GenX scenarios (e.g., .../genx_results).
        scenario_name: Name of the scenario folder inside scenario_data_path.
        output_folder_path: Destination root where op_inputs and results_summary will be written.
        scenario_to_year_map: Mapping from period label (e.g., "p1") to planning year.
    debug_overwrites: When True, produce overwrite debug reports from generator builds.
        verbose: When False, suppress informational logging (warnings still print). Defaults to False.

    Returns:
        Tuple of (resource_capacity_csv_path, warning_messages).
    """
    genx_result_scenario_path = scenario_data_path / scenario_name
    model_name = scenario_name

    if scenario_to_year_map is None:
        scenario_to_year_map = {"p1": 2030}
    # Normalize keys for case-insensitive lookup
    scenario_to_year_map = {k.lower(): v for k, v in scenario_to_year_map.items()}

    warnings: list[dict[str, str | None]] = []

    def record_warning(
        message: str,
        *,
        period_name: str | None = None,
        period_key: str | None = None,
    ) -> None:
        normalized = message.strip()
        if normalized.lower().startswith("warning:"):
            normalized = normalized.split(":", 1)[1].strip() if ":" in normalized else normalized

        warnings.append(
            {
                "scenario": model_name,
                "period_name": period_name,
                "period_key": period_key,
                "message": normalized,
            }
        )

        label_parts = [model_name]
        if period_name:
            label_parts.append(period_name)
        elif period_key:
            label_parts.append(period_key)
        label = " / ".join(label_parts)
        print(f"Warning: {label}: {normalized}")

    def _extract_period_key_from_results_name(folder_name: str) -> str | None:
        lower = folder_name.lower()
        for prefix in ("results_", "results-"):
            if lower.startswith(prefix):
                suffix = lower[len(prefix) :].strip()
                return suffix or None
        return None

    def _collect_results_period_dirs(*roots: Path | None) -> dict[str, Path]:
        mapping: dict[str, Path] = {}
        for root in roots:
            if root is None:
                continue
            try:
                children = list(root.iterdir())
            except (FileNotFoundError, PermissionError, NotADirectoryError):
                continue
            for child in children:
                if not child.is_dir():
                    continue
                child_lower = child.name.lower()
                if child_lower == "results":
                    try:
                        sub_children = list(child.iterdir())
                    except (FileNotFoundError, PermissionError, NotADirectoryError):
                        continue
                    for sub in sub_children:
                        if not sub.is_dir():
                            continue
                        key = _extract_period_key_from_results_name(sub.name)
                        if key and key not in mapping:
                            mapping[key] = sub
                else:
                    key = _extract_period_key_from_results_name(child.name)
                    if key and key not in mapping:
                        mapping[key] = child
        return mapping

    def _ensure_period_results_folder(
        period_dir: Path,
        *,
        period_key: str,
        period_name: str,
        results_mapping: dict[str, Path],
    ) -> Path:
        target_results_dir = period_dir / "results"
        source_results_dir = results_mapping.get(period_key)
        if source_results_dir and source_results_dir.exists():
            same_location = False
            if target_results_dir.exists():
                try:
                    same_location = target_results_dir.resolve() == source_results_dir.resolve()
                except (FileNotFoundError, PermissionError):
                    same_location = False
            if not same_location:
                try:
                    if target_results_dir.exists():
                        shutil.rmtree(target_results_dir)
                    shutil.copytree(source_results_dir, target_results_dir)
                    if verbose:
                        print(
                            f"Copied standalone results folder from {source_results_dir} to {target_results_dir}"
                        )
                except Exception as copy_err:
                    record_warning(
                        f"Failed to copy results from {source_results_dir} ({copy_err})",
                        period_name=period_name,
                        period_key=period_key,
                    )
        if not target_results_dir.exists():
            target_results_dir.mkdir(parents=True, exist_ok=True)
        return target_results_dir

    # Discover Inputs_p* period directories. Check the scenario root first,
    # then (case-insensitively) an `Inputs` subfolder if present. Stop at the
    # first parent that contains matching Inputs_p* folders. If none are found,
    # fall back to treating the scenario root as a single-period case.
    # Find an Inputs subfolder if it exists (case-insensitive)
    inputs_folder = None
    for p in genx_result_scenario_path.iterdir():
        if p.is_dir() and p.name.lower() == "inputs":
            inputs_folder = p
            break

    candidate_parents = [genx_result_scenario_path]
    if inputs_folder is not None:
        candidate_parents.append(inputs_folder)

    period_dirs = []
    for parent in candidate_parents:
        period_dirs = sorted(
            [
                p
                for p in parent.iterdir()
                if p.is_dir() and p.name.lower().startswith("inputs_p")
            ],
            key=lambda p: p.name.lower(),
        )
        if period_dirs:
            break

    use_single_period = False
    if not period_dirs:
        period_dirs = [genx_result_scenario_path]
        use_single_period = True

    search_roots: list[Path] = [genx_result_scenario_path]
    if inputs_folder is not None:
        search_roots.append(inputs_folder)
        inputs_parent = inputs_folder.parent
        if inputs_parent != genx_result_scenario_path:
            search_roots.append(inputs_parent)
    results_period_dirs = _collect_results_period_dirs(*search_roots)

    op_inputs_root = output_folder_path / f"{model_name}_op_inputs" / "Inputs"
    results_summary_path = output_folder_path / f"{model_name}_results_summary"
    op_inputs_root.mkdir(parents=True, exist_ok=True)
    results_summary_path.mkdir(parents=True, exist_ok=True)

    policy_files = [
        "CO2_cap.csv",
        "Capacity_reserve_margin.csv",
        "Energy_share_requirement.csv",
        "Minimum_capacity_requirement.csv",
    ]
    system_files = [
        ("Fuels_data.csv", "Fuels_data.csv"),
        ("Demand_data.csv", "Load_data.csv"),
        ("Period_map.csv", "Period_map.csv"),
        ("Network.csv", "Network.csv"),
        ("Representative_periods.csv", "Representative_periods.csv"),
    ]
    results_files = ["capacityfactor.csv", "costs.csv", "emissions.csv", "nse.csv"]

    annual_demand_frames = []
    resource_capacity_frames = []
    emissions_frames = []
    generation_frames = []
    dispatch_frames = []

    for period_dir in period_dirs:
        if use_single_period:
            period_name = "Inputs_p1"
            period_key = "p1"
        else:
            period_name = period_dir.name
            period_identifier = period_name.split("_", 1)[-1]
            period_key = period_identifier.lower()

        planning_year = scenario_to_year_map.get(period_key)
        if planning_year is None:
            record_warning(
                "Skipped because no planning year mapping was found in scenario_to_year_map.",
                period_name=period_name,
                period_key=period_key,
            )
            continue
        print("=" * 40)
        print(f"Processing {period_name} -> planning year {planning_year}")

        op_inputs_path = op_inputs_root / period_name
        op_inputs_path.mkdir(parents=True, exist_ok=True)
        results_subfolder = op_inputs_path / "Results"
        results_subfolder.mkdir(parents=True, exist_ok=True)

        period_results_dir = _ensure_period_results_folder(
            period_dir,
            period_key=period_key,
            period_name=period_name,
            results_mapping=results_period_dirs,
        )

        # Copy policy files specific to this period
        for filename in policy_files:
            file_path = period_dir / "policies" / filename
            if file_path.exists():
                shutil.copy2(file_path, op_inputs_path / filename)

        # Copy select system files for this period
        for source_name, dest_name in system_files:
            source_file = period_dir / "system" / source_name
            if source_file.exists():
                shutil.copy2(source_file, op_inputs_path / dest_name)

        # Copy select results files into Results subfolder
        for filename in results_files:
            file_path = period_results_dir / filename
            if file_path.exists():
                shutil.copy2(file_path, results_subfolder / filename)

        # Build generators data for this period
        generators_data_file = op_inputs_path / "Generators_data.csv"
        try:
            build_generators_data(
                period_dir,
                generators_data_file,
                debug_overwrites=debug_overwrites,
                verbose=verbose,
                warning_callback=lambda msg, p=period_name, k=period_key: record_warning(
                    msg,
                    period_name=p,
                    period_key=k,
                ),
            )
        except Exception as e:
            record_warning(
                f"Could not build Generators_data.csv ({e})",
                period_name=period_name,
                period_key=period_key,
            )

        # Annual demand aggregation
        annual_tmp_path = results_summary_path / f"annual_demand_{period_key}.csv"
        try:
            create_annual_demand_csv(
                period_dir,
                annual_tmp_path,
                planning_year=planning_year,
                verbose=verbose,
            )
            annual_demand_frames.append(pd.read_csv(annual_tmp_path))
        except Exception as e:
            record_warning(
                f"Could not create annual_demand.csv ({e})",
                period_name=period_name,
                period_key=period_key,
            )
        finally:
            if annual_tmp_path.exists():
                try:
                    annual_tmp_path.unlink()
                except Exception:
                    pass

        case_label = f"Results_{period_key}"

        # Resource capacity summary
        try:
            resource_capacity_path = create_resource_capacity(
                model_name=model_name,
                case_name=case_label,
                scenario_folder_path=op_inputs_path,
                genx_scenario_results_path=period_dir,
                results_summary_folder_path=results_summary_path,
                planning_year=planning_year,
                unit="MW",
            )
            resource_capacity_frames.append(pd.read_csv(resource_capacity_path))
        except Exception as e:
            record_warning(
                f"Could not create resource capacity summary ({e})",
                period_name=period_name,
                period_key=period_key,
            )

        # Emissions summary
        try:
            emissions_path = create_emissions_summary(
                genx_scenario_results_path=period_dir,
                scenario_name=model_name,
                output_folder_path=output_folder_path,
                planning_year=planning_year,
                case=case_label,
                unit="tons",
            )
            emissions_frames.append(pd.read_csv(emissions_path))
        except Exception as e:
            record_warning(
                f"Could not create emissions summary ({e})",
                period_name=period_name,
                period_key=period_key,
            )

        # Generation summary
        try:
            generation_path = create_generation_summary(
                genx_scenario_results_path=period_dir,
                scenario_name=model_name,
                output_folder_path=output_folder_path,
                planning_year=planning_year,
                case=case_label,
                unit="MWh",
            )
            generation_frames.append(pd.read_csv(generation_path))
        except Exception as e:
            record_warning(
                f"Could not create generation summary ({e})",
                period_name=period_name,
                period_key=period_key,
            )

        # Dispatch summary
        try:
            dispatch_path = create_dispatch_summary(
                genx_scenario_results_path=period_dir,
                scenario_name=model_name,
                output_folder_path=output_folder_path,
                planning_year=planning_year,
                case=case_label,
                weight_value=1.0,
                verbose=verbose,
            )
            dispatch_frames.append(pd.read_csv(dispatch_path))
        except Exception as e:
            record_warning(
                f"Could not create dispatch summary ({e})",
                period_name=period_name,
                period_key=period_key,
            )

    # Aggregate and overwrite final summary files
    if annual_demand_frames:
        annual_demand = pd.concat(annual_demand_frames, ignore_index=True)
        annual_demand.to_csv(results_summary_path / "annual_demand.csv", index=False)

    if resource_capacity_frames:
        resource_capacity = pd.concat(resource_capacity_frames, ignore_index=True)
        resource_capacity_output = results_summary_path / "resource_capacity.csv"
        resource_capacity.to_csv(resource_capacity_output, index=False)
    else:
        resource_capacity_output = results_summary_path / "resource_capacity.csv"

    if emissions_frames:
        emissions = pd.concat(emissions_frames, ignore_index=True)
        emissions.to_csv(results_summary_path / "emissions.csv", index=False)

    if generation_frames:
        generation = pd.concat(generation_frames, ignore_index=True)
        generation.to_csv(results_summary_path / "generation.csv", index=False)

    if dispatch_frames:
        dispatch = pd.concat(dispatch_frames, ignore_index=True)
        dispatch_path = results_summary_path / "dispatch.csv"
        dispatch.to_csv(dispatch_path, index=False)
        dispatch.to_csv(dispatch_path.with_suffix(".csv.gz"), index=False, compression="gzip")

    return resource_capacity_output, warnings


def export_all_genx_scenarios(
    scenarios_root: Path,
    output_folder_path: Path,
    scenario_to_year_map: dict[str, int] | None = None,
    *,
    debug_overwrites: bool = False,
    verbose: bool = False,
) -> dict[str, Path]:
    """
    Iterate through all scenario folders in scenarios_root and export each for plotting.

    Args:
        scenarios_root: Path containing one subfolder per GenX scenario (e.g., .../genx_results).
        output_folder_path: Destination root where outputs will be written.
        scenario_to_year_map: Optional mapping from period label to planning year.
        debug_overwrites: When True, generator builds write overwrite debug reports.
        verbose: When False, suppress informational logging (warnings still print). Defaults to False.

    Returns:
        Mapping from scenario name to the created resource capacity CSV path (only successful ones).
    """
    scenario_dirs = sorted([p for p in scenarios_root.iterdir() if p.is_dir()])
    print(f"Detected {len(scenario_dirs)} scenario folder(s) in {scenarios_root}")

    results: dict[str, Path] = {}
    warnings_by_scenario: dict[str, list[dict[str, str | None]]] = {}

    def group_warnings_by_period(
        warning_entries: list[dict[str, str | None]]
    ) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for entry in warning_entries:
            period_label = entry.get("period_name") or entry.get("period_key") or "unspecified"
            message = entry.get("message") or ""
            groups.setdefault(str(period_label), []).append(message)
        return groups
    total = len(scenario_dirs)
    for idx, scenario_dir in enumerate(scenario_dirs, start=1):
        scenario_name = scenario_dir.name
        print("=" * 80)
        print(f"[{idx}/{total}] Processing scenario '{scenario_name}'...")
        try:
            out_path, warnings = export_genx_for_plotting(
                scenarios_root,
                scenario_name,
                output_folder_path,
                scenario_to_year_map=scenario_to_year_map,
                debug_overwrites=debug_overwrites,
                verbose=verbose,
            )
            results[scenario_name] = out_path
            if warnings:
                warnings_by_scenario[scenario_name] = warnings
                if verbose:
                    print(f"\n\n[{idx}/{total}] Finished '{scenario_name}' WITH WARNINGS -> {out_path}")
                    print("Warnings summary:")
                    grouped = group_warnings_by_period(warnings)
                    for period_label in sorted(grouped.keys()):
                        print(f"  - {scenario_name} / {period_label}:")
                        for msg in grouped[period_label]:
                            print(f"      • {msg}")
        except Exception as e:  # keep going on failure
            print(f"\n\n[{idx}/{total}] FAILED '{scenario_name}': {e}")
        if verbose:
            print("=" * 80)

    print(f"Finished exporting {len(results)}/{total} scenarios to {output_folder_path}")
    if warnings_by_scenario:
        print("\nSummary of warnings:")
        for scenario_name, scenario_warnings in warnings_by_scenario.items():
            print(f"\n- {scenario_name}:")
            grouped = group_warnings_by_period(scenario_warnings)
            for period_label in sorted(grouped.keys()):
                print(f"  - {period_label}:")
                for msg in grouped[period_label]:
                    print(f"    • {msg}")
    else:
        print("No warnings encountered across processed scenarios.")

    return results


if __name__ == "__main__":

    scenario_to_year_map={
        "p1": 2030,
        "p2": 2040,
        "p3": 2050,
    }
    all_genx_scenarios_path = Path(r"C:\Users\Sriki\MIP_results_comparison-1\genx_scenarios_results")

    export_all_genx_scenarios(
        all_genx_scenarios_path,
        Path(r"C:\Users\Sriki\MIP_results_comparison-1\genx-scenarios"),
        scenario_to_year_map=scenario_to_year_map,
        debug_overwrites=True,
    )

    all_genx_scenarios_20_weeks_path = Path(r"C:\Users\Sriki\MIP_results_comparison-1\genx_scenarios_results_20_weeks")
    export_all_genx_scenarios(
        all_genx_scenarios_20_weeks_path,
        Path(r"C:\Users\Sriki\MIP_results_comparison-1\genx-scenarios-20-weeks"),
        scenario_to_year_map=scenario_to_year_map,
        debug_overwrites=True,
    )
