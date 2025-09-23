from pathlib import Path
import shutil
from build_generators_data import build_generators_data
from create_resource_capacity import create_resource_capacity
from create_emissions_summary import create_emissions_summary
from create_generation_summary import create_generation_summary   
from create_dispatch_summary import create_dispatch_summary
import pandas as pd
all_genx_scenarios_path = Path(r"C:\Users\Sriki\MIP_results_comparison-1\genx_results")
scenario_name = "p4_Mod_Elect_Low_RE"


def create_annual_demand_csv(scenario_path: Path, output_path: Path, planning_year: int) -> None:
    """
    Create an annual_demand.csv file from the Demand_data.csv in the GenX scenario.

    Args:
        scenario_path: Path to the GenX scenario folder containing 'system/Demand_data.csv'.
        output_path: Path where the annual_demand.csv will be saved.
        planning_year: Planning year to annotate in the output rows.
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
    print(f"Wrote annual demand CSV to: {output_path}")


def export_genx_for_plotting(scenario_data_path: Path, scenario_name: str, output_folder_path: Path) -> Path:
    """
    Prepare GenX scenario outputs for plotting by copying inputs/results and building summaries.

    Args:
        scenario_data_path: Path to the root folder containing all GenX scenarios (e.g., .../genx_results).
        scenario_name: Name of the scenario folder inside scenario_data_path.
        output_folder_path: Destination root where op_inputs and results_summary will be written.

    Returns:
        Path to the created resource capacity CSV file.
    """
    genx_result_scenario_path = scenario_data_path / scenario_name

    model_name = scenario_name
    op_inputs_path = output_folder_path / f"{model_name}_op_inputs" / "Inputs" / "Inputs_p1"
    op_inputs_path.mkdir(parents=True, exist_ok=True)
    results_subfolder = op_inputs_path / "Results"
    results_subfolder.mkdir(parents=True, exist_ok=True)

    # FROM POLICIES SUBFOLDER ================================================================
    # transfer CO2_cap and CO2_cap_slack from GenX results to output folder
    # Transfer policy files from GenX results to output folder
    policy_files = [
        "CO2_cap.csv",
        "Capacity_reserve_margin.csv",
        "Energy_share_requirement.csv",
        "Minimum_capacity_requirement.csv",
    ]

    for filename in policy_files:
        file_path = genx_result_scenario_path / "policies" / filename
        if file_path.exists():
            print(f"DEBUG: Copying {filename}")
            shutil.copy2(file_path, op_inputs_path)

    print(f"DEBUG: Copied policy files to {op_inputs_path}")

    # FROM SYSTEM SUBFOLDER =============================================================================
    # Define file mappings: (source_name, destination_name)
    system_files = [
        ("Fuels_data.csv", "Fuels_data.csv"),
        ("Demand_data.csv", "Load_data.csv"),
        ("Period_map.csv", "Period_map.csv"),
        ("Network.csv", "Network.csv"),
        ("Representative_periods.csv", "Representative_periods.csv"),
    ]

    for source_name, dest_name in system_files:
        source_file = genx_result_scenario_path / "system" / source_name
        if source_file.exists():
            print(f"DEBUG: Copying {source_name}")
            shutil.copy2(source_file, op_inputs_path / dest_name)

    print(f"DEBUG: Copied system files to {op_inputs_path}")

    # TO THE RESULTS SUBFOLDER ================================================================
    # Copy all CSV files from Results subfolder
    results_files = ["capacityfactor.csv", "costs.csv", "emissions.csv", "nse.csv"]
    for filename in results_files:
        file_path = genx_result_scenario_path / "results" / filename
        if file_path.exists():
            print(f"DEBUG: Copying {filename}")
            shutil.copy2(file_path, results_subfolder)

    # BUILD GENERATORS_DATA.CSV FILE from build_generators_data.py file==========================
    # Create generators_data.csv file
    generators_data_file = op_inputs_path / "Generators_data.csv"
    build_generators_data(genx_result_scenario_path, generators_data_file, debug_overwrites=True)

    #===================================================================================
    # RESULTS SUMMARY CREATION
    results_summary_path = output_folder_path / f"{model_name}_results_summary"
    results_summary_path.mkdir(parents=True, exist_ok=True)

    # Create annual demand CSV in results summary folder (from system/Demand_data.csv)
    try:
        create_annual_demand_csv(
            genx_result_scenario_path,
            results_summary_path / "annual_demand.csv",
            planning_year=2030,
        )
    except Exception as e:
        print(f"Warning: Could not create annual_demand.csv: {e}")

    # Create resource capacity file in results summary folder
    output = create_resource_capacity(
        model_name=model_name,
        case_name="Results_p1",
        scenario_folder_path=op_inputs_path,
        genx_scenario_results_path=genx_result_scenario_path,
        results_summary_folder_path=results_summary_path,
        planning_year=2030,
        unit="MW",
    )
    print(f"Wrote resource capacity CSV to: {output}")

    # Also create emissions summary (uses default case "Results_p1")
    emissions_output = create_emissions_summary(
        genx_scenario_results_path=genx_result_scenario_path,
        scenario_name=model_name,
        output_folder_path=output_folder_path,
        planning_year=2030,
        case="Results_p1",
        unit="tons",
    )
    print(f"Wrote emissions summary CSV to: {emissions_output}")

    # Create generations summary
    generations_output = create_generation_summary(
        genx_scenario_results_path=genx_result_scenario_path,
        scenario_name=model_name,
        output_folder_path=output_folder_path,
        planning_year=2030,
        case="Results_p1",
        unit="MWh",
    )
    print(f"Wrote generations summary CSV to: {generations_output}")

    # Create dispatch summary (filters hours to Weight == 1.0 by default)
    dispatch_output = create_dispatch_summary(
        genx_scenario_results_path=genx_result_scenario_path,
        scenario_name=model_name,
        output_folder_path=output_folder_path,
        planning_year=2030,
        case="Results_p1",
        weight_value=1.0,
    )
    print(f"Wrote dispatch summary CSV to: {dispatch_output}")
    return output


def export_all_genx_scenarios(scenarios_root: Path, output_folder_path: Path) -> dict[str, Path]:
    """
    Iterate through all scenario folders in scenarios_root and export each for plotting.

    Args:
        scenarios_root: Path containing one subfolder per GenX scenario (e.g., .../genx_results).
        output_folder_path: Destination root where outputs will be written.

    Returns:
        Mapping from scenario name to the created resource capacity CSV path (only successful ones).
    """
    scenario_dirs = sorted([p for p in scenarios_root.iterdir() if p.is_dir()])
    print(f"Detected {len(scenario_dirs)} scenario folder(s) in {scenarios_root}")

    results: dict[str, Path] = {}
    total = len(scenario_dirs)
    for idx, scenario_dir in enumerate(scenario_dirs, start=1):
        scenario_name = scenario_dir.name
        print("=" * 80)
        print(f"[{idx}/{total}] Processing scenario '{scenario_name}'...")
        try:
            out_path = export_genx_for_plotting(scenarios_root, scenario_name, output_folder_path)
            results[scenario_name] = out_path
            print(f"\n\n[{idx}/{total}] Completed '{scenario_name}' -> {out_path}")
        except Exception as e:  # keep going on failure
            print(f"\n\n[{idx}/{total}] FAILED '{scenario_name}': {e}")
        print("=" * 80)

    print(f"Finished exporting {len(results)}/{total} scenarios to {output_folder_path}")
    return results


if __name__ == "__main__":
    export_all_genx_scenarios(
        all_genx_scenarios_path,
        Path(r"C:\Users\Sriki\MIP_results_comparison-1\20-week-genx"),
    )
