#%%
'''
Generator Data Builder for GenX Simulations
This module provides functionality to build and consolidate generator data from multiple
resource files in a scenario folder, creating a unified Generators_Data.csv file
compatible with GenX power system modeling.
The module reads various resource files (CSV/Excel) from a scenario's resources folder,
merges their attributes by resource name, applies category mappings, and outputs a
consolidated generators dataset.
Key Features:
- Reads multiple file formats (CSV, Excel) from resources folder
- Merges attributes across files by Resource column
- Maps generator categories (THERM, VRE, STOR, etc.) based on file names
- Incorporates R_ID mapping from NetRevenue.csv results
- Performs column renaming for GenX compatibility
- Optional schema validation against target column structure
Input Requirements:
- scenario_folder_path: Path to scenario folder containing:
    - resources/ directory with generator data files:
        * Thermal.csv (thermal generators)
        * Storage.csv (storage resources)
        * Vre.csv (variable renewable energy)
        * Must_run.csv (must-run units)
        * Hydro.csv (hydro resources)
        * Flex_demand.csv (flexible demand)
        * policy_assignments/ subfolder (optional)
    - results/NetRevenue.csv (optional, for R_ID mapping)
- save_file_path: Output path for consolidated Generators_Data.csv
Expected File Structure:
scenario_folder/
├── resources/
│   ├── Thermal.csv
│   ├── Storage.csv
│   ├── Vre.csv
│   ├── Must_run.csv
│   ├── Hydro.csv
│   ├── Flex_demand.csv
│   ├── policy_assignments/ (optional)
│   └── other_resource_files.csv/xlsx
└── results/
        └── NetRevenue.csv (optional)
Usage Example:
        # Basic usage
                scenario_folder_path="path/to/scenario",
                save_file_path="output/Generators_Data.csv"
        # With schema validation
                scenario_folder_path="path/to/scenario", 
                save_file_path="output/Generators_Data.csv",
                compare_to_target=True
File Requirements:
- All resource files must contain a 'Resource' column with unique generator names
- Files can be in CSV (.csv) or Excel (.xlsx, .xls) format
- Resource names should be consistent across all files for proper merging
Output:
- Consolidated CSV file with all generator attributes
- Resource column as index
- Missing values filled with 0
- Category columns (THERM, VRE, etc.) as binary indicators
- GenX-compatible column naming and structure
'''
import pandas as pd
from pathlib import Path
import re
#%%
# Load the target generator data and columns
target_df_columns = ['Resource', 'region', 'technology', 'cluster', 'R_ID', 'Zone', 'Num_VRE_Bins', 'THERM', 'VRE', 'MUST_RUN', 'STOR', 'FLEX', 'HYDRO', 'LDS', 'CapRes_1', 'CapRes_2', 'Min_Share', 'Max_Share', 'Existing_Cap_MWh', 'Existing_Cap_MW', 'Existing_Charge_Cap_MW', 'num_units', 'unmodified_existing_cap_mw', 'New_Build', 'Cap_Size', 'Min_Cap_MW', 'Max_Cap_MW', 'Max_Cap_MWh', 'Min_Cap_MWh', 'Max_Charge_Cap_MW', 'Min_Charge_Cap_MW', 'Min_Share_percent', 'Max_Share_percent', 'capex_mw', 'Inv_Cost_per_MWyr', 'Fixed_OM_Cost_per_MWyr', 'capex_mwh', 'Inv_Cost_per_MWhyr', 'Fixed_OM_Cost_per_MWhyr', 'Var_OM_Cost_per_MWh', 'Var_OM_Cost_per_MWh_In', 'Inv_Cost_Charge_per_MWyr', 'Fixed_OM_Cost_Charge_per_MWyr', 'Start_Cost_per_MW', 'Start_Fuel_MMBTU_per_MW', 'Heat_Rate_MMBTU_per_MWh', 'heat_rate_mmbtu_mwh_iqr', 'heat_rate_mmbtu_mwh_std', 'Fuel', 'Min_Power', 'Self_Disch', 'Eff_Up', 'Eff_Down', 'Hydro_Energy_to_Power_Ratio', 'Min_Duration', 'Max_Duration', 'Reg_Max', 'Rsv_Max', 'Reg_Cost', 'Rsv_Cost', 'Max_Flexible_Demand_Delay', 'Max_Flexible_Demand_Advance', 'Flexible_Demand_Energy_Eff', 'CO2_Capture_Rate', 'CO2_Capture_Cost_per_Metric_Ton', 'co2_pipeline_annuity_mw', 'co2_pipeline_capex_mw', 'storage_cost_tonne', 'tonne_co2_captured_mwh', 'co2_cost_mwh', 'Ramp_Up_Percentage', 'Ramp_Dn_Percentage', 'Up_Time', 'Down_Time', 'spur_miles', 'spur_capex', 'offshore_spur_miles', 'offshore_spur_capex', 'tx_miles', 'tx_capex', 'interconnect_annuity', 'interconnect_capex_mw', 'regional_cost_multiplier', 'variable_CF', 'RETRO', 'Num_RETRO_Sources', 'Retro1_Source', 'Retro1_Efficiency', 'Retro1_Inv_Cost_per_MWyr', 'Retro2_Source', 'Retro2_Efficiency', 'Retro2_Inv_Cost_per_MWyr', 'MinCapTag_1', 'MinCapTag_2', 'ESR_12', 'Retro3_Efficiency', 'CapRes_7', 'CapRes_9', 'ESR_2', 'ESR_14', 'ESR_10', 'MinCapTag_3', 'ESR_8', 'ESR_1', 'ESR_6', 'gen_is_variable', 'ESR_16', 'ESR_13', 'Retro3_Source', 'CapRes_6', 'ESR_5', 'CapRes_4', 'CapRes_8', 'ESR_15', 'CapRes_5', 'MinCapTag_5', 'ESR_3', 'CapRes_3', 'MinCapTag_4', 'ESR_4', 'ESR_7', 'ESR_9', 'CapRes_10', 'ESR_11', 'Min_Retired_Cap_MW', 'Min_Retired_Energy_Cap_MW', 'Min_Retired_Charge_Cap_MW', 'Capital_Recovery_Period', 'WACC', 'Lifetime', 'old_Inv_Cost_per_MWyr', 'old_Inv_Cost_per_MWhyr', 'original_Fixed_OM_Cost_per_MWyr', 'original_Fixed_OM_Cost_per_MWhyr']
def _read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Excel into DataFrame based on extension."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def build_generators_data(
    scenario_folder_path: str | Path,
    save_file_path: str | Path,
    *,
    compare_to_target: bool = False,
) -> pd.DataFrame:
    """Build and save Generators_Data from resources for a scenario.

    Args:
        scenario_folder_path: Path to scenario folder containing 'resources' and optionally 'results/NetRevenue.csv'.
        save_file_path: Output CSV path to write the combined Generators_Data.
        compare_to_target: If True, prints a column diff vs target_df_columns.

    Returns:
        The combined generators DataFrame.
    """
    scenario_folder = Path(scenario_folder_path)
    save_path = Path(save_file_path)

    resource_path = scenario_folder / "resources"
    if not resource_path.exists():
        raise FileNotFoundError(f"Resource folder not found: {resource_path}")

    resource_files = [
        p for p in resource_path.iterdir() if p.is_file() and p.suffix.lower() in [".csv", ".xlsx", ".xls"]
    ]
    policy_assignment_folder = resource_path / "policy_assignments"
    if policy_assignment_folder.exists():
        resource_files += [
            p for p in policy_assignment_folder.iterdir() if p.is_file() and p.suffix.lower() in [".csv", ".xlsx", ".xls"]
        ]
    print("Files in resources:", [p.name for p in resource_files])

    # Gather unique resource names
    new_gen_data = {}
    resource_names = set()
    for p in resource_files:
        try:
            df = _read_table(p)
        except Exception as e:
            print(f"Warning: Skipping {p.name}: {e}")
            continue
        if 'Resource' not in df.columns:
            print(f"Warning: {p.name} missing 'Resource' column; skipping.")
            continue
        resource_names.update(df['Resource'].dropna().astype(str).tolist())
    print(f"Total unique resources found: {len(resource_names)}")

    for r in resource_names:
        if r not in new_gen_data:
            new_gen_data[r] = {}

    # Merge attributes from each file
    for p in resource_files:
        print(f"Processing file: {p.name}")
        try:
            df = _read_table(p)
        except Exception as e:
            print(f"Warning: Skipping {p.name}: {e}")
            continue
        if 'Resource' not in df.columns:
            print(f"Warning: {p.name} missing 'Resource' column; skipping.")
            continue
        for _, row in df.iterrows():
            resource = str(row['Resource'])
            if resource not in new_gen_data:
                new_gen_data[resource] = {}
            for col, value in row.items():
                if col != 'Resource':
                    new_gen_data[resource][col] = value

    category_dict = {
        "Thermal.csv": "THERM",
        "Storage.csv": "STOR",
        "Vre.csv": "VRE",
        "Must_run.csv": "MUST_RUN",
        "Hydro.csv": "HYDRO",
        "Flex_demand.csv": "FLEX",
    }
    for file in resource_files:
        if file.name in category_dict:
            try:
                df = _read_table(file)
            except Exception as e:
                print(f"Warning: Could not read {file.name} for category mapping: {e}")
                continue
            present = set(df['Resource'].astype(str)) if 'Resource' in df.columns else set()
            for r in resource_names:
                new_gen_data[r][category_dict[file.name]] = 1 if r in present else 0

    # Add R_ID from results/NetRevenue.csv if present
    net_revenue_file = scenario_folder / "results" / "NetRevenue.csv"
    rid_mapping = {}
    if net_revenue_file.exists():
        try:
            net_revenue_df = pd.read_csv(net_revenue_file)
            if 'Resource' in net_revenue_df.columns and 'R_ID' in net_revenue_df.columns:
                for _, row in net_revenue_df.iterrows():
                    rid_mapping[str(row['Resource'])] = row['R_ID']
        except Exception as e:
            print(f"Warning: Failed to read NetRevenue.csv: {e}")
    else:
        print(f"Warning: NetRevenue.csv not found at {net_revenue_file}")

    for r in new_gen_data:
        if r in rid_mapping:
            new_gen_data[r]['R_ID'] = rid_mapping[r]
        # else: leave missing

    # Key renames
    for r in new_gen_data:
        for i in [1, 2, 3]:
            der_key = f'Derating_factor_{i}'
            cap_key = f'CapRes_{i}'
            if der_key in new_gen_data[r]:
                new_gen_data[r][cap_key] = new_gen_data[r].pop(der_key)

    for r in new_gen_data:
        for i in range(1, 10):
            min_cap = f'Min_Cap_{i}'
            min_tag = f'MinCapTag_{i}'
            if min_cap in new_gen_data[r]:
                new_gen_data[r][min_tag] = new_gen_data[r].pop(min_cap)

    # To DataFrame
    new_gen_data_df = pd.DataFrame.from_dict(new_gen_data, orient='index').rename_axis('Resource')
    new_gen_data_df = new_gen_data_df.fillna(0)

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    new_gen_data_df.to_csv(save_path, index=True, encoding="utf-8")
    print(f"Saved combined generators data to: {save_path}")

    # Optional compare
    if compare_to_target:
        target_df = pd.DataFrame(columns=target_df_columns)
        compare_columns(target_df, new_gen_data_df, df_name="combined_generators")

    return new_gen_data_df


#%%
# Check if the combined generators DataFrame has the same columns as the target DataFrame
def compare_columns(target_df, combined_df, df_name="combined_generators", normalize=True, collapse_ws=True):
    target_columns = target_df.columns.tolist()
    combined_columns = combined_df.columns.tolist()
    print(f"\nTarget columns: {target_columns}")
    print(f"\nCombined columns: {combined_columns}")
    if normalize:
        target_columns = [col.strip() for col in target_columns]
        combined_columns = [col.strip() for col in combined_columns]

    if collapse_ws:
        target_columns = [re.sub(r'\s+', ' ', col) for col in target_columns]
        combined_columns = [re.sub(r'\s+', ' ', col) for col in combined_columns]

    missing_cols = set(target_columns) - set(combined_columns)
    extra_cols = set(combined_columns) - set(target_columns)
    # Sort both containers for consistent comparison
    missing_cols = sorted(list(missing_cols))
    extra_cols = sorted(list(extra_cols))
    print(f"\n{df_name} column comparison:")
    print(f"\nMissing columns: {missing_cols}")
    print(f"\nExtra columns: {extra_cols}")

    return missing_cols, extra_cols

# (No top-level execution)
if __name__ == "__main__":
    
    df = build_generators_data(
    r"C:\Users\Sriki\MIP_results_comparison-1\20-week-genx_simulations\p1_High_Elect_Mid_RE",
    r"C:\Users\Sriki\MIP_results_comparison-1\20-week-genx_simulations\GenX_op_inputs\Inputs\Inputs_p1\Generators_Data.csv",
    compare_to_target=True,  # optional schema check
    )