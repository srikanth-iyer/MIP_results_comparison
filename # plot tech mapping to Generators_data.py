# plot tech mapping to Generators_data.csv resource coverage
#%%
from pathlib import Path
import pandas as pd

#%%
def tech_to_type(df: pd.DataFrame) -> pd.DataFrame:
    # Create dictionaries to map unique resource names to their tech types and existence
    tech_type_map = {}
    # existence_map = {}
    for column_str in df.columns:
        if "resource" in column_str.lower():
            resource_column_name = column_str
            break
       # Get unique resource names (as strings) and sort by length
    unique_resource_names = sorted(
        df[resource_column_name].dropna().astype(str).unique().tolist(),
        key=lambda x: len(x)
    )

    # Iterate over each unique resource name and determine its type and existence
    for resource_name in unique_resource_names:
        for tech, tech_type in TECH_MAP.items():
            if tech.lower() in resource_name.lower():
                tech_type_map[resource_name] = tech_type
                # existence_map[resource_name] = existing
                break
        else:  # If no tech match, set to default
            tech_type_map[resource_name] = "Not Specified"
            # existence_map[resource_name] = False

    # Special case for 'unserved_load'
    if "unserved_load" in tech_type_map:
        tech_type_map["unserved_load"] = "Other"

    # Map the tech_type and existence to the DataFrame using vectorized operations
    df["tech_type"] = df[resource_column_name].map(tech_type_map)
    # df["existing"] = df["resource_name"].map(existence_map)
    # df["new_build"] = ~df["existing"]

    return df

TECH_MAP = { # new tech mapping from npatankar
    "offshore_wind_turbine": "Offshore_Wind",
    # "offshorewind": "Offshore_Wind", # added by me
    "onshore_wind_turbine": "Onshore_Wind",
    # "landbasedwind": "Onshore_Wind", # added by me
    "solar_photovoltaic": "SolarPV",
    "LandbasedWind_Class3_Moderate_": "Onshore_Wind",
    "landbasedwind_class3_moderate_": "Onshore_Wind",
    "UtilityPV_Class1_Moderate_": "SolarPV",
    "utilitypv_class1_moderate_": "SolarPV",
    # "utilitypv": "SolarPV", # added by me
    "OffShoreWind_Class3_Moderate_fixed_1": "Offshore_Wind",
    "offshorewind_class3_moderate_fixed_1": "Offshore_Wind",
    "OffShoreWind_Class12_Moderate_floating_1": "Offshore_Wind",
    "offshorewind_class12_moderate_floating_1": "Offshore_Wind",
    "OffShoreWind_Class3_Moderate_fixed_0": "Offshore_Wind",
    "offshorewind_class3_moderate_fixed_0": "Offshore_Wind",
    "OffShoreWind_Class12_Moderate_floating_0": "Offshore_Wind",
    "offshorewind_class12_moderate_floating_0": "Offshore_Wind",
    "Batteries": "Li-ion_Battery",
    "batteries": "Li-ion_Battery",
    "hydroelectric_pumped_storage": "Pumped_Storage",
    "Battery_*_Moderate": "Li-ion_Battery",
    "battery_*_moderate": "Li-ion_Battery",
    "battery_moderate": "Li-ion_Battery",
    # "battery_advanced": "Li-ion_Battery", # added by me
    "Biomass": "Other_Renewables",
    "biomass": "Other_Renewables",
    "small_hydroelectric": "Other_Renewables",
    "natural_gas_fired_combined_cycle": "NGCC",
    "natural_gas_fired_combustion_turbine": "NGCT",
    "Nuclear": "Nuclear",
    "nuclear": "Nuclear",
    "natural_gas_steam_turbine": "NGST",
    "conventional_steam_coal": "Coal",
    "naturalgas_hframe_cc_95_ccs_moderate": "NGCCS",
    "naturalgas_hframe_cc_moderate": "NGCC",
    "naturalgas_fframe_ct_moderate": "NGCT",
    "Nuclear_Nuclear_Moderate": "Nuclear",
    "nuclear_nuclear_moderate": "Nuclear",
    "hydrogen_fframe_ct_moderate": "New_zero_carbon_fuel",
    "hydrogen_hframe_cc_moderate": "New_zero_carbon_fuel",
    "conventional_hydroelectric": "Hydro",
    "distributed_generation": "Distributed_Solar",
    "res_water_heat": "Flex_Demand",
    "trans_light_duty": "Flex_Demand",
    "space_heat": "Flex_Demand",
    "water_heat": "Flex_Demand",
    "retrofit" : "Retrofit",
}
# iterate over scenarios and years to read Generators_data.csv, map tech types, and print counts of "Not Specified"
incomplete_coverage = []
incomplete_df = pd.DataFrame()
for scenario in ['And_No_IRA_op_inputs','And_Optimistic_op_inputs','And_Transmission_Constrained_op_inputs', 'Current_Policy_op_inputs','No_action_op_inputs']:
    scenario_path = Path(f"genx-scenarios/{scenario}/")
    for year in ['p1','p2','p3','p4']:
        
        generators_data_path = scenario_path / "Inputs" / f"inputs_{year}" / "Generators_data.csv"
        gen_data_df = pd.read_csv(generators_data_path)
        # print(gen_data_df["Resource"].unique())
        tech_to_type_df = tech_to_type(gen_data_df)
        # print(tech_to_type_df['tech_type'].value_counts())
        not_specified_count = tech_to_type_df[tech_to_type_df['tech_type'] == "Not Specified"].shape[0]
        if not_specified_count > 0:
            print(scenario, year, not_specified_count)
            incomplete_coverage.append((scenario, year, not_specified_count))

            # Show rows with Not Specified tech type; use print for plain script environments
            # print(tech_to_type_df.loc[tech_to_type_df['tech_type'] == "Not Specified", ['Resource','tech_type']].to_string(index=False))
            incomplete_rows = tech_to_type_df.loc[
                tech_to_type_df['tech_type'] == "Not Specified",
                ['Resource', 'tech_type']
            ].copy()
            incomplete_rows['scenario'] = scenario
            incomplete_rows['year'] = year
            incomplete_df = pd.concat([incomplete_df, incomplete_rows], ignore_index=True)

incomplete_df.to_csv("incomplete_tech_mapping.csv", index=False)


# %%

    