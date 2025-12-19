


region_map = {
    "NENG_Rest": [1],
    "NY_Z_A": [2],
    "NY_Z_B": [3],
    "NY_Z_C&E": [4],
    "NY_Z_D": [5],
    "NY_Z_F": [6],
    "NY_Z_G-I": [7],
    "NY_Z_J": [8],
    "NY_Z_K": [9],
    "PJM_EMAC": [10],
    "PJM_Rest": [11],
}

# Explicit region color map (edit as needed)
REGION_COLOR_MAP = {
    "NENG_Rest": "#4E79A7",  # blue
    "NY_Z_A": "#F28E2B",     # orange
    "NY_Z_B": "#E15759",     # red
    "NY_Z_C&E": "#76B7B2",   # teal
    "NY_Z_D": "#59A14F",     # green
    "NY_Z_F": "#EDC948",     # yellow
    "NY_Z_G-I": "#B07AA1",   # purple
    "NY_Z_J": "#FF9DA7",     # pink
    "NY_Z_K": "#9C755F",     # brown
    "PJM_EMAC": "#86BCB6",   # light teal
    "PJM_Rest": "#BAB0AC",   # gray
}

TECH_MAP = { # new tech mapping from npatankar
    "offshore_wind_turbine": "Offshore_Wind",
    "offshorewind": "Offshore_Wind", # added by me
    "onshore_wind_turbine": "Onshore_Wind",
    "landbasedwind": "Onshore_Wind", # added by me
    "solar_photovoltaic": "SolarPV",
    "LandbasedWind_Class3_Moderate_": "Onshore_Wind",
    "landbasedwind_class3_moderate_": "Onshore_Wind",
    "UtilityPV_Class1_Moderate_": "SolarPV",
    "utilitypv_class1_moderate_": "SolarPV",
    "utilitypv": "SolarPV", # added by me
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
    "battery_advanced": "Li-ion_Battery", # added by me
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

# TECH_COLOR_MAP = { # updated to align with Kavi color palette
#     # Wind
#     "Offshore_Wind": "#FF00FF",
#     "Onshore_Wind": "#00FF00",

#     # Solar
#     "SolarPV": "#BCBD22",
#     "Distributed_Solar": "#8C564B",

#     # Storage
#     "Li-ion_Battery": "#FFFF00",
#     "Pumped_Storage": "#2E8B9B",

#     # Hydro
#     "Hydro": "#76D7C4",

#     # Other renewables / retrofits
#     "Other_Renewables": "#800080",
#     "Retrofit": "#C2C02B",

#     # Natural gas family
#     "NGCC": "#E377C2",
#     "NGCT": "#D991CE",
#     "NGST": "#EEA9D9",
#     "NGCCS": "#C5B0D5",

#     # Nuclear
#     "Nuclear": "#0000FF",

#     # Zero-carbon fuels / hydrogen
#     "New_zero_carbon_fuel": "#17BECF",

#     # Coal
#     "Coal": "#000000",

#     # Flex / demand
#     "Flex_Demand": "#7F7F7F",
# }
TECH_COLOR_MAP_AND_ORDER = { # color mapping aligned with legacy colors
    "Hydro": "#76D472",
    "Coal": "#FF8900",
    "New_zero_carbon_fuel": "#3DA443",
    "NGST": "#F3CB53",
    "NGCT": "#249A95",
    "NGCC": "#F7CD4B",
    "NGCCS": "#EB9696",
    "Nuclear": "#77BEB6",
    # "Flex_Demand": "#868686",
    "Retrofit": "#B46537E0",
    "Other_Renewables": "#BA9900",
    "SolarPV": "#FF8900",
    "Distributed_Solar": "#FFBC71",
    "Onshore_Wind": "#DB6565",
    "Offshore_Wind": "#FF9797",
    "Pumped_Storage": "#76D472",
    "Li-ion_Battery": "#4379AB",
}

COLOR_MAP = {k: TECH_COLOR_MAP_AND_ORDER[k] for k in list(TECH_COLOR_MAP_AND_ORDER.keys())} # NOTE:  npatankar
TECH_ORDER = list(TECH_COLOR_MAP_AND_ORDER.keys())[::-1]
TECH_STACK_ORDER = {v: i for i, v in enumerate(list(TECH_COLOR_MAP_AND_ORDER.keys())[::-1])}  

SCENARIO_MAPPING_AND_ORDER = {
    "No_action": "No Action",
    "Current_Policy" : "Current Policy",
    "And_No_IRA": "And No IRA",
    "And_Optimistic": "And Optimistic",
    "And_Transmission_Constrained": "And Transmission Constrained",
    "Or_Big_Wires_Act": "Or Big Wires Act",
}

# Ordered list and color map for net revenue components used by plotting helpers
NET_REVENUE_COMPONENT_ORDER = [
    "EnergyRevenue",
    "SubsidyRevenue",
    "RegSubsidyRevenue",
    "ReserveMarginRevenue",
    "OperatingReserveRevenue",
    "OperatingRegulationRevenue",
    "ESRRevenue",
    "RPSRevenue",
    "Charge_cost",
    "Inv_cost_MW",
    "Inv_cost_charge_MW",
    "Inv_cost_MWh",
    "SunkCost",
    "Fixed_OM_cost_MW",
    "Fixed_OM_cost_MWh",
    "Fixed_OM_cost_charge_MW",
    "Var_OM_cost_in",
    "Var_OM_cost_out",
    "Fuel_cost",
    "StartCost",
    "EmissionsCost",
    "CO2SequestrationCost",
    "Cost",
    "Revenue",
    "Profit",
]

NET_REVENUE_COLOR_MAP = {
    "EnergyRevenue": "#7fb3ff",
    "SubsidyRevenue": "#5ad1dd",
    "RegSubsidyRevenue": "#36aebd",
    "ReserveMarginRevenue": "#4f8ecc",
    "OperatingReserveRevenue": "#6bd17a",
    "OperatingRegulationRevenue": "#4ba561",
    "ESRRevenue": "#98d28c",
    "RPSRevenue": "#4cbfc4",
    "Charge_cost": "#f2a541",
    "Inv_cost_MW": "#f47c3c",
    "Inv_cost_charge_MW": "#e86a32",
    "Inv_cost_MWh": "#f6a073",
    "SunkCost": "#f8c291",
    "Fixed_OM_cost_MW": "#f9c74f",
    "Fixed_OM_cost_MWh": "#f9844a",
    "Fixed_OM_cost_charge_MW": "#f8961e",
    "Var_OM_cost_in": "#ffb4a2",
    "Var_OM_cost_out": "#ff7b7b",
    "Fuel_cost": "#d95f02",
    "StartCost": "#c43c29",
    "EmissionsCost": "#6a4c93",
    "CO2SequestrationCost": "#8e6c9d",
    "Cost": "#9d4edd",
    "Revenue": "#0096c7",
    "Profit": "#1791ca",
}
