


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
TECH_MAP = {
# TECH_MAP_original = { 
    "batteries": "Battery",
    "battery": "Battery",
    "biomass_": "Other",
    "conventional_hydroelectric": "Hydro",
    "hydroelectric": "Hydro",
    "conventional_steam_coal": "Coal",
    "conventional steam coal": "Coal",
    "geothermal": "Geothermal",
    "natural_gas_fired_combined_cycle": "Natural Gas CC",
    "natural gas fired combined cycle": "Natural Gas CC",
    "natural_gas_fired_combustion_turbine": "Natural Gas CT",
    "natural gas fired combustion turbine": "Natural Gas CT",
    "natural_gas_internal_combustion_engine": "Natural Gas Other",
    "natural gas internal combustion engine": "Natural Gas Other",
    "natural_gas_steam_turbine": "Natural Gas Other",
    "natural gas steam turbine": "Natural Gas Other",
    "onshore_wind_turbine": "Wind",
    "onshore wind": "Wind",
    "petroleum_liquids": "Other",
    "small_hydroelectric": "Hydro",
    "solar_photovoltaic": "Solar",
    "photovoltaic": "Solar",
    "hydroelectric_pumped_storage": "Hydro",
    "nuclear_nuclear": "Nuclear",
    "nuclear_1": "Nuclear",
    "nuclear": "Nuclear",
    "offshore_wind_turbine": "Wind",
    "distributed_generation": "Distributed Solar",
    "naturalgas_ccavgcf": "Natural Gas CC",
    "NaturalGas_HFrame_CC_moderate": "Natural Gas CC",
    "naturalgas_ctavgcf": "Natural Gas CT",
    "NaturalGas_FFrame_CT": "Natural Gas CT",
    "landbasedwind": "Wind",
    "utilitypv": "Solar",
    "naturalgas_ccccsavgcf": "CCS",
    "ccs": "CCS",
    "offshorewind": "Wind",
    "offshore wind": "Wind",
    "hydrogen": "Hydrogen",
    "res_water_heat": "Flex Demand", 
    "trans_light_duty": "Flex Demand", 
    "space_heat": "Flex Demand", 
    "water_heat": "Flex Demand", 
}
_COLOR_MAP = { # NOTE: temporarily disabled to use the new tech maps
    "Battery": "#4379AB",
    "CCS": "#96CCEB",
    "Coal": "#FF8900",
    "Distributed Solar": "#FFBC71",
    "Geothermal": "#3DA443",
    "Hydro": "#76D472",
    "Hydrogen": "#BA9900",
    "Natural Gas CC": "#F7CD4B",
    "Natural Gas CT": "#249A95",
    "Nuclear": "#77BEB6",
    "Other": "#000000", 
    "Solar": "#C44AF1",
    "Wind": "#FF9797",
    "Flex Demand": "#868686",
}
_COLOR_MAP_from_kavi={
    "Other_Renewables": "#800080",  
    "Existing_NG": "#ff7f0e",      
    "None": "#FF0000",            
    "Nuclear": "#0000FF",          
    "Existing_Wind": "#9467bd",    
    "Existing_Solar": "#8c564b",  
    "NG": "#e377c2",              
    "Battery": "#FFFF00",          
    "NG_CCS": "#c5b0d5",            
    "Onshore_Wind": "#00FF00",      
    "Offshore_Wind": "#FF00FF",    
    "Utility_PV": "#bcbd22",        
}
TECH_ORDER = [
    "Other",
    # "Flex Demand", # NOTE: temporarily disabled because we don't want to plot Flex Demand
    "Nuclear",
    "CCS",
    "Natural Gas CC",
    "Natural Gas CT",
    "Coal",
    "Geothermal",
    "Hydro",
    "Distributed Solar",
    "Solar",
    "Wind",
    "Hydrogen",
    "Battery",
]
COLOR_MAP = {k: _COLOR_MAP[k] for k in TECH_ORDER[::-1]}
TECH_STACK_ORDER = {v: i for i, v in enumerate(TECH_ORDER)} 
SCENARIO_MAPPING_AND_ORDER = {
    "No_action": "No Action",
    "Current_Policy" : "Current Policy",
    "And_No_IRA": "And No IRA",
    "And_Optimistic": "And Optimistic",
    "And_Transmission_Constrained": "And Transmission Constrained",
}