"""Shared configuration for GenX figure functions.

This module centralizes static configuration such as region groupings,
technology mappings, and color palettes so they can be reused across
notebook utilities and chart-building helpers. Keeping these definitions
in one location makes it easier to audit and extend the visualization
suite while keeping the main logic modules lean.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# ------------------------------
# Region mappings and metadata
# ------------------------------

REGION_MAP: Dict[str, List[int]] = {
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

REGION_COLOR_MAP: Dict[str, str] = {
    "NENG_Rest": "#4E79A7",  # blue
    "NY_Z_A": "#F28E2B",  # orange
    "NY_Z_B": "#E15759",  # red
    "NY_Z_C&E": "#76B7B2",  # teal
    "NY_Z_D": "#59A14F",  # green
    "NY_Z_F": "#EDC948",  # yellow
    "NY_Z_G-I": "#B07AA1",  # purple
    "NY_Z_J": "#FF9DA7",  # pink
    "NY_Z_K": "#9C755F",  # brown
    "PJM_EMAC": "#86BCB6",  # light teal
    "PJM_Rest": "#BAB0AC",  # gray
}

# ------------------------------
# Technology taxonomy
# ------------------------------

TECH_MAP: Dict[str, str] = {
    "batteries": "Battery",
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
    "battery": "Battery",
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

EXISTING_TECH_MAP: Dict[str, str] = {
    "batteries": "Battery",
    "biomass_": "Other",
    "conventional_hydroelectric": "Hydro",
    "conventional_steam_coal": "Coal",
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
    "petroleum_liquids": "Other",
    "small_hydroelectric": "Hydro",
    "solar_photovoltaic": "Solar",
    "hydroelectric_pumped_storage": "Hydro",
    "nuclear_1": "Nuclear",
    "offshore_wind_turbine": "Wind",
    "distributed_generation": "Distributed Solar",
}

_COLOR_MAP: Dict[str, str] = {
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

TECH_ORDER: List[str] = [
    "Other",
    "Flex Demand",
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

COLOR_MAP: Dict[str, str] = {tech: _COLOR_MAP[tech] for tech in reversed(TECH_ORDER)}
TECH_STACK_ORDER: Dict[str, int] = {tech: idx for idx, tech in enumerate(TECH_ORDER)}

WRAPPED_CASE_NAME_MAP: Dict[str, str] = {
    "genx-scenarios": "GenX\nScenarios",
}

LINE_NAMES: List[str] = [
    # Placeholder for future line list extension. Define explicit names here
    # when the transmission routines need to filter by known lines.
]

DATA_COLS = {
    "resource_capacity": [
        "model",
        "zone",
        "agg_zone",
        "planning_year",
        "unit",
        "tech_type",
        "resource_name",
        "start_value",
        "end_value",
        "new_build",
        "existing",
    ],
    "generation": [
        "model",
        "zone",
        "agg_zone",
        "planning_year",
        "tech_type",
        "resource_name",
        "value",
    ],
    "transmission": [
        "model",
        "line",
        "line_name",
        "planning_year",
        "unit",
        "start_value",
        "end_value",
    ],
    "transmission_expansion": [
        "model",
        "line",
        "line_name",
        "planning_year",
        "unit",
        "value",
    ],
    "emissions": [
        "model",
        "zone",
        "agg_zone",
        "planning_year",
        "unit",
        "value",
    ],
    "dispatch": [
        "model",
        "resource_name",
        "tech_type",
        "zone",
        "agg_zone",
        "hour",
        "planning_year",
        "value",
    ],
    "capacityfactor": [
        "resource_name",
        "tech_type",
        "zone",
        "agg_zone",
        "AnnualSum",
        "Capacity",
        "CapacityFactor",
        "value",
        "zone",
        "agg_zone",
    ],
    "nse": ["zone", "agg_zone", "value"],
    "costs": ["Costs", "Total"],
    "emissions_plant": ["resource_name", "tech_type", "value", "zone"],
}


@dataclass(frozen=True)
class TechMetadata:
    """Convenience container for mapping technology labels to colors and order."""

    name: str
    display_order: int
    color: str


def iter_tech_metadata() -> List[TechMetadata]:
    """Return technology metadata tuples ordered for stacking convenience."""

    return [
        TechMetadata(name=tech, display_order=TECH_STACK_ORDER[tech], color=COLOR_MAP[tech])
        for tech in TECH_ORDER
    ]


def reverse_region_map() -> Dict[int, str]:
    """Return a reverse lookup of `REGION_MAP` for assigning aggregate zones."""

    return {zone: region for region, zones in REGION_MAP.items() for zone in zones}
