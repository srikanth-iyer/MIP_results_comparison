"Calculate current, additional, and retired capacity"

from pathlib import Path

import pandas as pd

parent_dir = Path(__file__).parent.parent
TECH_MAP = {
    "batteries": "Battery",
    "biomass_": "Other",
    "conventional_hydroelectric": "Hydro",
    "conventional_steam_coal": "Coal",
    "geothermal": "Geothermal",
    "natural_gas_fired_combined_cycle": "Natural Gas CC",
    "natural_gas_fired_combustion_turbine": "Natural Gas CT",
    "natural_gas_internal_combustion_engine": "Natural Gas Other",
    "natural_gas_steam_turbine": "Natural Gas Other",
    "onshore_wind_turbine": "Wind",
    "petroleum_liquids": "Other",
    "small_hydroelectric": "Hydro",
    "solar_photovoltaic": "Solar",
    "hydroelectric_pumped_storage": "Hydro",
    "nuclear": "Nuclear",
    "nuclear_1": "Nuclear",
    "offshore_wind_turbine": "Wind",
    "distributed_generation": "Distributed Solar",
    "naturalgas_ccavgcf": "Natural Gas CC",
    "NaturalGas_HFrame_CC": "Natural Gas CC",
    "naturalgas_ctavgcf": "Natural Gas CT",
    "NaturalGas_FFrame_CT": "Natural Gas CT",
    "battery": "Battery",
    "landbasedwind": "Wind",
    "utilitypv": "Solar",
    "naturalgas_ccccsavgcf": "CCS",
    "ccs": "CCS",
    "offshorewind": "Wind",
    "hydrogen": "Hydrogen",
}

EXISTING_TECH_MAP = {
    "batteries": "Battery",
    "biomass_": "Other",
    "conventional_hydroelectric": "Hydro",
    "conventional_steam_coal": "Coal",
    "geothermal": "Geothermal",
    "natural_gas_fired_combined_cycle": "Natural Gas CC",
    "natural_gas_fired_combustion_turbine": "Natural Gas CT",
    "natural_gas_internal_combustion_engine": "Natural Gas Other",
    "natural_gas_steam_turbine": "Natural Gas Other",
    "onshore_wind_turbine": "Wind",
    "petroleum_liquids": "Other",
    "small_hydroelectric": "Hydro",
    "solar_photovoltaic": "Solar",
    "hydroelectric_pumped_storage": "Hydro",
    "nuclear_1": "Nuclear",
    "offshore_wind_turbine": "Wind",
    "distributed_generation": "Distributed Solar",
}

_TECH_MAP = {}
for k, v in TECH_MAP.items():
    if k in EXISTING_TECH_MAP.keys():
        _TECH_MAP[k] = (v, True)
    else:
        _TECH_MAP[k] = (v, False)


def tech_to_type(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, "existing"] = False
    df.loc[:, "tech_type"] = "Not Specified"
    for tech, (t, ex) in _TECH_MAP.items():
        df.loc[
            df["resource_name"].str.contains(tech, case=False),
            ["tech_type", "existing"],
        ] = [
            t,
            ex,
        ]
    df.loc[df["resource_name"] == "unserved_load", "tech_type"] = "Other"

    return df


def group_sum_capacity(cap: pd.DataFrame) -> pd.DataFrame:
    "Some models report capacity at the generator level"

    return cap.groupby(
        ["planning_year", "model", "zone", "tech_type", "resource_name", "unit"],
        as_index=False,
    )[["start_value", "end_value"]].sum()


def set_start_cap(cap: pd.DataFrame) -> pd.DataFrame:

    p1_start_cap = pd.read_csv(
        parent_dir / "bin" / "Generators_data.csv",
        index_col="Resource",
        usecols=["Resource", "Existing_Cap_MW", "Existing_Cap_MWh"],
    )

    years = cap.planning_year.unique()
    cap = cap.set_index("resource_name")
    cap.loc[
        (cap["planning_year"] == years[0]) & (cap["unit"] == "MW") & (cap["existing"]),
        "start_value",
    ] = p1_start_cap["Existing_Cap_MW"]
    cap.loc[
        (cap["planning_year"] == years[0]) & (cap["unit"] == "MWh") & (cap["existing"]),
        "start_value",
    ] = p1_start_cap["Existing_Cap_MWh"]

    for prev_year, year in zip(years[:-1], years[1:]):
        for kind in ["MW", "MWh"]:
            cap.loc[
                (cap["planning_year"] == year) & (cap["unit"] == kind), "start_value"
            ] = cap.loc[
                (cap["planning_year"] == prev_year) & (cap["unit"] == kind), "end_value"
            ]

    return cap.reset_index()


def convert_mwh_format(cap: pd.DataFrame) -> pd.DataFrame:

    if "start_MWh" not in cap.columns:
        return cap

    energy_cap = cap.loc[(cap["start_MWh"] > 0) | (cap["end_MWh"] > 0), :]
    energy_cap.loc[:, "start_value"] = energy_cap["start_MWh"]
    energy_cap.loc[:, "end_value"] = energy_cap["end_MWh"]
    energy_cap.loc[:, "unit"] = "MWh"

    return pd.concat([cap, energy_cap], ignore_index=True).drop(
        columns=["start_MWh", "end_MWh"]
    )


def new_retired_capacity(cap: pd.DataFrame) -> pd.DataFrame:

    cap["change_value"] = cap["end_value"] - cap["start_value"]
    cap["retired_value"] = cap["change_value"].where(cap["change_value"] < 0, -0) * -1
    cap["new_build_value"] = cap["change_value"].where(cap["change_value"] > 0, 0)

    return cap


def add_file_to_folder(resource_path: Path):

    _cap = pd.read_csv(resource_path)
    cap = (
        _cap.pipe(convert_mwh_format)
        .pipe(group_sum_capacity)
        .pipe(tech_to_type)
        .pipe(set_start_cap)
        .pipe(new_retired_capacity)
    )
    # cap = tech_to_type(_cap)
    tech_cap = (
        cap.groupby(
            ["model", "zone", "tech_type", "planning_year", "existing", "unit"]
        )[["start_value", "end_value", "retired_value", "new_build_value"]]
        .sum()
        .round(1)
    )
    tech_cap.to_csv(resource_path.parent / "aggregated_capacity_calc.csv")


def main():

    fns = list(parent_dir.rglob("resource_capacity.csv"))
    for fn in fns:
        if "26z" not in str(fn) and "case_settings" not in str(fn):
            print(fn.parent)
            add_file_to_folder(fn)


if __name__ == "__main__":
    main()
