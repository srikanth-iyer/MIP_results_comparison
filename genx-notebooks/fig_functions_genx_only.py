import math
import os
import re
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import altair as alt
import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import warnings
from shapely import wkt
from shapely.geometry.base import BaseGeometry
from fig_functions_config import (
    region_map,
    REGION_COLOR_MAP,
    TECH_MAP,
    TECH_ORDER,
    COLOR_MAP,
    TECH_STACK_ORDER,
    SCENARIO_MAPPING_AND_ORDER,
    NET_REVENUE_COMPONENT_ORDER,
    NET_REVENUE_COLOR_MAP,
)
# alt.data_transformers.enable("vegafusion")

try:
    pd.options.mode.copy_on_write = True
except:
    pass



# WRAPPED_CASE_NAME_MAP = {
    # "genx-scenarios-10-days": "GenX\nScenarios 10 Days", # NOTE: [SRI] newly added case
    # "genx-scenarios-20-weeks": "GenX\nScenarios 20 Weeks", # NOTE: [SRI] newly added case
    # "20-week-genx": "Only GenX\n20-week simulation", # NOTE: [SRI] newly added case
    # "No_action": "No Action",
    # "Current_Policy" : "Current Policy",
    # "And_No_IRA": "And No IRA",
    # "And_Optimistic": "And Optimistic",
    # "And_Transmission_Constrained": "And Transmission Constrained",
# }


# LINE_NAMES = [
#     "BASN_to_CANO",
#     "BASN_to_CASO",
#     "BASN_to_NWPP",
#     "BASN_to_RMRG",
#     "BASN_to_SRSG",
#     "CANO_to_CASO",
#     "CANO_to_NWPP",
#     "CASO_to_NWPP",
#     "CASO_to_SRSG",
#     "FRCC_to_SRSE",
#     "ISNE_to_NYCW",
#     "ISNE_to_NYUP",
#     "MISC_to_MISE",
#     "MISC_to_MISS",
#     "MISC_to_MISW",
#     "MISC_to_PJMC",
#     "MISC_to_PJMW",
#     "MISC_to_SPPC",
#     "MISC_to_SPPN",
#     "MISC_to_SPPS",
#     "MISC_to_SRCE",
#     "MISE_to_MISW",
#     "MISE_to_PJMW",
#     "MISS_to_SPPC",
#     "MISS_to_SPPS",
#     "MISS_to_SRCE",
#     "MISS_to_SRSE",
#     "MISW_to_NWPP",
#     "MISW_to_PJMC",
#     "MISW_to_SPPC",
#     "MISW_to_SPPN",
#     "NWPP_to_RMRG",
#     "NYCW_to_NYUP",
#     "NYCW_to_PJME",
#     "NYUP_to_PJME",
#     "PJMC_to_PJMW",
#     "PJMD_to_PJME",
#     "PJMD_to_PJMW",
#     "PJMD_to_SRCA",
#     "PJME_to_PJMW",
#     "PJMW_to_SRCA",
#     "PJMW_to_SRCE",
#     "RMRG_to_SPPC",
#     "RMRG_to_SPPN",
#     "RMRG_to_SRSG",
#     "SPPC_to_SPPN",
#     "SPPC_to_SPPS",
#     "SPPN_to_SPPS",
#     "SPPS_to_SRSG",
#     "SPPS_to_TRE",
#     "SPPS_to_TREW",
#     "SRCA_to_SRCE",
#     "SRCA_to_SRSE",
#     "SRCE_to_SRSE",
#     "TRE_to_TREW",
#     "NWPP_to_SPPN",
#     "SRSG_to_TRE",
# ]

# MODEL_NAMES mapping removed; we now keep model names as found in paths


def order_dict():
    return {
        # "case": list(WRAPPED_CASE_NAME_MAP.keys()),
        #              [
        #     WRAPPED_CASE_NAME_MAP[c]
        #     for c in list(input.base_cases()) + list(input.cp_cases())
        # ],
        # Sort models alphabetically instead of using a predefined order
        # "model": "ascending",
        'model': list(SCENARIO_MAPPING_AND_ORDER.values()),
        "tech_type": TECH_ORDER,
    }


def sort_nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a nested dictionary, iterate through all levels to sort keys by length.
    Dictionary values can be more nested dictionaries, strings, numbers, or lists.

    Parameters
    ----------
    d : Dict[str, Any]
        The nested dictionary to be sorted.

    Returns
    -------
    Dict[str, Any]
        The sorted dictionary where keys are sorted by length at each level.

    """
    sorted_dict = dict()

    for key, value in sorted(d.items(), key=lambda x: len(str(x[0]))):
        if isinstance(value, dict):
            sorted_dict[key] = sort_nested_dict(value)
        else:
            sorted_dict[key] = value

    return sorted_dict


_TECH_MAP = {}
# existing_tech_map = { # NOTE: getting all the tech in the tech_map with the word 'existing' somewhere in the label
#     k: v
#     for k, v in sort_nested_dict(TECH_MAP).items()
#     if "existing" in str(k).lower()
# }

for k, v in sort_nested_dict(TECH_MAP).items(): # NOTE: new method to label existing tech (they should have the word 'existing' somewhere in the label)
    # if "existing" in str(k).lower():
    #     _TECH_MAP[k] = (v, True)
    # else:
    #     _TECH_MAP[k] = (v, False)
    _TECH_MAP[k] = (v, False) # NOTE: WE DON'T USE THIS ANYMORE. WE infer Existing from the New_Build column. I'm keeping this for backward compatibility
    # if k in EXISTING_TECH_MAP.keys(): # NOTE: modified to make nice with the new mapping from Kavi
    #     _TECH_MAP[k] = (v, True)
    # else:
    #     _TECH_MAP[k] = (v, False)
_TECH_MAP = sort_nested_dict(_TECH_MAP)


def tech_to_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # avoid fragmented frames before mutating
    # Create dictionaries to map unique resource names to their tech types and existence
    tech_type_map = {}
    # existence_map = {}
    if "resource_name" not in df.columns:
        for column_str in df.columns:
            if "resource" in column_str.lower():
                # add a new column instead of renaming the existing one
                df["resource_name"] = df[column_str].copy()
                break
    # Get unique resource names
    unique_resource_names = sorted(
        df["resource_name"].unique().tolist(), key=lambda x: len(str(x[0]))
    )

    # Iterate over each unique resource name and determine its type and existence
    for resource_name in unique_resource_names:
        for tech, (tech_type, existing) in _TECH_MAP.items():
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
    df["tech_type"] = df["resource_name"].map(tech_type_map)
    # df["existing"] = df["resource_name"].map(existence_map)
    # df["new_build"] = ~df["existing"]

    return df


def reverse_dict_of_lists(d: Dict[str, list]) -> Dict[str, List[str]]:
    """Reverse the mapping in a dictionary of lists so each list item maps to the key

    Parameters
    ----------
    d : Dict[str, List[str]]
        A dictionary with string keys and lists of strings.

    Returns
    -------
    Dict[str, str]
        A reverse mapped dictionary where the item of each list becomes a key and the
        original keys are mapped as values.
    """
    if isinstance(d, dict):
        rev = {v: k for k in d for v in d[k]}
    else:
        rev = dict()
    return rev


rev_region_map = reverse_dict_of_lists(region_map)


def map_zone_values_to_regions(series: pd.Series) -> pd.Series:
    """Translate numeric or labelled zone identifiers to region names."""

    def _translate(value: Any) -> Any:
        if pd.isna(value):
            return value

        if isinstance(value, (int, np.integer)):
            return rev_region_map.get(int(value), value)

        value_str = str(value).strip()
        match = re.search(r"(\d+)", value_str)
        if match:
            zone_id = int(match.group(1))
            return rev_region_map.get(zone_id, value_str)

        try:
            zone_id = int(value_str)
        except ValueError:
            return rev_region_map.get(value_str, value_str)
        return rev_region_map.get(zone_id, value_str)

    return series.apply(_translate)

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
        "transmission_path_name",
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
        "value"
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


# @lru_cache
# def load_tx_line_length(folder: Path, line_names: List[str]) -> pd.DataFrame:
#     df = pd.read_csv(
#         folder / "network_costs_conus_26_zone.csv",
#         usecols=["start_region", "dest_region", "total_mw-km_per_mw"],
#     ).rename(columns={"total_mw-km_per_mw": "km"})
#     df["transmission_path_name"] = df["dest_region"] + "_to_" + df["start_region"]
#     for idx, row in df.iterrows():
#         if row["transmission_path_name"] not in line_names:
#             df.loc[:, "transmission_path_name"] = df["transmission_path_name"].str.replace(
#                 row["transmission_path_name"], reverse_line_name(row["transmission_path_name"])
#             )

#     return df  # .drop(columns=["start_region", "dest_region"])


def load_data(data_path: Path, fn: str, case_name: str = None) -> pd.DataFrame:
    df_list = []
    fn = f"{fn.split('.')[0]}.*"
    for f in data_path.rglob(fn):
        if not ("output" in f.parts[-2] or "Results" in f.parts[-2]):
            # print(f.parts[-2])
            _df = pd.read_csv(f)  # , engine="pyarrow")
            # , skip_blank_lines=True, engine="pyarrow" na_filter=False, )
            _df = _df.dropna(how="all")
            if case_name is not None:
                _df["case"] = case_name
            df_list.append(_df)
    if not df_list:
        print(f"Warning: No files matching pattern '{fn}' found in {data_path}")
        return pd.DataFrame(columns=DATA_COLS[fn.split(".")[0]])
    df = pd.concat(df_list, ignore_index=True)
    if "resource_name" in df.columns:
        df = tech_to_type(df) #NOTE: acquiring tech_type from the csv file itself
        df = df.query("~tech_type.str.contains('Other')")
    if "transmission_path_name" in df.columns:
        # df = fix_tx_line_names(df)
        # line_length = load_tx_line_length(data_path.parent / "notebooks", LINE_NAMES)
        # df = pd.merge(df, line_length, on=["transmission_path_name"])
        if "end_value" in df.columns:
            df["end_cap"] = df["end_value"].copy()
            df["start_cap"] = df["start_value"].copy()
            # df["end_value"] = df["end_value"] * df["km"]
            # df["start_value"] = df["start_value"] * df["km"]
        else:
            df["cap"] = df["value"].copy()
            # df["value"] = df["value"] * df["km"]
    if "zone" in df.columns:
        # df.loc[:, "agg_zone"] = df.loc[:, "zone"].map(rev_region_map)
        df.loc[:, "agg_zone"] = df.loc[:, "zone"].copy() # NOTE: cap takes input from resource_capacity that already has zone in name form 
    for col in ["value", "start_value", "end_value"]:
        if col in df.columns:
            df.loc[:, col] = df[col].round(0)
    if "dispatch" in fn:
        df = fill_dispatch_hours(df)
    return df


def fill_dispatch_hours(dispatch: pd.DataFrame) -> pd.DataFrame:
    if dispatch.empty:
        return dispatch

    dispatch = dispatch.groupby(
        [
            "planning_year",
            "model",
            "agg_zone",
            "zone",
            "tech_type",
            "resource_name",
            "hour",
        ],
        as_index=False,
    )["value"].sum()
    group_cols = [
        "planning_year", 
        "model", 
        "agg_zone", 
        "zone", 
        "tech_type"
        ]
    hours = dispatch["hour"].unique()
    index_cols = ["resource_name"]
    df_list = []
    for _, _df in dispatch.groupby(group_cols):
        multi_index = pd.MultiIndex.from_product(
            [_df[col].unique() for col in index_cols] + [hours],
            names=index_cols + ["hour"],
        )
        _df = _df.set_index(index_cols + ["hour"])
        _df = _df.reindex(index=multi_index, fill_value=0)
        _df = _df.reset_index()
        for val, col in zip(_, group_cols):
            _df[col] = val
        df_list.append(_df)

    dispatch = pd.concat(df_list, ignore_index=True)

    return dispatch


def find_periods(files: List[Path]) -> List[str]:
    if not files:
        return []
    part_idx = -3
    for i, part in enumerate(files[0].parts):
        if "Inputs_p" in part or "Results_p" in part:
            part_idx = i
            break

    periods = list(set([f.parts[part_idx] for f in files]))
    return periods


def _case_year_mapping_from_csv(csv_path: Path) -> Dict[str, int]:
    try:
        df = pd.read_csv(csv_path, usecols=["case", "planning_year"])
    except (FileNotFoundError, ValueError):
        return {}

    if df.empty or "case" not in df or "planning_year" not in df:
        return {}

    mapping: Dict[str, int] = {}
    rows = (
        df[["case", "planning_year"]]
        .dropna(subset=["case", "planning_year"])
        .drop_duplicates(subset=["case"])
    )

    for case, planning_year in rows.itertuples(index=False):
        case_str = str(case)
        if not case_str:
            continue
        suffix = case_str.split("_", 1)[-1].lower() if "_" in case_str else case_str.lower()
        if suffix and suffix not in mapping:
            try:
                mapping[suffix] = int(planning_year)
            except (TypeError, ValueError):
                continue

    return mapping


@lru_cache(maxsize=None)
def _discover_period_mapping(data_path_str: str) -> Dict[str, int]:
    data_path = Path(data_path_str)
    mapping: Dict[str, int] = {}

    summary_dirs = sorted(data_path.glob("*_results_summary"))
    summary_files = (
        "resource_capacity.csv",
        "generation.csv",
        "emissions.csv",
        "dispatch.csv",
        "annual_demand.csv",
    )

    for summary_dir in summary_dirs:
        for filename in summary_files:
            summary_path = summary_dir / filename
            if suffix_mapping := _case_year_mapping_from_csv(summary_path):
                for key, value in suffix_mapping.items():
                    mapping.setdefault(key, value)

    if not mapping:
        mapping["p1"] = 2030

    return mapping


def load_genx_operations_data(
    data_path: Path,
    fn: str,
    period_dict: Dict[str, int] | None = None,
    hourly_data: bool = False,
    model_costs_only: bool = False,
) -> pd.DataFrame:
    df_list = []
    nrows = None
    if hourly_data:
        nrows = 5
    files = list(data_path.rglob(fn))
    files = [f for f in files if "op_inputs" in str(f) and "_Results" not in str(f)]
    periods = find_periods(files)
    period_keys = {
        p.split("_", 1)[-1].lower()
        for p in periods
        if "_" in p
    }
    # if any("p6" in p for p in periods):
    #     period_dict = (
    #         {"p1": 2027, "p2": 2030, "p3": 2035, "p4": 2040, "p5": 2045, "p6": 2050},
    #     )
    if not files:
        return pd.DataFrame()
    inferred_periods = _discover_period_mapping(str(data_path.resolve()))
    combined_period_dict: Dict[str, int] = {**inferred_periods}
    if period_dict:
        combined_period_dict.update({k.lower(): v for k, v in period_dict.items()})

    missing_periods = sorted(period_keys - combined_period_dict.keys())
    if missing_periods:
        raise KeyError(
            "Missing planning year mapping for period(s): "
            + ", ".join(missing_periods)
            + f" in {data_path}. Provide period_dict or ensure results summary files include planning_year values."
        )

    df_list = Parallel(n_jobs=1)(
        delayed(_load_op_data)(
            f,
            hourly_data,
            nrows,
            combined_period_dict,
            model_costs_only,
        )
        for f in files
    )
    if not df_list:
        return pd.DataFrame(columns=DATA_COLS.get(fn.split(".")[0], []))
    df = pd.concat(df_list, ignore_index=True)
    if fn == "costs.csv":
        try:
            df = add_genx_op_network_cost(df, data_path).pipe(calc_op_percent_total)
            # df = append_npv_cost(df) # NOTE: Commenting this out for now . Check if we need NPV
        except FileNotFoundError:
            pass
    if "Resource" in df.columns:
        df = df.rename(columns={"Resource": "resource_name"}).pipe(tech_to_type)
        try:
            df.loc[:, "zone"] = df["resource_name"].str.split("_").str[0]
        except:
            df.loc[:, "zone"] = df["resource_name"].str.split("_").list[0]
        # df.loc[df["resource_name"].str.contains("TRE_WEST"), "zone"] = "TRE_WEST"
    if "zone" in df.columns:
        df.loc[:, "agg_zone"] = df.loc[:, "zone"].map(rev_region_map)
    for col in df.columns:
        if col != "percent_total" and pd.api.types.is_numeric_dtype(df[col]):
            df.loc[:, col] = df[col].round(1)
    return df.round(1)


@lru_cache
def load_op_gen_data(period: str, load_base: bool = True) -> pd.DataFrame:
    if load_base:
        path = (
            Path(__file__).parent.parent
            / "genx-op-inputs"
            / "base_52_week_commit"
            / "Inputs"
            / f"Inputs_{period}"
            / "Generators_data.csv"
        )
    else:
        path = (
            Path(__file__).parent.parent
            / "genx-op-inputs"
            / "current_policies_52_week_commit"
            / "Inputs"
            / f"Inputs_{period}"
            / "Generators_data.csv"
        )
    vom = pd.read_csv(
        path,
        usecols=[
            "Resource",
            "Var_OM_Cost_per_MWh",
            "Heat_Rate_MMBTU_per_MWh",
            # "Fixed_OM_Cost_per_MWyr",
            # "Fixed_OM_Cost_per_MWhyr",
            # "base_Fixed_OM_Cost_per_MWyr",
            # "base_Fixed_OM_Cost_per_MWhyr",
            # "Existing_Cap_MW",
            # "Existing_Cap_MWh",
        ],
    )
    vom["co2_tonne_mwh"] = 0.0
    vom.loc[vom["Resource"].str.contains("natural"), "co2_tonne_mwh"] = (
        vom["Heat_Rate_MMBTU_per_MWh"] * 0.05306
    )
    vom.loc[vom["Resource"].str.contains("cc_95_ccs"), "co2_tonne_mwh"] *= 0.05
    vom.loc[vom["Resource"].str.contains("coal"), "co2_tonne_mwh"] = (
        vom["Heat_Rate_MMBTU_per_MWh"] * 0.09552
    )
    return vom


def load_op_generators_data(f: Path) -> pd.DataFrame:
    try:
        cols = [
            "Resource",
            "Fixed_OM_Cost_per_MWyr",
            "Fixed_OM_Cost_per_MWhyr",
            "base_Fixed_OM_Cost_per_MWyr",
            "base_Fixed_OM_Cost_per_MWhyr",
            "Existing_Cap_MW",
            "Existing_Cap_MWh",
        ]
        df = pd.read_csv(f, usecols=cols)
        df["fixed_costs"] = (df["Existing_Cap_MW"] * df["Fixed_OM_Cost_per_MWyr"]) + (
            df["Existing_Cap_MWh"] * df["Fixed_OM_Cost_per_MWhyr"]
        )
        df["base_fixed_costs"] = (
            df["Existing_Cap_MW"] * df["base_Fixed_OM_Cost_per_MWyr"]
        ) + (df["Existing_Cap_MWh"] * df["base_Fixed_OM_Cost_per_MWhyr"])
    except ValueError:
        cols = [
            "Resource",
            "Fixed_OM_Cost_per_MWyr",
            "Fixed_OM_Cost_per_MWhyr",
            "Existing_Cap_MW",
            "Existing_Cap_MWh",
        ]
        df = pd.read_csv(f, usecols=cols)
        df["fixed_costs"] = (df["Existing_Cap_MW"] * df["Fixed_OM_Cost_per_MWyr"]) + (
            df["Existing_Cap_MWh"] * df["Fixed_OM_Cost_per_MWhyr"]
        )
        df["base_fixed_costs"] = df["fixed_costs"].copy()

    return df


def _load_op_data(
    f: Path,
    hourly_data: bool,
    nrows=None,
    period_dict: Dict[str, int] | None = None,
    model_costs_only: bool = False,
) -> pd.DataFrame:
    if period_dict is None:
        raise ValueError("period_dict must be provided when loading GenX operations data")
    fn = f.name
    model_part = -3
    _df = pd.read_csv(f, nrows=nrows)  # , dtype_backend="pyarrow")
    if hourly_data:
        if fn == "nse.csv":
            _df = total_from_nse_hourly_data(_df)
        elif fn == "emissions.csv":
            _df = total_from_emissions_hourly_data(_df)
        elif "Resource" in _df.columns:
            _df = total_from_resource_op_hourly_data(_df)
        else:
            raise ValueError(f"There is no hourly data function for file {fn}")
    if "Results_p" in str(f):
        period_str = f.parent.stem.split("_")[-1]
        period = period_dict[period_str]
        _df.loc[:, "planning_year"] = period
        model_part = -4
    elif "Inputs_p" in str(f):
        period_str = f.parents[1].stem.split("_")[-1]
        period = period_dict[period_str]
        _df.loc[:, "planning_year"] = period
        model_part = -5
    # Use the model name directly from the path segment (no remapping or case changes)
    model = f.parts[model_part].split("_")[0]
    _df.loc[:, "model"] = model
    if fn == "costs.csv" and model_costs_only:
        return _df

    if fn == "costs.csv":
        if not (f.parent / "capacityfactor.csv").exists():
            return pd.DataFrame()
        # Need to modify so that model costs are reported, then subsidies are also reported
        # as their own bars. Can use the difference between base/policy values.
        fixed_gen_cost = load_op_generators_data(f.parents[1] / "Generators_data.csv")

        annual_gen = pd.read_csv(f.parent / "capacityfactor.csv")
        annual_gen["co2_tonne"] = 0
        base_gen_data = load_op_gen_data(period_str, load_base=True)
        current_gen_data = load_op_gen_data(period_str, load_base=False)
        annual_gen_base = pd.merge(annual_gen, base_gen_data, on="Resource")
        annual_gen_base["var_om"] = (
            annual_gen_base["AnnualSum"] * annual_gen_base["Var_OM_Cost_per_MWh"]
        )
        annual_gen_base["co2_tonne"] = (
            annual_gen_base["co2_tonne_mwh"] * annual_gen_base["AnnualSum"]
        )
        total_vom_base = annual_gen_base["var_om"].sum()
        annual_gen_current = pd.merge(annual_gen, current_gen_data, on="Resource")
        annual_gen_current["var_om"] = (
            annual_gen_current["AnnualSum"] * annual_gen_current["Var_OM_Cost_per_MWh"]
        )
        annual_gen_current["co2_tonne"] = (
            annual_gen_current["co2_tonne_mwh"] * annual_gen_current["AnnualSum"]
        )
        total_vom_current = annual_gen_current["var_om"].sum()
        s = pd.DataFrame(
            data={
                "Costs": ["cVOM"],
                "Total": [total_vom_base],
                "planning_year": [period],
                "model": [model],
            }
        )
        if "base-50" in str(f):
            social_cost = 50
        elif "base-1000" in str(f):
            social_cost = 1000
        else:
            social_cost = 200

        co2_cost = pd.DataFrame(
            data={
                "Costs": ["cCO2"],
                "Total": [annual_gen_base["co2_tonne"].sum() * social_cost],
                "planning_year": [period],
                "model": [model],
            }
        )
        _df = pd.concat([_df, s, co2_cost])

        # Take out unmet policy penalty costs since we are include full social cost of carbon
        _df = _df.query("Costs != 'cUnmetPolicyPenalty'")

        if "current" in str(f) or (_df["Total"] < 0).any():
            _df.loc[_df["Costs"] == "cVar", "Total"] -= total_vom_current
            policy_cost_ptc = total_vom_base - total_vom_current
            policy_cost_itc = (
                fixed_gen_cost["base_fixed_costs"].sum()
                - fixed_gen_cost["fixed_costs"].sum()
            )
            _df.loc[_df["Costs"] == "cFix", "Total"] -= policy_cost_itc
            s = pd.DataFrame(
                data={
                    "Costs": ["cPolicy_PTC", "cPolicy_ITC"],
                    "Total": [policy_cost_ptc, policy_cost_itc],
                    "planning_year": [period, period],
                    "model": [model, model],
                }
            )
            _df = pd.concat([_df, s])

        else:
            _df.loc[_df["Costs"] == "cVar", "Total"] -= total_vom_base

    return _df


def total_from_resource_op_hourly_data(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "Resource": df.columns[1:-1],
            "value": df.iloc[1, 1:-1],
        }
    ).reset_index(drop=True)

    return data


def total_from_nse_hourly_data(df: pd.DataFrame) -> pd.DataFrame:
    reg_map = {i + 1: r for i, r in enumerate(sorted(sum(region_map.values(), [])))}
    data = pd.DataFrame(
        {
            "Zone": df.iloc[0, 1:-1].astype(int).to_list(),
            "value": df.iloc[1, 1:-1],
        }
    ).reset_index(drop=True)
    data["zone"] = data["Zone"].map(reg_map)
    return data


def total_from_emissions_hourly_data(df: pd.DataFrame) -> pd.DataFrame:
    reg_map = {i + 1: r for i, r in enumerate(sorted(sum(region_map.values(), [])))}
    data = pd.DataFrame(
        {
            "Zone": df.columns[1:-1].astype(int).to_list(),
            "value": df.iloc[1, 1:-1],
        }
    ).reset_index(drop=True)
    data["zone"] = data["Zone"].map(reg_map)
    return data


def calc_op_percent_total(
    op_costs: pd.DataFrame, group_by=["model", "planning_year"]
) -> pd.DataFrame:
    by = [c for c in group_by if c in op_costs.columns]
    df_list = []
    for _, _df in op_costs.query("Costs != 'cTotal'").groupby(by):
        _df.loc[:, "percent_total"] = (_df["Total"] / _df["Total"].sum()).round(3)
        df_list.append(_df)
    return pd.concat(df_list)


def add_genx_op_network_cost(
    op_costs: pd.DataFrame,
    data_path: Path,
    original_network_fn: str = "original_network.csv",
    final_network_fn: str = "Network.csv",
    period_dict={
        "p1": 2030, # NOTE: Updated to 2030 from 2027 for new genx data
        # "p2": 2030,
        # "p3": 2035,
        # "p4": 2040,
        # "p5": 2045,
        # "p6": 2050,
    },
) -> pd.DataFrame:
    read_cols = [
        "Network_Lines",
        "Line_Max_Flow_MW",
        "Line_Reinforcement_Cost_per_MWyr",
    ]
    for f in data_path.rglob(original_network_fn):
        model_part = -2
        original_df = pd.read_csv(f, usecols=read_cols).set_index("Network_Lines")
        if "Inputs_p" in str(f):
            model_part = -4
            period = period_dict[f.parent.stem.split("_")[-1]]

        final_df = pd.read_csv(
            f.parent / final_network_fn, usecols=read_cols
        ).set_index("Network_Lines")
        # Use the model name directly from the path segment (no remapping or case changes)
        model = f.parts[model_part].split("_")[0]
        new_tx_cost = (
            (final_df["Line_Max_Flow_MW"] - original_df["Line_Max_Flow_MW"])
            * original_df["Line_Reinforcement_Cost_per_MWyr"]
        ).sum()
        if "Inputs_p" in str(f):
            op_costs.loc[
                (op_costs["model"] == model)
                & (op_costs["Costs"] == "cNetworkExp")
                & (op_costs["planning_year"] == period),
                "Total",
            ] = new_tx_cost
        else:
            op_costs.loc[
                (op_costs["model"] == model) & (op_costs["Costs"] == "cNetworkExp"),
                "Total",
            ] = new_tx_cost

    return op_costs


# def reverse_line_name(s: str) -> str:
#     segments = s.split("_to_")
#     return segments[-1] + "_to_" + segments[0]


# def fix_tx_line_names(df: pd.DataFrame) -> pd.DataFrame:
#     for idx, row in df.iterrows():
#         if row["line_name"] not in LINE_NAMES:
#             df.loc[:, "line_name"] = df["line_name"].str.replace(
#                 row["line_name"], reverse_line_name(row["line_name"])
#             )
#     line_count = df.groupby("line_name", as_index=False)["model"].count()
#     median_count = line_count["model"].median()
#     reversed_lines = line_count.query("model < @median_count")

#     for idx, row in reversed_lines.iterrows():
#         df.loc[:, "line_name"] = df["line_name"].str.replace(
#             row["line_name"], reverse_line_name(row["line_name"])
#         )

#     return df


def calc_period_retirements(
    cap: pd.DataFrame,
    by_region: bool = False,
    by_agg_zone: bool = False,
    multi_case: bool = True,
) -> pd.DataFrame:
    idx = pd.IndexSlice
    start_cap = cap.query(
        "model == 'GenX' and new_build == False and planning_year==2027"
    )
    years = sorted(cap.planning_year.unique())
    by = ["tech_type", "model", "planning_year"]
    idx_cols = ["tech_type"]
    if by_region:
        by.append("zone")
        idx_cols.append("zone")
    if by_agg_zone:
        by.append("agg_zone")
        idx_cols.append("agg_zone")
    if multi_case:
        by.append("case")
        idx_cols.append("case")

    # Create a complete multi-index product of cols because some models
    # do not report small or zero capacity values. Missing rows mess up the
    # calculations and figures.
    midx = pd.MultiIndex.from_product([cap[c].unique() for c in by], names=by)
    exist_cap = (
        (
            cap.query("new_build == False")
            .groupby(by)[["start_value", "end_value"]]
            .sum()
            # .set_index(idx_cols)
        )
        .reindex(midx, fill_value=0)
        .reset_index()
        .set_index(idx_cols)
    )

    start_cap = exist_cap.query("model == 'GenX' and planning_year == 2027").drop(
        columns="end_value"
    )  # .set_index("tech_type")

    retire_cap = exist_cap.copy()
    retire_cap.loc[:, "end_value"] = 0
    retire_cap = retire_cap  # .set_index("tech_type")

    for model in cap.model.unique():
        retire_idx = retire_cap["planning_year"] == years[0]
        exist_start_idx = (exist_cap["planning_year"] == years[0]) & (
            exist_cap["model"] == "GenX"
        )
        exist_end_idx = (exist_cap["planning_year"] == years[0]) & (
            exist_cap["model"] == model
        )
        retire_cap.loc[retire_idx, "end_value"] = (
            exist_cap.loc[exist_end_idx, "end_value"]
            - exist_cap.loc[exist_start_idx, "start_value"]
        )

    idx_cols.append("model")
    retire_cap = retire_cap.reset_index().set_index(idx_cols)
    exist_cap = exist_cap.reset_index().set_index(idx_cols)
    for year, prev_year in zip(years[1:], years[:-1]):
        retire_cap.loc[
            retire_cap["planning_year"] == year, "end_value"
        ] = exist_cap.loc[exist_cap["planning_year"] == year, "end_value"].sub(
            exist_cap.loc[exist_cap["planning_year"] == prev_year, "end_value"],
            fill_value=0,
        )

    return retire_cap.query("tech_type != 'Distributed Solar'").rename(
        columns={"end_value": "value"}
    )


def calc_mean_annual_cap(
    cap: pd.DataFrame,
    by_region: bool = False,
    by_agg_zone: bool = False,
    new_build: bool = True,
    existing: bool = False,
    value_col: str = "end_value",
) -> pd.DataFrame:
    idx = pd.IndexSlice
    years = sorted(cap.planning_year.unique(), reverse=True)
    by = ["case", "tech_type", "planning_year"]
    if by_region:
        by.append("zone")
    if by_agg_zone:
        by.append("agg_zone")
    if new_build:
        _cap = cap.query("new_build == True and unit == 'MW'")
    elif existing:
        _cap = cap.query("new_build == False and unit == 'MW'")
    elif "unit" in cap.columns:
        _cap = cap.query("unit == 'MW'")
    else:
        _cap = cap.copy()
    idx_cols = ["case", "tech_type"]
    if by_region:
        idx_cols.append("zone")
    if by_agg_zone:
        idx_cols.append("agg_zone")

    df_list = []

    # Need to make sure that all techs are in all regions for cases submitted by each
    # model. Reindexing across all models screws up the min/max error bars when a model
    # has not submitted a case because the minimum becomes 0.
    for model in cap.model.unique():
        _model_cap = _cap.query("model==@model")
        midx = pd.MultiIndex.from_product(
            [_model_cap[c].unique() for c in by], names=by
        )
        _annual_cap = (
            _model_cap.query("model==@model")
            .groupby(by)[value_col]
            .sum()
            .reindex(midx)  # , fill_value=0)
            .reset_index()
            .set_index(idx_cols)
        )
        _annual_cap["model"] = model
        df_list.append(_annual_cap)
    annual_cap = pd.concat(df_list)
    if not new_build and not by_agg_zone:
        return (
            annual_cap.reset_index()
            .groupby(["case", "planning_year", "tech_type"], as_index=False)[value_col]
            .mean()
            .round({"end_value": 1})
        )
    elif not new_build and by_agg_zone:
        return (
            annual_cap.reset_index()
            .groupby(
                ["case", "planning_year", "tech_type", "agg_zone"], as_index=False
            )[value_col]
            .mean()
            .round({"end_value": 1})
        )
    for year, prev_year in zip(years[:-1], years[1:]):
        annual_cap.loc[annual_cap["planning_year"] == year, value_col] = annual_cap.loc[
            annual_cap["planning_year"] == year, value_col
        ].sub(
            annual_cap.loc[annual_cap["planning_year"] == prev_year, value_col],
            fill_value=0,
        )

    by = ["case", "tech_type", "planning_year"]
    if by_region:
        by.append("zone")
    if by_agg_zone:
        by.append("agg_zone")
    annual_cap_mean = pd.DataFrame(annual_cap.groupby(by)[value_col].mean())

    by = ["case", "tech_type", "model"]
    min_max_by = ["case", "tech_type"]
    if by_region:
        by.append("zone")
        min_max_by.append("zone")
    if by_agg_zone:
        by.append("agg_zone")
        min_max_by.append("agg_zone")
    annual_cap_mean.loc[idx[:, :, 2050], "min"] = (
        annual_cap.groupby(by)[value_col].sum().groupby(min_max_by).min().values
    )
    annual_cap_mean.loc[idx[:, :, 2050], "max"] = (
        annual_cap.groupby(by)[value_col].sum().groupby(min_max_by).max().values
    )
    return annual_cap_mean.round(1)


def calc_mean_annual_gen(
    gen: pd.DataFrame,
    by_region: bool = False,
    by_agg_zone: bool = False,
    new_build: bool = False,
    value_col: str = "value",
) -> pd.DataFrame:
    idx = pd.IndexSlice
    years = sorted(gen.planning_year.unique(), reverse=True)
    by = ["case", "tech_type", "planning_year"]
    if by_region:
        by.append("zone")
    if by_agg_zone:
        by.append("agg_zone")
    if new_build:
        _gen = gen.query("new_build == True")
    else:
        _gen = gen.copy()
    idx_cols = ["case", "tech_type"]
    if by_region:
        idx_cols.append("zone")
    if by_agg_zone:
        idx_cols.append("agg_zone")
    df_list = []

    # Need to make sure that all techs are in all regions for cases submitted by each
    # model. Reindexing across all models screws up the min/max error bars when a model
    # has not submitted a case because the minimum becomes 0.
    for model in gen.model.unique():
        _model_gen = _gen.query("model==@model")
        midx = pd.MultiIndex.from_product(
            [_model_gen[c].unique() for c in by], names=by
        )
        _annual_cap = (
            _model_gen.query("model==@model")
            .groupby(by)[value_col]
            .sum()
            .reindex(midx)  # , fill_value=0)
            .reset_index()
            .set_index(idx_cols)
        )
        _annual_cap["model"] = model
        df_list.append(_annual_cap)
    annual_gen = pd.concat(df_list)
    # annual_gen = _gen.groupby(by, as_index=False)[value_col].sum().set_index(idx_cols)
    # if not new_build:
    by = ["case", "tech_type", "planning_year"]
    if by_region:
        by.append("zone")
    if by_agg_zone:
        by.append("agg_zone")
    avg_gen = (
        pd.DataFrame(
            annual_gen.reset_index().groupby(by)[value_col].agg(["mean", "min", "max"])
        )
        .rename(columns={"mean": "value"})
        .round(1)
    )
    return avg_gen
    # for year, prev_year in zip(years[:-1], years[1:]):
    #     annual_gen.loc[annual_gen["planning_year"] == year, value_col] = annual_gen.loc[
    #         annual_gen["planning_year"] == year, value_col
    #     ].sub(
    #         annual_gen.loc[annual_gen["planning_year"] == prev_year, value_col],
    #         fill_value=0,
    #     )

    # by = ["case", "tech_type", "planning_year"]
    # if by_region:
    #     by.append("zone")
    # if by_agg_zone:
    #     by.append("agg_zone")
    # annual_cap_mean = pd.DataFrame(annual_gen.groupby(by)[value_col].mean())

    # by = ["case", "tech_type", "model", "planning_year"]
    # min_max_by = ["case", "tech_type", "planning_year"]
    # if by_region:
    #     by.append("zone")
    #     min_max_by.append("zone")
    # if by_agg_zone:
    #     by.append("agg_zone")
    #     min_max_by.append("agg_zone")
    # annual_cap_mean.loc[idx[:, :, 2050], "min"] = (
    #     annual_gen.groupby(by)[value_col].sum().groupby(min_max_by).min().values
    # )
    # annual_cap_mean.loc[idx[:, :, 2050], "max"] = (
    #     annual_gen.groupby(by)[value_col].sum().groupby(min_max_by).max().values
    # )
    # return annual_cap_mean.round(1)


def title_case(s: str) -> str:
    if isinstance(s, str):
        return s.replace("_", " ").title()


VAR_ABBR_MAP = {
    "model": "m",
    "case": "c",
    "planning_year": "y",
    "resource_name": "rn",
    "agg_zone": "az",
    "zone": "z",
    "tech_type": "tt",
    "cost_type": "ct",
    "netrevenue_component": "nrc",
    "value": "v",
    "end_value": "ev",
    "line_name": "ln",
    "transmission_path_name": "tpn",
    "Region": "r",
}

VAR_ABBR_TITLE_MAP = {v: title_case(k) for k, v in VAR_ABBR_MAP.items()}
VAR_ABBR_TITLE_MAP["v"] = "Capacity (GW)"
VAR_ABBR_TITLE_MAP["v"] = "Generation (TWh)"
VAR_ABBR_TITLE_MAP["nrc"] = "Net Revenue Component"




def _ordered_netrev_components(values: List[str]) -> List[str]:
    ordered = [comp for comp in NET_REVENUE_COMPONENT_ORDER if comp in values]
    remainder = [comp for comp in values if comp not in ordered]
    return ordered + remainder


def _apply_netrev_sign(df: pd.DataFrame) -> pd.DataFrame:
    component_col = "netrevenue_component"
    if component_col not in df.columns or "value" not in df.columns:
        return df

    data = df.copy()
    components = data[component_col].astype(str)

    cost_mask = components.str.contains("cost", case=False, na=False)
    data.loc[cost_mask, "value"] = -data.loc[cost_mask, "value"].abs()

    # revenue_mask = components.str.contains("revenue", case=False, na=False)
    # data.loc[revenue_mask, "value"] = data.loc[revenue_mask, "value"].abs()

    # profit_mask = components.str.fullmatch("profit", case=False)
    # data.loc[profit_mask, "value"] = data.loc[profit_mask, "value"].abs()

    return data


def configure_full_label_display(chart: alt.Chart) -> alt.Chart:
    """Ensure long labels (e.g., model names) render in full across axes, legends, and headers."""
    return (
        chart.configure_axis(labelLimit=0, labelPadding=2, titlePadding=15)
        .configure_legend(labelLimit=0)
        .configure_header(labelLimit=0)
    )


def config_chart_row_col(
    chart: alt.Chart, row_var: str, col_var: str, x_var: str
) -> alt.Chart:
    if row_var is not None and col_var is not None and row_var == col_var:
        row_var = None
    if col_var is not None and row_var is not None:
        chart = chart.facet(
            column=alt.Column(VAR_ABBR_MAP[col_var])
            .sort(order_dict().get(col_var))
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15),
            row=alt.Row(VAR_ABBR_MAP[row_var])
            .sort(order_dict().get(row_var))
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15),
        )
    elif col_var is not None:
        chart = chart.facet(
            column=alt.Column(VAR_ABBR_MAP[col_var])
            .sort(order_dict().get(col_var))
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    elif row_var is not None:
        chart = chart.facet(
            row=alt.Row(VAR_ABBR_MAP[row_var])
            .sort(order_dict().get(row_var))
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
        titleFontSize=20, labelFontSize=16
    )
    if col_var == "case":
        chart = (
            chart.configure(lineBreak="\n")
            .configure_axis(labelFontSize=15, titleFontSize=15)
            .configure_legend(titleFontSize=20, labelFontSize=16)
        )
    if x_var == "case":
        chart = (
            chart.configure(lineBreak="\n")
            .configure_axis(labelFontSize=15, titleFontSize=15)
            .configure_legend(titleFontSize=20, labelFontSize=16)
            .configure_axisX(
                labelBaseline="line-bottom", labelFontSize=15, titleFontSize=15
            )
        )
    chart = configure_full_label_display(chart)
    return chart


def chart_total_cap(
    cap: pd.DataFrame,
    mark_type="bar",
    x_var="model",
    col_var=None,
    row_var="planning_year",
    color="tech_type",
    order=None,
    width=alt.Step(40),
    height=200,
) -> alt.Chart:
    # Apply default ordering based on x_var if order is not provided
    if order is None and x_var in order_dict():
        order = order_dict()[x_var]
    group_by = ["tech_type", x_var, color]
    _tooltips = [
        alt.Tooltip(VAR_ABBR_MAP["tech_type"], title="Technology"),
        alt.Tooltip(VAR_ABBR_MAP["end_value"], title="Capacity (GW)", format=",.0f"),
        alt.Tooltip(VAR_ABBR_MAP[x_var], title=title_case(x_var)),
    ]
    if col_var is not None:
        group_by.append(col_var)
        _tooltips.append(
            alt.Tooltip(
                VAR_ABBR_MAP[col_var], title=VAR_ABBR_TITLE_MAP[VAR_ABBR_MAP[col_var]]
            )
        )
    if row_var is not None:
        group_by.append(row_var)
        _tooltips.append(alt.Tooltip(VAR_ABBR_MAP[row_var], title=title_case(row_var)))
    if "new_build" in cap.columns:
        group_by.append("new_build")
        _tooltips.append(alt.Tooltip("new_build").title("New Build"))
    group_by = [c for c in set(group_by) if c in cap.columns]
    cap_data = cap.groupby(group_by, as_index=False)["end_value"].sum()
    cap_data["end_value"] /= 1000
    cap_data = cap_data.rename(columns=VAR_ABBR_MAP)
    cap_data["o"] = cap_data["tt"].map(TECH_STACK_ORDER)
    if mark_type.lower() == "line":
        c = alt.Chart(cap_data).mark_line()
    else:
        c = alt.Chart(cap_data).mark_bar()

    if color == "tech_type":
        _color = (
            alt.Color("tt")
            .scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values()))
            .title(title_case("tech_type"))
        )
    else:
        _color = alt.Color(f"{VAR_ABBR_MAP[color]}").title(title_case(color))
    x_axis = alt.Axis(title=title_case(x_var))
    if x_var == "model":
        x_axis = alt.Axis(
            title="Scenarios",
            # titlePadding=45,
            # labelPadding=10,
        )
    chart = c.encode(
        x=alt.X(VAR_ABBR_MAP[x_var], sort=order, axis=x_axis),
        y=alt.Y("sum(ev)").title("Capacity (GW)"),
        color=_color,
        tooltip=_tooltips,
        order=alt.Order("o"),
    ).properties(width=width, height=height)
    chart = config_chart_row_col(chart, row_var, col_var, x_var)
    return chart


def chart_avg_new_tech_variation(
    annual_new_cap_mean: pd.DataFrame,
    x_var: str = "tech_type",
    col_var: str = "case",
    row_var: str = None,
    order=None,
    xOffset: str = None,
    bars_only=False,
    height=200,
    width=alt.Step(40),
) -> alt.Chart:
    data = annual_new_cap_mean.reset_index()
    for col in ["end_value", "min", "max"]:
        data[col] = (data[col] / 1000).round(1)
    tooltips = [
        alt.Tooltip("y", title="Planning Year"),
        alt.Tooltip("ev", title="Capacity (GW)", format=",.0f"),
    ]
    if x_var:
        tooltips.append(alt.Tooltip(VAR_ABBR_MAP[x_var], title=title_case(x_var)))
    if col_var:
        tooltips.append(alt.Tooltip(VAR_ABBR_MAP[col_var], title=title_case(col_var)))
    if row_var:
        tooltips.append(alt.Tooltip(VAR_ABBR_MAP[row_var], title=title_case(row_var)))
    # if bars_only:
    #     return bars

    bars = (
        alt.Chart()
        .mark_bar()
        .encode(
            y=alt.Y("ev").title("Capacity (GW)"),
            x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var)),
            color=alt.Color("tt")
            .scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values()))
            .title(title_case("tech_type")),
            opacity=alt.Opacity("y:O", sort="descending").title(
                title_case("planning_year")
            ),
            order=alt.Order(
                # Sort the segments of the bars by this field
                "y",
                sort="ascending",
            ),
            tooltip=tooltips,
            # xOffset=alt.XOffset(VAR_ABBR_MAP.get(xOffset)),
        )
    )
    if xOffset:
        bars.encode(
            xOffset=alt.XOffset(VAR_ABBR_MAP.get(xOffset)),
        )

    error_bars = (
        alt.Chart()
        .mark_errorbar()
        .encode(
            x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var)),
            y=alt.Y("max").title("Capacity (GW)"),
            y2=alt.Y2("min"),
        )
        .properties(width=width, height=height)
    )
    data = data.rename(columns=VAR_ABBR_MAP)
    # data["o"] = data["tt"].map(TECH_STACK_ORDER)
    chart = alt.layer(bars, error_bars, data=data)
    chart = config_chart_row_col(chart, row_var, col_var, x_var)
    return chart


def chart_retirements(
    retire_cap: pd.DataFrame,
    x_var: str = "tech_type",
    col_var: str = "case",
    row_var: str = None,
    order=None,
    width=alt.Step(40),
    height=200,
) -> alt.Chart:
    data = retire_cap.reset_index()
    data["value"] /= 1000
    tooltips = [
        alt.Tooltip("y", title="Planning Year"),
        alt.Tooltip("sum(v)", title="Capacity (GW)", format=",.0f"),
    ]
    if x_var:
        tooltips.append(alt.Tooltip(VAR_ABBR_MAP[x_var], title=title_case(x_var)))
    if col_var:
        tooltips.append(alt.Tooltip(VAR_ABBR_MAP[col_var], title=title_case(col_var)))
    if row_var:
        tooltips.append(alt.Tooltip(VAR_ABBR_MAP[row_var], title=title_case(row_var)))
    data = data.rename(columns=VAR_ABBR_MAP)
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            y=alt.Y("sum(v)").title("Capacity (GW)"),
            x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var)),
            color=alt.Color("tt")
            .scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values()))
            .title(title_case("tech_type")),
            opacity=alt.Opacity("y:O", sort="descending").title(
                title_case("planning_year")
            ),
            order=alt.Order(
                # Sort the segments of the bars by this field
                "y",
                sort="ascending",
            ),
            tooltip=tooltips,
            # xOffset=alt.XOffset(VAR_ABBR_MAP.get(xOffset)),
        )
        .properties(height=height, width=width)
    )
    chart = config_chart_row_col(chart, row_var, col_var, x_var)
    return chart


def chart_regional_cap(
    cap: pd.DataFrame,
    group_by=["agg_zone", "tech_type", "model", "planning_year"],
    x_var="model",
    row_var="planning_year",
    order=None,
    width=alt.Step(40),
    height=200,
) -> alt.Chart:
    # Apply default ordering based on x_var if order is not provided
    if order is None and x_var in order_dict():
        order = order_dict()[x_var]
    data = cap.groupby(group_by, as_index=False)["end_value"].sum()
    data["end_value"] /= 1000
    # data = data.rename(columns={"agg_zone": "Region"})
    data = data.rename(columns=VAR_ABBR_MAP)
    data["o"] = data["tt"].map(TECH_STACK_ORDER)

    if x_var=="model": #NOTE: Temp fix where model is named Scenarios. this is because we are using diff scenarios as "models" in this plotting function and csv files
        x_title = "Scenarios"
    else:
        x_title = title_case(x_var)
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(x_title),
            y=alt.Y("ev").title("Capacity (GW)"),
            color=alt.Color("tt").scale(
                domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())
            )
            # .scale(scheme="tableau20")
            .title(title_case("tech_type")),
            column=alt.Column("az")
            .title("Region")
            .header(labelFontSize=15, titleFontSize=20),
            row=alt.Row(VAR_ABBR_MAP[row_var])
            .title(title_case(row_var))
            .header(labelFontSize=15, titleFontSize=20),
            tooltip=[
                alt.Tooltip("tt", title="Technology"),
                alt.Tooltip("ev", title="Capacity (GW)", format=",.0f"),
                alt.Tooltip("az"),
                alt.Tooltip(VAR_ABBR_MAP[row_var], title=title_case(row_var)),
                alt.Tooltip(VAR_ABBR_MAP[x_var], title=title_case(x_var)),
            ],
            order="o",
        )
        .properties(width=width, height=height)
    )
    chart = (
        chart.configure_axis(labelFontSize=15, titleFontSize=15)
        .configure_axis(labelFontSize=15, titleFontSize=15)
        .configure_legend(titleFontSize=20, labelFontSize=16)
    )
    chart = configure_full_label_display(chart)
    return chart


def chart_total_gen(
    gen: pd.DataFrame,
    cap: pd.DataFrame = None,
    x_var="model",
    col_var=None,
    row_var="planning_year",
    order=None,
    width=alt.Step(40),
    height=200,
) -> alt.Chart:
    # Apply default ordering based on x_var if order is not provided
    if order is None and x_var in order_dict():
        order = order_dict()[x_var]
    if gen.empty:
        return None
    merge_by = ["tech_type", "resource_name", x_var, "planning_year"]
    group_by = ["tech_type", x_var, "planning_year"]
    _tooltips = [
        alt.Tooltip("tt", title="Technology"),
        alt.Tooltip("v", title="Generation (TWh)", format=",.0f"),
    ]
    if "new_build" in gen.columns:
        merge_by.append("new_build")
        group_by.append("new_build")
        _tooltips.append(alt.Tooltip("new_build", title="New Build"))
    if col_var is not None:
        group_by.append(col_var)
        merge_by.append(col_var)
        _tooltips.append(alt.Tooltip(VAR_ABBR_MAP[col_var]).title(title_case(col_var)))
    if row_var is not None:
        _tooltips.append(alt.Tooltip(VAR_ABBR_MAP[row_var]).title(title_case(row_var)))
        merge_by.append(row_var)
        group_by.append(row_var)
    merge_by = list(set(merge_by))
    group_by = list(set(group_by))
    if cap is not None:
        _cap = (
            cap.query("unit=='MW'")
            .groupby(
                merge_by,
                # ["tech_type", "resource_name", "model", "planning_year"],
                as_index=False,
            )["end_value"]
            .sum()
        )
        _gen = pd.merge(
            gen,
            _cap,
            # on=["tech_type", "resource_name", "model", "planning_year"],
            on=merge_by,
            how="left",
        )
        _gen.fillna({"end_value": 0}, inplace=True)
        _gen["potential_gen"] = _gen["end_value"] * 8760

        data = _gen.groupby(group_by, as_index=False)[
            ["value", "potential_gen", "end_value"]
        ].sum()
        data["capacity_factor"] = (data["value"] / data["potential_gen"]).round(3)
        _tooltips.extend(
            [
                alt.Tooltip("capacity_factor", title="Capacity Factor"),
                alt.Tooltip("ev", title="Capacity (MW)", format=",.0f"),
            ]
        )

    else:
        data = gen.groupby(group_by, as_index=False)["value"].sum()

    if (Path.cwd() / "annual_demand_genx.csv").exists():
        demand = pd.read_csv(Path.cwd() / "annual_demand_genx.csv")
        demand.loc[:, "agg_zone"] = demand.loc[:, "zone"].map(rev_region_map)
        demand_by_year = demand.groupby(["planning_year"], as_index=False)[
            "annual_demand"
        ].sum()
        data = pd.merge(
            data,
            demand_by_year,
            on=["planning_year"],
            how="left",
        )
        data.loc[:, "annual_demand"] = data["annual_demand"] / 1_000_000
    else:
        demand = None
    data["value"] /= 1_000_000
    data = data.rename(columns=VAR_ABBR_MAP)
    data["o"] = data["tt"].map(TECH_STACK_ORDER)
    if x_var =="model":
        x_var_title = "Scenarios"
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var_title)),
            y=alt.Y("v").title("Generation (TWh)"),
            color=alt.Color("tt")
            .scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values()))
            .title(title_case("tech_type")),
            tooltip=_tooltips,
            order="o",
        )
        .properties(width=width, height=height)
    )
    # if demand is not None:
    #     line = (
    #         alt.Chart()
    #         .mark_rule()
    #         .encode(
    #             y=alt.Y("annual_demand"),
    #         )
    #     )
    #     chart = alt.layer(chart, line, data=data)
    chart = config_chart_row_col(chart, row_var, col_var, x_var)
    return chart


def chart_costs(
    costs: pd.DataFrame,
    x_var: str = "model",
    col_var: Optional[str] = "planning_year",
    row_var: Optional[str] = None,
    order: Optional[List[str]] = None,
    width=alt.Step(40),
    height=200,
) -> Optional[alt.Chart]:
    """Stacked bar chart of cost breakdowns by scenario and planning year."""
    cost_type_color_map = {
        "cCO2": "#af5da4",
        "cFix": "#68A8E4",
        "cVar": "#54B750",
        "cStart": "#DB6565",
        "cNSE": "#EB9696",
        "cFuel": "#F7CD4B",
    }
    if costs.empty:
        return None
    if "cost_type" not in costs.columns:
        raise KeyError("chart_costs requires a 'cost_type' column")

    data = costs.loc[costs["cost_type"] != "cTotal"].copy()
    if data.empty:
        return None

    if "model" in data.columns:
        data["model"] = data["model"].map(SCENARIO_MAPPING_AND_ORDER).fillna(data["model"])

    if "zone" in data.columns:
        data["zone"] = map_zone_values_to_regions(data["zone"])

    if col_var is not None and col_var not in data.columns:
        col_var = None
    if row_var is not None and row_var not in data.columns:
        row_var = None

    if order is None and x_var in order_dict():
        order = order_dict()[x_var]
    if order:
        present = set(data[x_var].unique())
        order = [val for val in order if val in present]
        if not order:
            order = None

    group_by = ["cost_type", x_var]
    for axis in (col_var, row_var):
        if axis is not None:
            group_by.append(axis)
    group_by = [c for c in dict.fromkeys(group_by) if c in data.columns]

    grouped = data.groupby(group_by, as_index=False)["value"].sum()
    grouped["value"] = grouped["value"] / 1e9

    cost_sort = list(dict.fromkeys(grouped["cost_type"].tolist()))

    grouped = grouped.rename(columns=VAR_ABBR_MAP)

    x_field = VAR_ABBR_MAP.get(x_var, x_var)
    x_title = "Scenarios" if x_var == "model" else title_case(x_var)
    x_kwargs: Dict[str, Any] = {"title": x_title}
    if order:
        x_kwargs["sort"] = order
    x_encoding = alt.X(x_field, **x_kwargs)

    y_encoding = alt.Y("v", stack="zero").title("Cost ( Billion USD )")

    color_kwargs: Dict[str, Any] = {"title": "Cost Type"}
    if cost_sort:
        color_kwargs["sort"] = cost_sort
    scale_domain = cost_sort if cost_sort else list(cost_type_color_map.keys())
    if scale_domain:
        color_kwargs["scale"] = alt.Scale(
            domain=scale_domain,
            range=[cost_type_color_map.get(ct, "#999999") for ct in scale_domain],
        )
    color_encoding = alt.Color("ct", **color_kwargs)

    _tooltips = [
        alt.Tooltip("ct", title="Cost Type"),
        alt.Tooltip("v", title="Cost ( Billion USD )", format=",.2f"),
    ]
    scenario_field = VAR_ABBR_MAP.get(x_var, x_var)
    _tooltips.append(alt.Tooltip(scenario_field, title=x_title))
    if col_var is not None:
        _tooltips.append(
            alt.Tooltip(VAR_ABBR_MAP[col_var], title=title_case(col_var))
        )
    if row_var is not None:
        _tooltips.append(
            alt.Tooltip(VAR_ABBR_MAP[row_var], title=title_case(row_var))
        )

    chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=x_encoding,
            y=y_encoding,
            color=color_encoding,
            tooltip=_tooltips,
        )
        .properties(width=width, height=height)
    )

    chart = config_chart_row_col(chart, row_var, col_var, x_var)
    return chart


def chart_revenue(
    revenue: pd.DataFrame,
    x_var: str = "model",
    col_var: Optional[str] = "planning_year",
    row_var: Optional[str] = None,
    order: Optional[List[str]] = None,
    width=alt.Step(40),
    height=200,
) -> Optional[alt.Chart]:
    """Stacked bar chart of net-revenue components by scenario and planning year."""

    component_col = "netrevenue_component"
    if revenue.empty:
        return None
    if component_col not in revenue.columns:
        raise KeyError("chart_revenue requires a 'netrevenue_component' column")

    data = _apply_netrev_sign(revenue)
    if "model" in data.columns:
        data["model"] = data["model"].map(SCENARIO_MAPPING_AND_ORDER).fillna(data["model"])

    if "zone" in data.columns:
        data["zone"] = map_zone_values_to_regions(data["zone"])

    if col_var is not None and col_var not in data.columns:
        col_var = None
    if row_var is not None and row_var not in data.columns:
        row_var = None

    if order is None and x_var in order_dict():
        order = order_dict()[x_var]
    if order:
        present = set(data[x_var].unique())
        order = [val for val in order if val in present]
        if not order:
            order = None

    group_by = [component_col, x_var]
    for axis in (col_var, row_var):
        if axis is not None:
            group_by.append(axis)
    group_by = [c for c in dict.fromkeys(group_by) if c in data.columns]

    grouped = data.groupby(group_by, as_index=False)["value"].sum()
    grouped["value"] = grouped["value"] / 1e9

    component_sort = list(dict.fromkeys(grouped[component_col].tolist()))

    grouped = grouped.rename(columns=VAR_ABBR_MAP)

    x_field = VAR_ABBR_MAP.get(x_var, x_var)
    x_title = "Scenarios" if x_var == "model" else title_case(x_var)
    x_kwargs: Dict[str, Any] = {"title": x_title}
    if order:
        x_kwargs["sort"] = order
    x_encoding = alt.X(x_field, **x_kwargs)

    y_encoding = alt.Y("v", stack="zero").title("Revenue (Billion USD)")

    color_kwargs: Dict[str, Any] = {"title": "Revenue Component"}
    if component_sort:
        color_kwargs["sort"] = component_sort
    color_kwargs["scale"] = alt.Scale(scheme="category20")
    color_encoding = alt.Color("nrc", **color_kwargs)

    tooltips = [
        alt.Tooltip("nrc", title="Revenue Component"),
        alt.Tooltip("v", title="Revenue (Billion USD)", format=",.2f"),
    ]
    scenario_field = VAR_ABBR_MAP.get(x_var, x_var)
    tooltips.append(alt.Tooltip(scenario_field, title=x_title))
    if col_var is not None:
        tooltips.append(
            alt.Tooltip(VAR_ABBR_MAP[col_var], title=title_case(col_var))
        )
    if row_var is not None:
        tooltips.append(
            alt.Tooltip(VAR_ABBR_MAP[row_var], title=title_case(row_var))
        )

    chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=x_encoding,
            y=y_encoding,
            color=color_encoding,
            tooltip=tooltips,
        )
        .properties(width=width, height=height)
    )

    chart = config_chart_row_col(chart, row_var, col_var, x_var)
    return chart


def chart_revenue_costs(
    revenue: pd.DataFrame,
    x_var: str = "tech_type",
    col_var: Optional[str] = None,
    row_var: Optional[str] = "model",
    order: Optional[List[str]] = None,
    width=alt.Step(32),
    height=180,
    scale_divisor: float = 1_000_000,
) -> Optional[alt.Chart]:
    """Facet chart of revenue and cost components by technology with rows per scenario."""

    component_col = "netrevenue_component"
    if revenue.empty:
        return None
    for required in (component_col, x_var):
        if required not in revenue.columns:
            raise KeyError(
                f"chart_revenue_costs requires a '{required}' column"
            )

    data = _apply_netrev_sign(revenue)
    if "model" in data.columns:
        data["model"] = data["model"].map(SCENARIO_MAPPING_AND_ORDER).fillna(data["model"])

    if "zone" in data.columns:
        data["zone"] = map_zone_values_to_regions(data["zone"])

    if col_var is not None and col_var not in data.columns:
        col_var = None
    if row_var is not None and row_var not in data.columns:
        row_var = None

    if order is None and x_var in order_dict():
        order = order_dict()[x_var]
    if order:
        present = set(data[x_var].dropna().unique())
        order = [val for val in order if val in present]
        if not order:
            order = None

    group_by = [component_col, x_var]
    for axis in (row_var, col_var):
        if axis is not None:
            group_by.append(axis)
    if "planning_year" in data.columns:
        group_by.append("planning_year")
    group_by = [c for c in dict.fromkeys(group_by) if c in data.columns]

    grouped = data.groupby(group_by, as_index=False)["value"].sum()
    grouped = grouped.dropna(subset=[x_var])
    if scale_divisor != 0:
        grouped["value"] = grouped["value"] / scale_divisor

    present_components = list(dict.fromkeys(grouped[component_col].tolist()))
    component_sort = _ordered_netrev_components(present_components)

    grouped = grouped.rename(columns=VAR_ABBR_MAP)

    x_field = VAR_ABBR_MAP.get(x_var, x_var)
    x_kwargs: Dict[str, Any] = {"title": title_case(x_var)}
    if order:
        x_kwargs["sort"] = order
    x_encoding = alt.X(x_field, **x_kwargs)

    if math.isclose(scale_divisor, 1_000_000):
        unit_label = "Net Revenue (Million USD)"
    elif math.isclose(scale_divisor, 1_000_000_000):
        unit_label = "Net Revenue (Billion USD)"
    else:
        unit_label = "Net Revenue (USD)"
    y_encoding = alt.Y("v", stack="zero").title(unit_label)

    color_kwargs: Dict[str, Any] = {"title": "Revenue / Cost Type"}
    if component_sort:
        color_kwargs["sort"] = component_sort
        color_kwargs["scale"] = alt.Scale(
            domain=component_sort,
            range=[NET_REVENUE_COLOR_MAP.get(c, "#888888") for c in component_sort],
        )
    color_encoding = alt.Color("nrc", **color_kwargs)

    tooltips = [
        alt.Tooltip("nrc", title="Component"),
        alt.Tooltip("v", title=unit_label, format=",.2f"),
        alt.Tooltip(x_field, title=title_case(x_var)),
    ]
    tooltip_axes = [
        axis
        for axis in (row_var, col_var, "planning_year")
        if axis and axis in data.columns
    ]
    for axis in tooltip_axes:
        if axis != x_var and axis in VAR_ABBR_MAP:
            tooltips.append(
                alt.Tooltip(VAR_ABBR_MAP[axis], title=title_case(axis))
            )

    bar_chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=x_encoding,
            y=y_encoding,
            color=color_encoding,
            tooltip=tooltips,
        )
        .properties(width=width, height=height)
    )

    total_group_fields: List[str] = []
    for col in (x_var, row_var, col_var, "planning_year"):
        col_field = VAR_ABBR_MAP.get(col, col)
        if col and col_field in grouped.columns and col_field not in total_group_fields:
            total_group_fields.append(col_field)

    point_chart = (
        alt.Chart(grouped)
        .transform_aggregate(
            net_total="sum(v)",
            groupby=total_group_fields,
        )
        .mark_point(
            filled=True,
            color="#1F1F1F",
            size=60,
            stroke="#FFFFFF",
            strokeWidth=0.5,
        )
        .encode(
            x=alt.X(x_field, **x_kwargs),
            y=alt.Y("net_total:Q").title(unit_label),
            tooltip=[
                alt.Tooltip(x_field, title=title_case(x_var)),
                alt.Tooltip("net_total:Q", title="Net Total", format=",.2f"),
                *[
                    alt.Tooltip(VAR_ABBR_MAP.get(axis, axis), title=title_case(axis))
                    for axis in tooltip_axes
                    if VAR_ABBR_MAP.get(axis, axis) != x_field
                ],
            ],
        )
        .properties(width=width, height=height)
    )

    chart = alt.layer(bar_chart, point_chart)

    chart = config_chart_row_col(chart, row_var, col_var, x_var)
    return chart


def chart_avg_gen_variation(
    annual_gen_mean: pd.DataFrame,
    x_var: str = "tech_type",
    col_var: str = "case",
    row_var: str = None,
    xOffset: str = None,
    color: str = "tech_type",
    order=None,
    bars_only=False,
    height=200,
    width=alt.Step(60),
) -> alt.Chart:
    data = annual_gen_mean.reset_index()
    for col in ["value", "min", "max"]:
        data[col] = (data[col] / 1000000).round(0)
    tooltips = [
        alt.Tooltip("y", title="Planning Year"),
        alt.Tooltip("v", title="Generation (TWh)", format=",.0f"),
    ]
    if x_var:
        tooltips.append(alt.Tooltip(VAR_ABBR_MAP[x_var], title=title_case(x_var)))
    if col_var:
        tooltips.append(alt.Tooltip(VAR_ABBR_MAP[col_var], title=title_case(col_var)))
    if row_var:
        tooltips.append(alt.Tooltip(VAR_ABBR_MAP[row_var], title=title_case(row_var)))
    # if bars_only:
    #     return bars
    if color == "tech_type":
        _color = (
            alt.Color("tt")
            .scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values()))
            .title(title_case("tech_type"))
        )
    else:
        _color = alt.Color(VAR_ABBR_MAP.get(color)).title(title_case(color))
    if xOffset is None:
        bars = (
            alt.Chart()
            .mark_bar()
            .encode(
                y=alt.Y("sum(v)").title("Generation (TWh)"),
                x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var)),
                color=_color,
                tooltip=tooltips,
            )
            .properties(width=width, height=height)
        )

        error_bars = (
            alt.Chart()
            .mark_errorbar()
            .encode(
                x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var)),
                y=alt.Y("sum(max)").title("Generation (TWh)"),
                y2=alt.Y2("sum(min)"),
            )
            .properties(width=width, height=height)
        )

    else:
        bars = (
            alt.Chart()
            .mark_bar(width=10)
            .encode(
                y=alt.Y("sum(v)").title("Generation (TWh)"),
                x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var)),
                color=_color,
                tooltip=tooltips,
                xOffset=alt.XOffset(VAR_ABBR_MAP.get(xOffset)).title(
                    title_case(xOffset)
                ),
                opacity=alt.Opacity(VAR_ABBR_MAP.get(xOffset)).title(
                    title_case(xOffset)
                ),
            )
            .properties(width=width, height=height)
        )

        error_bars = (
            alt.Chart()
            .mark_errorbar()
            .encode(
                x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var)),
                y=alt.Y("sum(max)").title("Generation (TWh)"),
                y2=alt.Y2("sum(min)"),
                xOffset=alt.XOffset(VAR_ABBR_MAP.get(xOffset)).title(
                    title_case(xOffset)
                ),
            )
            .properties(width=width, height=height)
        )
    data = data.rename(columns=VAR_ABBR_MAP)
    chart = alt.layer(bars, error_bars, data=data)
    chart = config_chart_row_col(chart, row_var, col_var, x_var)
    return chart


def chart_gen_line(
    gen: pd.DataFrame,
    x_var="planning_year",
    x_offset=None,
    col_var="tech_type",
    row_var=None,
    color="model",
    height=150,
    width=140,
) -> alt.Chart:
    by = [x_var]
    if col_var:
        by.append(col_var)
    if row_var:
        by.append(row_var)
    if x_offset:
        by.append(x_offset)
    if color:
        by.append(color)

    data = gen.groupby(by, as_index=False)["value"].sum()
    data["value"] /= 1000000
    data = data.rename(columns=VAR_ABBR_MAP)
    if x_var == "planning_year":
        _x = alt.X(VAR_ABBR_MAP[x_var]).title(title_case(x_var)).axis(format="04d")
    else:
        _x = alt.X(VAR_ABBR_MAP[x_var]).title(title_case(x_var))
    chart = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=_x,
            y=alt.Y("v").title("Generation (TWh)").scale(type="log"),
            color=alt.Color("m").title("Model"),
        )
        .properties(height=height, width=width)
    )
    if col_var is not None and row_var is not None:
        chart = chart.facet(
            column=alt.Column(VAR_ABBR_MAP[col_var])
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15),
            row=alt.Row(VAR_ABBR_MAP[row_var])
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15),
        )
    elif row_var is not None:
        chart = chart.facet(
            row=alt.Row(VAR_ABBR_MAP[row_var])
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    elif col_var is not None:
        chart = chart.facet(
            column=alt.Column(VAR_ABBR_MAP[col_var])
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15)
        )

    chart = chart.configure_axis(labelFontSize=13, titleFontSize=15).configure_legend(
        titleFontSize=20, labelFontSize=16
    )
    chart = configure_full_label_display(chart)
    return chart


def chart_regional_gen(
    gen: pd.DataFrame,
    cap: pd.DataFrame,
    width=alt.Step(40),
    height=200,
) -> alt.Chart:

    if cap is not None:
        _cap = (
            cap.query("unit=='MW'")
            .groupby(
                ["tech_type", "resource_name", "model", "planning_year"], as_index=False
            )["end_value"]
            .sum()
        )
        _gen = pd.merge(
            gen,
            _cap,
            on=["tech_type", "resource_name", "model", "planning_year"],
            how="left",
        )
        _gen.fillna({"end_value": 0}, inplace=True)
        _gen["potential_gen"] = _gen["end_value"] * 8760
        data = _gen.groupby(
            ["agg_zone", "tech_type", "model", "planning_year"], as_index=False
        )[["value", "potential_gen"]].sum()
        data["capacity_factor"] = (data["value"] / data["potential_gen"]).round(3)
        _tooltips = [
            alt.Tooltip("tt", title="Technology"),
            alt.Tooltip("v", title="Generation (TWh)", format=",.0f"),
            alt.Tooltip("capacity_factor", title="Capacity Factor"),
        ]
    else:
        data = gen.groupby(
            ["agg_zone", "tech_type", "model", "planning_year"], as_index=False
        )["value"].sum()
        _tooltips = [
            alt.Tooltip("tt", title="Technology"),
            alt.Tooltip("value", title="Generation (TWh)", format=",.0f"),
        ]
    data = data.astype({"agg_zone": "object"})
    mapped_zones = data["agg_zone"].map(rev_region_map)
    data.loc[:, "agg_zone"] = mapped_zones.where(mapped_zones.notna(), data["agg_zone"])
    data.loc[:, "agg_zone"] = data["agg_zone"].astype(str)
    if (Path.cwd() / "annual_demand_genx.csv").exists():
        demand = pd.read_csv(Path.cwd() / "annual_demand_genx.csv")
        zone_idx = demand["zone"].astype(str).str.extract(r"(\d+)")[0].astype("Int64")
        mapped_demand_zones = zone_idx.map(
            lambda z: rev_region_map.get(int(z)) if pd.notna(z) else None
        )
        demand.loc[:, "agg_zone"] = mapped_demand_zones.where(
            mapped_demand_zones.notna(), demand["zone"]
        )
        demand_totals = (
            demand.groupby(["agg_zone", "planning_year"], as_index=False)["annual_demand"].sum()
        )
        data = pd.merge(
            data,
            demand_totals,
            on=["agg_zone", "planning_year"],
            how="left",
        )
        data.loc[:, "annual_demand"] = data["annual_demand"] / 1_000_000
    else:
        demand = None
    data["value"] /= 1_000_000
    data = data.rename(mapper=VAR_ABBR_MAP, axis="columns")

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("m").title("Model"),
            y=alt.Y("v").title("Generation (TWh)"),
            color=alt.Color("tt").scale(
                domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())
            )
            # .scale(scheme="tableau20")
            .title(title_case("tech_type")),
            # column="agg_zone",
            # row="planning_year:O",
            tooltip=_tooltips,
        )
        .properties(width=width, height=height)
    )

    # if demand is not None:
    #     line = (
    #         alt.Chart(data)
    #         .mark_rule()
    #         .encode(
    #             y=alt.Y("annual_demand"),
    #         )
    #         .properties(width=width, height=height)
    #     )
    #     chart = alt.layer(chart, line)

    chart = chart.facet(
        column=alt.Column("az")
        .title("Region")
        .header(titleFontSize=20, labelFontSize=15),
        row=alt.Row("y")
        .title(title_case("planning_year"))
        .header(titleFontSize=20, labelFontSize=15),
    )

    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
        titleFontSize=20, labelFontSize=16
    )
    chart = configure_full_label_display(chart)
    return chart


def chart_tx_expansion(
    tx_data: pd.DataFrame,
    x_var="model",
    facet_col="transmission_path_name",
    n_cols=10,
    order=None,
    height=200,
    width=alt.Step(20),
) -> alt.Chart:
    if tx_data.empty:
        return None
    if tx_data[x_var].nunique() < 4:
        width = 80
        
    tx_data["transmission_path_name"] = tx_data["transmission_path_name"].str.replace("_to_", " | ")

    group_cols = [x_var, "planning_year"]
    if facet_col is not None:
        group_cols.append(facet_col)
    data = tx_data.groupby(group_cols, as_index=False)["New_Trans_Capacity"].sum()

    _tooltip = [
        alt.Tooltip("sum(New_Trans_Capacity):Q", format=",.0f", title="Transmission Expansion (MW)"),
        alt.Tooltip("y", title=title_case("planning_year")),
    ]
    if facet_col == "transmission_path_name":
        _tooltip.append(
            alt.Tooltip(VAR_ABBR_MAP["transmission_path_name"], title=title_case("transmission_path_name"))
        )

    if order is None:
        order = sorted(data[x_var].unique())

    x_title = "Scenario" if x_var == "model" else title_case(x_var)

    first_year = tx_data["planning_year"].min()
    key_cols = [c for c in group_cols if c != "planning_year"]
    if "start_value" not in tx_data.columns:
        raise KeyError("chart_tx_expansion requires 'start_value' to compute the baseline capacity.")
    baseline = (
        tx_data.loc[tx_data["planning_year"] == first_year, key_cols + ["start_value"]]
        .groupby(key_cols, as_index=False)
        .agg(baseline_start_value=("start_value", "sum"))
    )
    data = data.merge(baseline, on=key_cols, how="left")
    data["baseline_start_value"] = data["baseline_start_value"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data["pct_of_start_value"] = 0.0
    mask = data["baseline_start_value"] > 0
    data.loc[mask, "pct_of_start_value"] = (
        data.loc[mask, "New_Trans_Capacity"] / data.loc[mask, "baseline_start_value"]
    )
    _tooltip.append(
        alt.Tooltip("baseline_start_value:Q", format=",.0f", title="Start value (MW)")
    )
    _tooltip.append(
        alt.Tooltip("pct_of_start_value", format=".1%", title="Share of start value")
    )

    if x_var == "case":
        _tooltip.append(alt.Tooltip("c", title="Case"))

    scenario_field = VAR_ABBR_MAP.get(x_var, x_var)
    _tooltip.append(alt.Tooltip(scenario_field, title=x_title))
    # data["New_Trans_Capacity"] /= 1000
    data = data.rename(columns=VAR_ABBR_MAP)
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            # xOffset="model:N",
            x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(x_title),
            y=alt.Y("sum(New_Trans_Capacity):Q").title("Transmission Expansion (MW)"),
            color=alt.Color("y:O", sort="descending").title(
                title_case("planning_year")
            ),
            # opacity=alt.Opacity("y:O", sort="descending").title(
            #     title_case("planning_year")
            # ),
            # facet=alt.Facet("line_name", columns=n_cols),
            order=alt.Order(
                # Sort the segments of the bars by this field
                "y",
                sort="ascending",
            ),
            tooltip=_tooltip,
        )
        .properties(
            height=height,
            width=width,
        )
    )
    if facet_col is not None:
        chart = chart.encode(
            facet=alt.Facet(VAR_ABBR_MAP[facet_col], columns=n_cols)
            .title(title_case(facet_col))
            .header(titleFontSize=20, labelFontSize=12)
        )
    else:
        text = (
            alt.Chart(data)
            .mark_text(dy=-5, fontSize=14)
            .encode(
                x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(x_title),
                y="sum(New_Trans_Capacity):Q",
                text=alt.Text("sum(New_Trans_Capacity):Q", format=".0f"),
            )
        )
        chart = alt.layer(chart, text).properties(width=width)
    chart = chart.configure_axis(labelFontSize=12, titleFontSize=13,titlePadding=10).configure_legend(
        titleFontSize=15, labelFontSize=15
    )
    if facet_col is None:
        chart = (
            chart.properties(height=400, width=300)
            .configure_axis(labelFontSize=12, titleFontSize=16,titlePadding=15)
        )
    # chart = configure_full_label_display(chart)
    # chart = (
    #     chart.configure_axis(
            # labelLimit=0,
    #         labelPadding=2, 
    #         titlePadding=10
    #         )
    #     # .configure_legend(labelLimit=0)
    #     # .configure_header(labelLimit=0)
    # )
    return chart


def chart_emissions_intensity(
    emiss,
    gen,
    x_var="Region",
    x_offset=None,
    col_var="model",
    row_var="planning_year",
    height=150,
    width=alt.Step(40),
) -> alt.Chart:

    by = [x_var]
    if col_var:
        by.append(col_var)
    if row_var:
        by.append(row_var)
    if x_offset:
        by.append(x_offset)
    emiss["Region"] = emiss["zone"].map(rev_region_map)
    gen["Region"] = gen["zone"].map(rev_region_map)

    emiss_data = emiss.groupby(by, as_index=False)["value"].sum()
    emiss_data = emiss_data.rename(columns={"value": "emissions"})
    gen_data = gen.groupby(by, as_index=False)["value"].sum()
    gen_data = gen_data.rename(columns={"value": "generation"})

    data = pd.merge(emiss_data, gen_data, on=by)
    data["emissions_intensity"] = data["emissions"] / data["generation"]
    data["emissions_intensity"] *= 1000
    data = data.rename(columns=VAR_ABBR_MAP)
    if x_var == "model":
        x_title = "Scenarios"
    else:
        x_title = title_case(x_var)

    # Apply default ordering based on x_var
    x_sort = order_dict().get(x_var) if x_var in order_dict() else None

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(VAR_ABBR_MAP[x_var]).sort(x_sort).title(x_title),
            y=alt.Y("emissions_intensity").title("kg/MWh"),
        )
        .properties(height=height, width=width)
    )
    chart = config_chart_row_col(chart, row_var, col_var, x_var)

    return chart


def chart_emissions(
    emiss: pd.DataFrame,
    x_var="model",
    row_var="planning_year",
    col_var=None,
    order=None,
    co2_limit=True,
    width=alt.Step(40),
    height=200,
) -> alt.FacetChart | alt.LayerChart:
    """
    Create a faceted or layered Altair chart visualizing CO2 emissions by region and model.

    Parameters
    ----------
    emiss : pd.DataFrame
        DataFrame containing emissions data with columns including 'zone', 'model', 'planning_year', and 'value'.
    x_var : str, optional
        The variable to use for the x-axis (default is "model").
    row_var : str, optional
        The variable to facet rows by (default is "planning_year").
    col_var : str, optional
        The variable to facet columns by (default is None).
    order : list or None, optional
        Order for sorting x-axis values (default is None).
    co2_limit : bool, optional
        Whether to display a CO2 limit line (default is True).
    width : int or alt.Step, optional
        Chart width (default is alt.Step(40)).
    height : int, optional
        Chart height (default is 200).

    Returns
    -------
    alt.FacetChart or alt.LayerChart
        An Altair chart object visualizing emissions by region and model, faceted by planning year and optionally by column variable.

    Expected DataFrame Structure
    ---------------------------
    emiss should contain at least the following columns:
        - 'zone': Region identifier (numeric or string)
        - 'model': Model name
        - 'planning_year': Year of planning (int)
        - 'value': Emissions value (numeric)
    """
    # Apply default ordering based on x_var if order is not provided
    if order is None and x_var in order_dict():
        order = order_dict()[x_var]
    if emiss.empty:
        return None
    _tooltips = [
        alt.Tooltip("sum(v)", format=",.0f", title="Million Tonnes"),
        alt.Tooltip("r", title="Region"),
    ]
    emiss["Region"] = emiss["zone"].map(rev_region_map)
    group_by = ["Region", x_var]
    if row_var is not None and row_var not in group_by:
        group_by.append(row_var)
    if col_var is not None and col_var not in group_by:
        group_by.append(col_var)

    data = emiss.groupby(group_by, as_index=False)["value"].sum()
    if col_var is not None:
        _tooltips.append(alt.Tooltip(VAR_ABBR_MAP[col_var]))
    if order is None:
        order = sorted(data[x_var].unique())
    data.loc[:, "value"] = data["value"] / 1e6
    data["limit"] = 0 # NOTE: this is for the horizontal black line denoting co2 limit for a given year
    # data.loc[data["planning_year"] == 2027, "limit"] = 873 #NOTE: these are for the co2 limits for each year. Currently commented out as these values need verifying
    # data.loc[data["planning_year"] == 2030, "limit"] = 186
    # data.loc[data["planning_year"] == 2035, "limit"] = 130
    # data.loc[data["planning_year"] == 2040, "limit"] = 86.7
    # data.loc[data["planning_year"] == 2045, "limit"] = 43.3
    data = data.rename(columns=VAR_ABBR_MAP)
    # Only include colors (and legend entries) for regions present in this dataset
    data_regions = [r for r in data["r"].unique().tolist() if pd.notna(r)]
    present_regions = [r for r in REGION_COLOR_MAP.keys() if r in data_regions]
    missing_regions = [r for r in data_regions if r not in REGION_COLOR_MAP]
    if missing_regions:
        warnings.warn(
            f"chart_emissions: Regions missing from REGION_COLOR_MAP: {missing_regions}. "
            "They will not use the explicit palette."
        )
    x_axis = alt.Axis(title=title_case(x_var))
    if x_var == "model":
        x_axis = alt.Axis(
            title="Scenarios", # NOTE: Temporary fix where model is named Scenarios. This is because we are using different scenarios as "models" in this plotting function and csv files
            # titlePadding=35,
            # labelPadding=10,
        )
    x_encoding = alt.X(VAR_ABBR_MAP[x_var], sort=order, axis=x_axis)

    base = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=x_encoding,
            y=alt.Y("sum(v)").title("CO2 (Million Tonnes)"),
            color=alt.Color("r")
            .scale(
                domain=present_regions,
                range=[REGION_COLOR_MAP[r] for r in present_regions],
            )
            .title("Region"),
            tooltip=_tooltips,
        )
    )
    text = (
        alt.Chart(data)
        .mark_text(dy=-5)
        .encode(
            x=x_encoding,
            y="sum(v):Q",
            text=alt.Text("sum(v):Q", format=",.0f"),
        )
    )
    if co2_limit:
        size = 2
    else:
        size = 0
    line = (
        alt.Chart(data)
        .mark_rule(size=size)
        .encode(
            y=alt.Y("limit"),
        )
    )
    chart = alt.layer(base, text, line).properties(width=width, height=height)
    chart = config_chart_row_col(chart, row_var, col_var, x_var)
    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
        titleFontSize=20, labelFontSize=16
    )
    chart = configure_full_label_display(chart)
    return chart

def safe_chart_dispatch_single_tech(data: pd.DataFrame, context: Optional[str] = None):
    """Render chart_dispatch_single_tech while keeping the notebook resilient."""
    try:
        return chart_dispatch_single_tech(data)
    except Exception as exc:  # pragma: no cover - VIS chart helper
        context_msg = f" for {context}" if context else ""
        print(f"chart_dispatch_single_tech failed{context_msg}: {exc}")
        return None
    
def chart_dispatch(data: pd.DataFrame) -> alt.Chart:
    # Map zone numbers to region names
    data["zone"] = data["zone"].map(rev_region_map)
    data = data.rename(
        columns={
            "model": "m",
            "tech_type": "tt",
            "zone": "z",
            "hour": "h",
            "value": "v",
        }
    )
    selection = alt.selection_point(fields=["model"], bind="legend")
    data["v"] /= 1000
    chart = (
        alt.Chart(data)
        .mark_line()
        .encode(
            x=alt.X("h").title("Hour").axis(values=list(range(0, 169, 24))),
            y=alt.Y("v").title("Dispatch (GW)"),
            color=alt.Color("m").legend(title="Model"),
            row=alt.Row("tt")
            .title("Tech Type")
            .header(titleFontSize=20, labelFontSize=15),
            column=alt.Column("z")
            .title("Region")
            .header(titleFontSize=20, labelFontSize=15),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        )
        .properties(width=250, height=150)
        .add_params(selection)
    ).resolve_scale(y="independent")
    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
        titleFontSize=18, labelFontSize=16
    )
    chart = configure_full_label_display(chart)
    return chart

def chart_dispatch_single_tech(data: pd.DataFrame) -> alt.Chart:
    # Map zone numbers to region names
    data["zone"] = data["zone"].map(rev_region_map)
    if "tech_type" not in data.columns:
        raise ValueError(
            "chart_dispatch_single_tech requires a 'tech_type' column to verify a single technology."
        )
    unique_techs = data["tech_type"].dropna().unique()
    if len(unique_techs) != 1:
        raise ValueError(
            "chart_dispatch_single_tech expects data for exactly one tech_type, "
            f"but received {len(unique_techs)} unique values: {unique_techs.tolist()}"
        )
    data = data.rename(
        columns={
            "model": "m",
            "tech_type": "tt",
            "zone": "z",
            "hour": "h",
            "value": "v",
        }
    )
    data = data.drop(columns=["tt"], errors="ignore")
    data["v"] /= 1000
    selection = alt.selection_point(fields=["model"], bind="legend")
    if "cluster" in data.columns:
        chart = (
            alt.Chart(data)
            .mark_line()
            .encode(
                x=alt.X("h").title("Hour").axis(values=list(range(0, 169, 24))),
                y=alt.Y("v").title("Dispatch (GW)"),
                color=alt.Color("m").legend(title="Model"),
                strokeDash="cluster",
                facet=alt.Facet("z", columns=3)
                .title("Zone")
                .header(titleFontSize=20, labelFontSize=15),
                opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            )
            .properties(width=250, height=150)
            .add_params(selection)
        ).resolve_scale(y="independent")
    else:
        chart = (
            alt.Chart(data)
            .mark_line()
            .encode(
                x=alt.X("h").title("Hour"),
                y=alt.Y("v").title("Dispatch (GW)"),
                color=alt.Color("m").legend(title="Model"),
                facet=alt.Facet("z", columns=3)
                .title("Zone")
                .header(titleFontSize=20, labelFontSize=15),
                opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            )
            .properties(width=250, height=150)
            .add_params(selection)
        ).resolve_scale(y="independent")
    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
        titleFontSize=18, labelFontSize=16
    )
    chart = configure_full_label_display(chart)
    return chart


# Calculate NPV for each cost category 
#
def calculate_npv(
    df: pd.DataFrame, period_len: Dict[int, int], discount_rate: float, base_year: int
):
    ''' 
    NPV is a financial metric used to evaluate the profitability of an investment by discounting future cash flows to their present value. 
    '''
    "From ChatGPT"
    df["Total"] = 0.0
    for index, row in df.iterrows():
        annual_cost = row["annual_cost"]
        planning_year = row["planning_year"]
        length = period_len[planning_year]
        npv = 0.0
        for i in range(length):
            year = planning_year - i
            discount_factor = 1 / (1 + discount_rate) ** (year - base_year)
            npv += annual_cost * discount_factor
        df.at[index, "Total"] = npv
    return df


def append_npv_cost(op_costs: pd.DataFrame) -> pd.DataFrame:
    period_len = {
        # 2027: 4,
        2030: 3,
        2035: 5,
        2040: 5,
        2045: 5,
        2050: 5,
    }
    discount_rate = 0.02
    annual_costs = (
        op_costs.groupby(["Costs", "planning_year", "model"], as_index=False)["Total"]
        .sum()
        .rename(columns={"Total": "annual_cost"})
    )

    npv_costs = (
        calculate_npv(annual_costs, period_len, discount_rate, 2024)
        .groupby(["model", "Costs"], as_index=False)["Total"]
        .sum()
        .pipe(calc_op_percent_total, ["model"])
    )
    npv_costs["planning_year"] = "NPV"

    return pd.concat([op_costs, npv_costs])


def single_op_cost_chart(
    data,
    x_var="model",
    col_var=None,
    row_var=None,
    order=None,
    width=alt.Step(40),
    height=200,
) -> alt.Chart:
    _tooltip = [alt.Tooltip("Total", format=",.0f").title("Cost")]
    chart_cols = ["Costs", "Total", VAR_ABBR_MAP[x_var]]

    if "percent_total" in data.columns:
        _tooltip.append(alt.Tooltip("percent_total:Q", format=".1%").title("% Total"))
        chart_cols.append("percent_total")
    if col_var is not None:
        _tooltip.append(alt.Tooltip(VAR_ABBR_MAP[col_var]).title(title_case(col_var)))
        _tooltip.append(alt.Tooltip("Costs").title("Category"))
        chart_cols.append(VAR_ABBR_MAP[col_var])
    if row_var is not None:
        _tooltip.append(alt.Tooltip(VAR_ABBR_MAP[row_var]).title(title_case(row_var)))
        chart_cols.append(VAR_ABBR_MAP[row_var])
    data = data.rename(columns=VAR_ABBR_MAP)
    base = (
        alt.Chart()
        .mark_bar()
        .encode(
            # xOffset="model:N",
            x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var)),
            y=alt.Y("Total").title("Costs (Billion $)"),
            color=alt.Color("Costs:N").title("Category"),
            tooltip=_tooltip,
        )
    )

    text = (
        alt.Chart()
        .mark_text(dy=-5, fontSize=11)
        .encode(
            x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var)),
            y="sum(Total):Q",
            text=alt.Text("sum(Total):Q", format=".0f"),
        )
    )

    chart = alt.layer(
        base,
        text,
        data=data[chart_cols].query("Total!=0 and Costs != 'cTotal'"),
    ).properties(width=width, height=height)

    final_chart = chart
    if row_var is None and col_var is None:
        pass
    elif row_var is None and col_var is not None:
        final_chart = chart.facet(
            column=alt.Column(VAR_ABBR_MAP[col_var])
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    elif col_var is None and row_var is not None:
        final_chart = chart.facet(
            row=alt.Row(VAR_ABBR_MAP[row_var])
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    else:
        final_chart = chart.facet(
            row=alt.Row(VAR_ABBR_MAP[row_var])
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15),
            column=alt.Column(VAR_ABBR_MAP[col_var])
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15),
        )

    return configure_full_label_display(final_chart)


def chart_op_cost(
    op_costs: pd.DataFrame, x_var="model", col_var=None, row_var=None, order=None
) -> alt.Chart:
    if op_costs.empty:
        return None
    if "NPV" in op_costs["planning_year"].unique():
        npv_costs = op_costs.loc[op_costs["planning_year"] == "NPV", :]
        op_costs = op_costs.loc[~(op_costs["planning_year"] == "NPV"), :]
    else:
        npv_costs = pd.DataFrame()
    if col_var is None and "planning_year" in op_costs.columns:
        col_var = "planning_year"

    data = op_costs.copy()
    data["Total"] /= 1e9
    data = data.rename(columns=VAR_ABBR_MAP)

    chart = single_op_cost_chart(data, x_var, col_var, row_var, order)
    if not npv_costs.empty:
        npv_data = npv_costs.copy()
        npv_data = npv_data.rename(columns=VAR_ABBR_MAP)
        npv_data["Total"] /= 1e9
        npv_chart = single_op_cost_chart(npv_data, x_var, col_var, row_var, order)
        if col_var == "planning_year":
            final_chart = chart | npv_chart
        elif row_var == "planning_year":
            final_chart = chart & npv_chart
        else:
            final_chart = chart
    else:
        final_chart = chart # NOTE: Keeping the original chart if no NPV data is present. Added this to avoid error ehn disabling calculate npv func in load_genx_operations_data()
    final_chart = (
        final_chart.configure_axis(labelFontSize=15, titleFontSize=15)
        .configure_axis(labelFontSize=15, titleFontSize=15)
        .configure_legend(titleFontSize=20, labelFontSize=16)
    )
    final_chart = configure_full_label_display(final_chart)
    return final_chart


def chart_op_nse(
    op_nse: pd.DataFrame,
    x_var="model",
    col_var=None,
    row_var=None,
    order=None,
    height=200,
    width=alt.Step(40),
) -> alt.Chart:
    cols = ["Segment", "Total", "model"]
    if "planning_year" in op_nse and row_var != "planning_year":
        col_var = "planning_year"
    if col_var is not None:
        cols.append(col_var)
    if row_var is not None:
        cols.append(row_var)
    if op_nse.empty:
        return None
    data = op_nse.copy()
    data["value"] /= 1000
    data = data.rename(columns=VAR_ABBR_MAP)
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            # xOffset="model:N",
            x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var)),
            y=alt.Y("sum(v)").title("Annual non-served GWh"),
            # color=alt.Color("model:N").title(title_case("model")),
            tooltip=alt.Tooltip("sum(v)", format=",.0f", title="NSE"),
        )
    .properties(width=width_px, height=height_px)
    )
    chart = config_chart_row_col(chart, row_var, col_var, x_var)
    return chart


def chart_op_emiss(
    op_emiss: pd.DataFrame,
    x_var="model",
    color="tech_type",
    col_var=None,
    row_var=None,
    order=None,
    width=alt.Step(40),
    height=200,
) -> alt.Chart:
    if op_emiss.empty:
        return None
    op_emiss["r"] = op_emiss["zone"].map(rev_region_map)
    if (
        col_var is None
        and "planning_year" in op_emiss.columns
        and row_var != "planning_year"
    ):
        col_var = "planning_year"
    _tooltip = [
        alt.Tooltip("v", format=",.0f", title="Emissions"),
        alt.Tooltip(color),
    ]
    by = [color, x_var]

    color_scale = "category10"
    if op_emiss[color].nunique() > 10:
        color_scale = "tableau20"
    if col_var is not None:
        _tooltip.append(alt.Tooltip(VAR_ABBR_MAP[col_var]).title(title_case(col_var)))
        by.append(col_var)
    if row_var is not None:
        _tooltip.append(alt.Tooltip(VAR_ABBR_MAP[row_var]).title(title_case(row_var)))
        by.append(row_var)

    by = list(set(by))
    data = op_emiss.groupby(by, as_index=False)["value"].sum().query("value>0")
    data["value"] /= 1e6
    data = data.rename(columns=VAR_ABBR_MAP)
    # If coloring by Region, use explicit palette and limit legend to present regions
    if color == "Region":
        data_regions = [r for r in data["r"].unique().tolist() if pd.notna(r)]
        present_regions = [r for r in REGION_COLOR_MAP.keys() if r in data_regions]
        missing_regions = [r for r in data_regions if r not in REGION_COLOR_MAP]
        if missing_regions:
            warnings.warn(
                f"chart_op_emiss: Regions missing from REGION_COLOR_MAP: {missing_regions}. "
                "They will not use the explicit palette."
            )
        region_color_encoding = (
            alt.Color(VAR_ABBR_MAP[color])
            .scale(
                domain=present_regions,
                range=[REGION_COLOR_MAP[r] for r in present_regions],
            )
            .title(title_case(color))
        )
    else:
        region_color_encoding = (
            alt.Color(VAR_ABBR_MAP[color]).scale(scheme=color_scale).title(
                title_case(color)
            )
        )
    base = (
        alt.Chart()
        .mark_bar()
        .encode(
            # xOffset="model:N",
            x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var)),
            y=alt.Y("v").title("CO2 (Million Tonnes)"),
            color=region_color_encoding,
            tooltip=_tooltip,
        )
    )

    text = (
        alt.Chart()
        .mark_text(dy=-5, fontSize=11)
        .encode(
            x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var)),
            y="sum(v):Q",
            text=alt.Text("sum(v):Q", format=".0f"),
        )
    )

    chart = alt.layer(
        base,
        text,
        data=data,
    ).properties(width=width, height=height)

    chart = config_chart_row_col(chart, row_var, col_var, x_var)
    return chart

# Geospatial utilities for transmission lines and regions
TX_LINE_SHP_PATH = (
    Path(__file__).parent
    / "interregional_connections_NYISO"
    / "interregional_connections_NYISO_neha.shp"
)

REGION_SHP_PATH = (
    Path(__file__).parent
    / "agg_IMP_Region"
    / "aggregated_regions.shp"
)

# transmission map functions

DEFAULT_REGION_COLOR = "#E2E6EC"

def _region_key(df: pd.DataFrame, start_col: str, end_col: str, sep: str = "||") -> pd.Series:
    cols = df[[start_col, end_col]].astype(str).fillna("")
    return cols.apply(lambda row: sep.join(sorted(row.tolist())), axis=1)


def _resolve_column_name(
    columns: Iterable[str],
    preferred: Optional[str],
    fallbacks: Iterable[str],
) -> Optional[str]:
    """Return the first column present in *columns* from preferred+fallback candidates."""
    seen: Set[str] = set()
    candidates: List[Optional[str]] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(fallbacks)
    for candidate in candidates:
        if candidate and candidate not in seen and candidate in columns:
            return candidate
        if candidate:
            seen.add(candidate)
    return None


def _prepare_tx_exp_data(
    tx_exp: pd.DataFrame,
    scenario_field: str,
    year_field: str,
    result_col: str,
    *,
    require_scenario: bool,
    require_year: bool,
    missing_scenario_msg: str,
    missing_year_msg: str,
    missing_result_msg: str,
    missing_path_msg: str,
) -> Tuple[pd.DataFrame, Optional[str], Optional[str], str]:
    """Normalize transmission expansion inputs and resolve column names."""
    data = tx_exp.copy()

    scenario_col = _resolve_column_name(data.columns, scenario_field, ["case", "model"])
    if scenario_col is None and require_scenario:
        raise KeyError(missing_scenario_msg)

    year_col = _resolve_column_name(data.columns, year_field, ["planning_year"])
    if year_col is None and require_year:
        raise KeyError(missing_year_msg)

    result_candidates = [result_col, "value", "New_Trans_Capacity"]
    result_resolved = _resolve_column_name(data.columns, result_candidates[0], result_candidates[1:])
    if result_resolved is None:
        raise KeyError(missing_result_msg)

    if "line_name" not in data.columns:
        fallback_line_col = next(
            (col for col in ("transmission_path_name", "line") if col in data.columns),
            None,
        )
        if fallback_line_col is not None:
            data["line_name"] = data[fallback_line_col]
        elif {"start_region", "dest_region"}.issubset(data.columns):
            data["line_name"] = data["start_region"].astype(str) + "_to_" + data["dest_region"].astype(str)
        else:
            data["line_name"] = "line"

    if "start_region" not in data.columns or "dest_region" not in data.columns:
        path_col = next((col for col in ("line_name", "transmission_path_name") if col in data.columns), None)
        if path_col is None:
            raise KeyError(missing_path_msg)
        regions = data[path_col].astype(str).str.split("_to_", n=1, expand=True)
        data["start_region"] = regions[0]
        data["dest_region"] = regions[1]

    data["start_region"] = data["start_region"].apply(_normalize_zone_name)
    data["dest_region"] = data["dest_region"].apply(_normalize_zone_name)

    return data, scenario_col, year_col, result_resolved


def _build_tx_map_assets(
    grouped: pd.DataFrame,
    result_col: str,
    focus_regions: Tuple[str, ...],
    bbox_buffer: float,
    region_simplify_tol: Optional[float],
    line_simplify_tol: Optional[float],
    show_full_map: bool,
    *,
    simplify_lines_when_full_map: bool,
) -> Optional[Tuple[gpd.GeoDataFrame, List[Dict[str, Any]], Dict[str, Any]]]:
    """Return merged line geometries plus background features and projection settings."""
    gdf = load_tx_line_geometries()
    if "region_key" not in gdf.columns:
        gdf = gdf.copy()
        gdf["region_key"] = _region_key(gdf, "region_1", "region_2")

    grouped = grouped.copy()
    if "region_key" not in grouped.columns:
        grouped["region_key"] = _region_key(grouped, "start_region", "dest_region")

    extent_keys = grouped["region_key"].unique().tolist()
    merged = (
        grouped.merge(
            gdf[["region_key", "geometry"]],
            on="region_key",
            how="left",
        )
        .dropna(subset=["geometry"])
        .reset_index(drop=True)
    )
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=getattr(gdf, "crs", None))
    if merged.empty:
        raise ValueError("No transmission lines matched the shapefile geometries.")

    simplify_lines = (
        line_simplify_tol is not None
        and line_simplify_tol > 0
        and (simplify_lines_when_full_map or not show_full_map)
    )
    if simplify_lines:
        merged = merged.copy()
        merged["geometry"] = merged.geometry.simplify(line_simplify_tol, preserve_topology=False)
        merged = merged.loc[~merged.geometry.is_empty]
        if merged.empty:
            return None

    if show_full_map:
        line_extent_subset = gdf.geometry
    else:
        line_extent_subset = gdf.loc[gdf["region_key"].isin(extent_keys)].geometry
        if line_extent_subset.empty:
            line_extent_subset = gdf.geometry
    line_extent_union = line_extent_subset.unary_union
    if isinstance(line_extent_union, (list, tuple)):
        line_extent_union = gpd.GeoSeries(line_extent_subset).unary_union
    if hasattr(line_extent_union, "is_empty") and line_extent_union.is_empty:
        line_extent_union = None

    _, (region_extent_geom, region_extent_crs) = _region_background_cache(focus_regions, bbox_buffer)
    extent_geom = region_extent_geom
    if not show_full_map and line_extent_union is not None and extent_geom is not None:
        extent_geom = extent_geom.union(line_extent_union)
    if bbox_buffer and bbox_buffer > 0 and extent_geom is not None:
        extent_geom = extent_geom.buffer(bbox_buffer)

    extent_crs = region_extent_crs or "EPSG:4326"
    extent_df = gpd.GeoDataFrame({"id": [0]}, geometry=[extent_geom], crs=extent_crs)
    if extent_df.crs is None:
        extent_df.set_crs(epsg=4326, inplace=True)
    elif extent_df.crs.to_epsg() != 4326:
        extent_df = extent_df.to_crs(epsg=4326)
    projection_fit = _geodf_to_feature_collection(extent_df)
    projection_kwargs = {"type": "mercator", "fit": projection_fit}

    region_gdf_full = load_region_geometries()
    if show_full_map or extent_geom is None:
        background_gdf = region_gdf_full.copy()
    else:
        background_gdf = region_gdf_full.loc[region_gdf_full.geometry.intersects(extent_geom)].copy()
        if background_gdf.empty:
            background_gdf = region_gdf_full.copy()

    if region_simplify_tol and region_simplify_tol > 0:
        simplified = background_gdf.copy()
        simplified["geometry"] = simplified.geometry.simplify(
            region_simplify_tol,
            preserve_topology=False,
        )
        simplified = simplified.loc[~simplified.geometry.is_empty]
        if not simplified.empty:
            background_gdf = simplified

    base_region_features = _geodf_to_feature_values(background_gdf)
    return merged, base_region_features, projection_kwargs


@lru_cache(maxsize=1)
def load_tx_line_geometries(shapefile_path: str | Path = TX_LINE_SHP_PATH) -> gpd.GeoDataFrame:
    path = Path(shapefile_path)
    if not path.exists():
        raise FileNotFoundError(f"Transmission shapefile not found at {path}")

    geodf = gpd.read_file(path)
    rename_map = {
        "Region_1": "region_1",
        "Region_2": "region_2",
        "REGION_1": "region_1",
        "REGION_2": "region_2",
        "region1": "region_1",
        "region2": "region_2",
        "Line_Name": "line_name",
        "LINE_NAME": "line_name",
        "LineName": "line_name",
    }
    geodf = geodf.rename(columns={k: v for k, v in rename_map.items() if k in geodf.columns})

    required_cols = {"region_1", "region_2", "geometry"}
    missing_cols = required_cols - set(geodf.columns)
    if missing_cols:
        raise ValueError(
            f"Missing expected columns {missing_cols} in transmission shapefile {path}"
        )

    if geodf.crs is not None and geodf.crs.to_epsg() != 4326:
        geodf = geodf.to_crs(epsg=4326)

    geodf = geodf.copy()
    for col in ("region_1", "region_2"):
        if col in geodf.columns:
            geodf[col] = geodf[col].apply(_normalize_zone_name)
    geodf["region_key"] = _region_key(geodf, "region_1", "region_2")
    return geodf
def _geodf_to_feature_collection(df: gpd.GeoDataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"type": "FeatureCollection", "features": []}
    return json.loads(df.to_json())


def _geodf_to_feature_values(df: gpd.GeoDataFrame) -> List[Dict[str, Any]]:
    return _geodf_to_feature_collection(df).get("features", [])


def _normalize_zone_name(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if pd.notna(value):
        return str(value).strip()
    return ""


@lru_cache(maxsize=1)
def load_region_geometries(shapefile_path: str | Path = REGION_SHP_PATH) -> gpd.GeoDataFrame:
    path = Path(shapefile_path)
    if not path.exists():
        raise FileNotFoundError(f"Region shapefile not found at {path}")

    geodf = gpd.read_file(path)
    rename_map = {
        "IPM_Region": "zone",
        "region": "zone",
        "Region": "zone",
        "ZONE": "zone",
        "name": "zone",
    }
    geodf = geodf.rename(columns={k: v for k, v in rename_map.items() if k in geodf.columns})
    if "zone" not in geodf.columns:
        raise ValueError("Region shapefile must contain an 'IPM_Region' or 'zone' column")

    if geodf.crs is not None and geodf.crs.to_epsg() != 4326:
        geodf = geodf.to_crs(epsg=4326)

    geodf = geodf.loc[:, ["zone", "geometry"]].copy()
    geodf["zone"] = geodf["zone"].apply(_normalize_zone_name)
    return geodf


@lru_cache(maxsize=64)
def _region_background_cache(
    focus_regions_key: Tuple[str, ...],
    bbox_buffer: float,
) -> Tuple[List[Dict[str, Any]], Tuple[BaseGeometry, Any]]:
    region_gdf = load_region_geometries()
    if "__all__" in focus_regions_key or not focus_regions_key:
        subset = region_gdf.copy()
    else:
        subset = region_gdf.loc[region_gdf["zone"].isin(focus_regions_key)].copy()
        if subset.empty:
            subset = region_gdf.copy()

    extent_source = subset if not subset.empty else region_gdf
    extent_geom = extent_source.geometry.unary_union.convex_hull
    if bbox_buffer and bbox_buffer > 0:
        extent_geom = extent_geom.buffer(bbox_buffer)
    features = _geodf_to_feature_values(subset)
    extent_crs = extent_source.crs or "EPSG:4326"
    return features, (extent_geom, extent_crs)


def chart_tx_map(
    tx_exp: pd.DataFrame,
    scenario_field: str = "model",
    year_field: str = "planning_year",
    colormap: str = "magma",
    reverse_colors: bool = False,
    min_total_expansion: float = 1,
    result_col: str = "value",
    order: Optional[List[Any]] = None,
    year_order: Optional[List[Any]] = None,
    height: int = 220,
    width: int = 320,
    background_fill: str = "#CCCCCC",
    background_stroke: str = "white",
    bbox_buffer: float = 0.1,
    region_simplify_tol: Optional[float] = None,
    line_simplify_tol: Optional[float] = None,
    show_full_map: bool = False,
    **kwargs,
) -> Optional[alt.Chart]:
    if tx_exp.empty:
        return None

    data, scenario_field, year_field, result_col = _prepare_tx_exp_data(
        tx_exp,
        scenario_field,
        year_field,
        result_col,
        require_scenario=True,
        require_year=True,
        missing_scenario_msg="chart_tx_map requires a 'case' or 'model' column.",
        missing_year_msg="chart_tx_map requires a planning year column.",
        missing_result_msg=f"chart_tx_map could not find '{result_col}' or a fallback metric column.",
        missing_path_msg="chart_tx_map needs either explicit start/dest columns or *_to_* names.",
    )

    focus_regions = pd.unique(
        pd.concat([data["start_region"], data["dest_region"]], ignore_index=True).dropna()
    )
    focus_regions = tuple(sorted({_normalize_zone_name(region) for region in focus_regions if region}))
    if show_full_map or not focus_regions:
        focus_regions = ("__all__",)

    group_cols = [
        scenario_field,
        year_field,
        "line_name",
        "start_region",
        "dest_region",
    ]
    grouped = data.groupby(group_cols, as_index=False)[result_col].sum()
    grouped = grouped.query(f"{result_col} >= @min_total_expansion")
    if grouped.empty:
        return None

    assets = _build_tx_map_assets(
        grouped,
        result_col,
        focus_regions,
        bbox_buffer,
        region_simplify_tol,
        line_simplify_tol,
        show_full_map,
        simplify_lines_when_full_map=True,
    )
    if assets is None:
        return None
    merged, base_region_features, projection_kwargs = assets

    base_region_data = alt.Data(values=base_region_features)
    region_fill_color = background_fill or DEFAULT_REGION_COLOR

    scenario_order = order or sorted(merged[scenario_field].unique())
    column_order = year_order or sorted(merged[year_field].unique())
    if not column_order:
        return None

    color_scale = alt.Scale(
        scheme=colormap,
        reverse=reverse_colors,
        domain=[merged[result_col].min(), merged[result_col].max()],
    )

    tooltip_fields = [
        alt.Tooltip("properties.line_name:N", title="Line"),
        alt.Tooltip(f"properties.{result_col}:Q", title="Expansion (MW)", format=",.0f"),
        alt.Tooltip(f"properties.{scenario_field}:N", title=title_case(scenario_field)),
        alt.Tooltip(f"properties.{year_field}:N", title=title_case(year_field)),
    ]

    mark_kwargs = {"filled": False}
    mark_kwargs.update(kwargs)

    def _base_map_panel(title: Optional[str] = None) -> alt.Chart:
        chart = (
            alt.Chart(data=base_region_data)
            .mark_geoshape(stroke=background_stroke, fill=region_fill_color)
            .project(**projection_kwargs)
            .properties(width=width, height=height)
        )
        if title is not None:
            chart = chart.properties(title=title)
        return chart

    row_charts = []
    for scenario in scenario_order:
        scenario_subset = merged.loc[merged[scenario_field] == scenario]
        panels = []
        for year_idx, year_value in enumerate(column_order):
            year_subset = scenario_subset.loc[scenario_subset[year_field] == year_value]
            column_title = str(year_value)
            if year_subset.empty:
                panels.append(_base_map_panel(column_title))
                continue

            feature_collection = json.loads(year_subset.to_json())
            feature_values = feature_collection.get("features", [])
            if not feature_values:
                panels.append(_base_map_panel(column_title))
                continue

            data_source = alt.Data(values=feature_values)
            lines = (
                alt.Chart(data=data_source)
                .mark_geoshape(**mark_kwargs)
                .encode(
                    stroke=alt.Stroke(
                        f"properties.{result_col}:Q",
                        scale=color_scale,
                        title="Expansion (MW)",
                    ),
                    strokeWidth=alt.StrokeWidth(
                        f"properties.{result_col}:Q",
                        scale=alt.Scale(range=[2, 5]),
                        legend=None,
                    ),
                    tooltip=tooltip_fields,
                )
                .project(**projection_kwargs)
                .properties(width=width, height=height)
            )
            panel = alt.layer(_base_map_panel(column_title), lines)
            panels.append(panel)

        if panels:
            row_chart = alt.hconcat(*panels).properties(title=str(scenario))
            row_charts.append(row_chart)

    if not row_charts:
        return None

    chart = alt.vconcat(*row_charts, spacing=25).resolve_scale(color="shared", strokeWidth="shared")
    chart = chart.configure_concat(spacing=20)
    chart = chart.configure_axis(labelFontSize=12, titleFontSize=14).configure_legend(
        titleFontSize=15, labelFontSize=13
    )
    chart = configure_full_label_display(chart)
    return chart


def chart_tx_map_light(
    tx_exp: pd.DataFrame,
    scenario_field: str = "model",
    year_field: str = "planning_year",
    result_col: str = "value",
    scenario_value: Optional[Any] = None,
    year_value: Optional[Any] = None,
    min_total_expansion: float = 10,
    bbox_buffer: float = 0.1,
    width: int = 560,
    height: int = 430,
    colormap: str = "magma",
    reverse_colors: bool = False,
    line_width_range: Tuple[int, int] = (1, 6),
    background_fill: str = "#B9B9B9",
    background_stroke: str = "#FFFFFF",
    title: Optional[str] = None,
    region_simplify_tol: Optional[float] = 0.1,
    line_simplify_tol: Optional[float] = 0.04,
    show_full_map: bool = False,
) -> Optional[alt.Chart]:
    """Render a lightweight single-map view with aggressively simplified geometry."""
    if tx_exp.empty:
        return None

    data, scenario_field, year_field, result_col = _prepare_tx_exp_data(
        tx_exp,
        scenario_field,
        year_field,
        result_col,
        require_scenario=False,
        require_year=False,
        missing_scenario_msg="chart_tx_map_light requires a 'case' or 'model' column.",
        missing_year_msg="chart_tx_map_light requires a planning year column.",
        missing_result_msg=f"chart_tx_map_light could not find '{result_col}' or a fallback metric column.",
        missing_path_msg="chart_tx_map_light needs either start/dest columns or *_to_* names.",
    )

    if scenario_value is not None and scenario_field and scenario_field in data.columns:
        data = data.loc[data[scenario_field] == scenario_value]
    if year_value is not None and year_field and year_field in data.columns:
        data = data.loc[data[year_field] == year_value]
    if data.empty:
        return None

    focus_regions = pd.unique(
        pd.concat([data["start_region"], data["dest_region"]], ignore_index=True).dropna()
    )
    focus_regions = tuple(sorted({_normalize_zone_name(region) for region in focus_regions if region}))
    if show_full_map:
        focus_regions = ("__all__",)
    elif not focus_regions:
        focus_regions = ("__all__",)

    group_cols = ["line_name", "start_region", "dest_region"]
    grouped = data.groupby(group_cols, as_index=False)[result_col].sum()
    grouped = grouped.query(f"{result_col} >= @min_total_expansion")
    if grouped.empty:
        return None
    assets = _build_tx_map_assets(
        grouped,
        result_col,
        focus_regions,
        bbox_buffer,
        region_simplify_tol,
        line_simplify_tol,
        show_full_map,
        simplify_lines_when_full_map=False,
    )
    if assets is None:
        return None
    merged, base_region_features, projection_kwargs = assets

    base_region_data = alt.Data(values=base_region_features)
    feature_collection = json.loads(merged.to_json())
    feature_values = feature_collection.get("features", [])

    base_map = (
        alt.Chart(data=base_region_data)
        .mark_geoshape(fill=background_fill, stroke=background_stroke)
        .project(**projection_kwargs)
        .properties(width=width, height=height)
    )

    if not feature_values:
        chart = base_map
    else:
        data_source = alt.Data(values=feature_values)
        value_min = grouped[result_col].min()
        value_max = grouped[result_col].max()
        if value_min == value_max:
            value_min = 0
            value_max = value_max if value_max != 0 else 1
        color_scale = alt.Scale(
            scheme=colormap,
            reverse=reverse_colors,
            domain=[value_min, value_max],
        )
        lines = (
            alt.Chart(data=data_source)
            .mark_geoshape(filled=False, strokeCap="round")
            .encode(
                stroke=alt.Stroke(f"properties.{result_col}:Q", scale=color_scale, title="Expansion (MW)"),
                strokeWidth=alt.StrokeWidth(
                    f"properties.{result_col}:Q",
                    scale=alt.Scale(range=list(line_width_range)),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("properties.line_name:N", title="Line"),
                    alt.Tooltip(f"properties.{result_col}:Q", title="Expansion (MW)", format=",.0f"),
                    alt.Tooltip("properties.start_region:N", title="From"),
                    alt.Tooltip("properties.dest_region:N", title="To"),
                ],
            )
            .project(**projection_kwargs)
            .properties(width=width, height=height)
        )
        chart = alt.layer(base_map, lines)

    if title is not None:
        chart = chart.properties(title=title)

    return chart.configure_view(stroke=None)


def chart_tx_line_trends(
    tx_exp: pd.DataFrame,
    scenario_field: str = "model",
    year_field: str = "planning_year",
    line_field: str = "transmission_path_name",
    result_col: str = "New_Trans_Capacity",
    min_total_expansion: float = 0.0,
    scenario_order: Optional[List[Any]] = None,
    height: int = 280,
    width: int = 360,
) -> Optional[alt.Chart]:
    """Plot per-line transmission expansion trends per scenario across planning years."""
    if tx_exp.empty:
        return None

    data = tx_exp.copy()

    if scenario_field not in data.columns:
        if "model" in data.columns:
            scenario_field = "model"
        elif "case" in data.columns:
            scenario_field = "case"
        else:
            raise KeyError("chart_tx_line_trends requires a scenario column such as 'model'.")

    if year_field not in data.columns:
        if "planning_year" in data.columns:
            year_field = "planning_year"
        else:
            raise KeyError("chart_tx_line_trends requires a planning year column.")

    candidate_line_fields = [line_field, "transmission_path_name", "line_name"]
    line_field = next((field for field in candidate_line_fields if field and field in data.columns), None)
    if line_field is None:
        if {"start_region", "dest_region"}.issubset(data.columns):
            line_field = "line_name"
            data[line_field] = (
                data["start_region"].apply(_normalize_zone_name)
                + "_to_"
                + data["dest_region"].apply(_normalize_zone_name)
            )
        else:
            raise KeyError(
                "chart_tx_line_trends needs a line identifier column such as 'transmission_path_name'."
            )

    if result_col not in data.columns:
        fallback_cols = [col for col in ("New_Trans_Capacity", "value") if col in data.columns]
        if not fallback_cols:
            raise KeyError(
                f"chart_tx_line_trends could not find '{result_col}' or a fallback capacity column."
            )
        result_col = fallback_cols[0]

    data[result_col] = pd.to_numeric(data[result_col], errors="coerce").fillna(0)

    group_cols = [scenario_field, year_field, line_field]
    summary = data.groupby(group_cols, as_index=False)[result_col].sum()
    summary = summary.query(f"{result_col} >= @min_total_expansion")
    if summary.empty:
        return None

    summary[line_field] = summary[line_field].fillna("Unknown path").astype(str)
    summary["line_label"] = summary[line_field].str.replace("_to_", " -> ", regex=False)
    summary["line_label"] = summary["line_label"].str.strip().replace("", "Unknown path")
    summary["year_label"] = summary[year_field].astype(str)
    summary["capacity_gw"] = summary[result_col] / 1000.0

    year_sort = [str(value) for value in sorted(summary[year_field].unique())]

    unique_scenarios = summary[scenario_field].unique().tolist()
    if scenario_order:
        scenario_values = [scenario for scenario in scenario_order if scenario in unique_scenarios]
        if not scenario_values:
            scenario_values = [scenario for scenario in unique_scenarios]
    else:
        if pd.api.types.is_categorical_dtype(summary[scenario_field]):
            scenario_values = [cat for cat in summary[scenario_field].cat.categories if cat in unique_scenarios]
        else:
            scenario_values = sorted(unique_scenarios, key=lambda value: str(value))

    selection = alt.selection_multi(fields=["line_label"], bind="legend", empty="all")
    has_costs = "Cost_Trans_Capacity" in summary.columns
    tooltip_fields = [
        alt.Tooltip("line_label:N", title="Path"),
        alt.Tooltip("year_label:N", title="Planning Year"),
        alt.Tooltip(f"{result_col}:Q", title="New Capacity (MW)", format=",.1f"),
        alt.Tooltip("capacity_gw:Q", title="New Capacity (GW)", format=",.3f"),
    ]
    if has_costs:
        tooltip_fields.append(
            alt.Tooltip("Cost_Trans_Capacity:Q", title="CapEx ($)", format=",.0f")
        )

    scenario_charts: List[alt.Chart] = []
    for scenario in scenario_values:
        scenario_df = summary.loc[summary[scenario_field] == scenario]
        if scenario_df.empty:
            continue

        chart = (
            alt.Chart(scenario_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("year_label:O", title="Planning Year", sort=year_sort),
                y=alt.Y(f"{result_col}:Q", title="New Capacity (MW)"),
                color=alt.Color("line_label:N", title="Transmission Path"),
                tooltip=tooltip_fields,
                opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            )
            .properties(width=width, height=height, title=str(scenario))
        )
        scenario_charts.append(chart)

    if not scenario_charts:
        return None

    combined = alt.hconcat(*scenario_charts, spacing=30).resolve_scale(y="shared")
    combined = combined.add_params(selection)
    combined = combined.configure_axis(labelFontSize=12, titleFontSize=13).configure_legend(
        titleFontSize=13,
        labelFontSize=12,
    )
    # combined = configure_full_label_display(combined)
    return combined


# def chart_tx_scenario_map(
#     tx_exp: pd.DataFrame,
#     gdf: gpd.GeoDataFrame,
#     order=list,
#     colormap="plasma",
#     result_col="value",
# ) -> alt.Chart:
#     gdf["lat"] = gdf.geometry.centroid.y
#     gdf["lon"] = gdf.geometry.centroid.x
#     tx_exp["lat1"] = tx_exp["start_region"].map(gdf.set_index("zone")["lat"])
#     tx_exp["lon1"] = tx_exp["start_region"].map(gdf.set_index("zone")["lon"])
#     tx_exp["lat2"] = tx_exp["dest_region"].map(gdf.set_index("zone")["lat"])
#     tx_exp["lon2"] = tx_exp["dest_region"].map(gdf.set_index("zone")["lon"])

#     model_figs = []
#     for model in tx_exp.model.unique():
#         scenario_figs = []
#         for scenario in order:
#             background = (
#                 alt.Chart(gdf, title=f"{model}_{scenario}")
#                 .mark_geoshape(
#                     stroke="white",
#                     fill="lightgray",
#                 )
#                 .project(type="albersUsa")
#                 .properties(height=325, width=400)
#             )
#             lines = (
#                 alt.Chart(
#                     tx_exp.query(
#                         "planning_year >= 2025 and model==@model and value > 0 and case == @scenario"
#                     )
#                 )
#                 .mark_rule()
#                 .encode(
#                     latitude="lat1",
#                     longitude="lon1",
#                     latitude2="lat2",
#                     longitude2="lon2",
#                     strokeWidth=f"sum({result_col})",
#                     color=alt.Color(f"sum({result_col}):Q")
#                     .scale(scheme="plasma")
#                     .title("Expansion (MW)"),
#                     tooltip=[
#                         alt.Tooltip("line_name"),
#                         alt.Tooltip(f"sum({result_col})", title="Expansion (MW)"),
#                     ],
#                 )
#                 .project(type="albersUsa")
#             )

#             scenario_figs.append(background + lines)

#         model_figs.append(alt.hconcat(*scenario_figs))
#     chart = alt.vconcat(*model_figs)
#     chart = chart.configure_title(fontSize=20, dy=35).configure_legend(
#         titleFontSize=20, labelFontSize=18
#     )
#     chart = configure_full_label_display(chart)
#     return chart


def chart_cap_factor_scatter( 
    cap: pd.DataFrame,
    gen: pd.DataFrame,
    dispatch: pd.DataFrame = None,
    color="model",
    col_var=None,
    row_var=None,
    frac=None, # NOTE: frac value of less than 1.0 will make the plot sample vary each time.
    name_str_replace=None,
) -> alt.Chart:
    if name_str_replace is not None:
        for k, v in name_str_replace.items():
            gen["resource_name"] = gen["resource_name"].str.replace(k, v)
            cap["resource_name"] = cap["resource_name"].str.replace(k, v)
            if dispatch is not None:
                dispatch["resource_name"] = dispatch["resource_name"].str.replace(k, v)

    for hour in [2, 4, 6, 8]:
        cap["resource_name"] = cap["resource_name"].str.replace(f"_{hour}hour", "")
        gen["resource_name"] = gen["resource_name"].str.replace(f"_{hour}hour", "")
        if dispatch is not None:
            dispatch["resource_name"] = dispatch["resource_name"].str.replace(
                f"_{hour}hour", ""
            )

    merge_by = ["tech_type", "resource_name", "planning_year", "model"]
    group_by = ["resource_name", "planning_year", "model"]
    _tooltips = [
        alt.Tooltip("name").title("Resource"),
        alt.Tooltip(color),
    ]
    if col_var is not None:
        group_by.append(col_var)
        merge_by.append(col_var)
        # _tooltips.append(alt.Tooltip(col_var))
    if row_var is not None:
        _tooltips.append(alt.Tooltip(VAR_ABBR_MAP[row_var]).title(title_case(row_var)))
        merge_by.append(row_var)
        group_by.append(row_var)
    merge_by = list(set(merge_by))
    group_by = list(set(group_by))

    _cap = (
        cap.query("unit=='MW'")
        .groupby(
            merge_by,
            # ["tech_type", "resource_name", "model", "planning_year"],
            as_index=False,
            sort=True,
        )["end_value"]
        .sum()
    )
    _cap = _cap.query("end_value >= 50")
    _gen = pd.merge(
        gen,
        _cap,
        # on=["tech_type", "resource_name", "model", "planning_year"],
        on=merge_by,
        how="left",
    )
    _gen = _gen.query("value >= 0")
    _gen.fillna({"end_value": 0}, inplace=True)
    _gen["potential_gen"] = _gen["end_value"] * 8760

    data = _gen.groupby(group_by, as_index=False, sort=True)[
        ["value", "potential_gen", "end_value"]
    ].sum()
    data["capacity_factor"] = (data["value"] / data["potential_gen"]).round(3)
    data = data.query("end_value >= 50").drop(columns=["potential_gen", "value"])

    # selection = alt.selection_point(fields=["model"], bind="legend")
    selector = alt.selection_point(fields=["id"])  # , "model", "planning_year"
    data["end_value"] = data["end_value"].astype(int)
    if frac:
        resources = data.sample(frac=frac)["resource_name"].unique()
        data = data.loc[data["resource_name"].isin(resources)]
    # Stable ID mapping: sort resource names
    name_id_map = {name: idx for idx, name in enumerate(sorted(data["resource_name"].unique()))}
    data["id"] = data["resource_name"].map(name_id_map)
    data = data.rename(
        columns={
            "planning_year": "y",
            "resource_name": "name",
            "capacity_factor": "cf",
            "end_value": "v",
        }
    )
    _tooltips.extend(
        [
            alt.Tooltip("cf", title="Capacity Factor"),
            alt.Tooltip("v", title="Capacity (MW)", format=",.0f"),
        ]
    )
    chart = (
        alt.Chart(data)
        .mark_point()
        .encode(
            x=alt.X("v").title("Capacity (MW)").scale(type="log"),
            y=alt.Y("cf").title("Capacity Factor"),
            color=color,
            shape=color,
            tooltip=_tooltips,
            opacity=alt.condition(selector, alt.value(1), alt.value(0.2)),
        )
        .add_params(selector)
        .properties(width=300, height=250)
        .interactive()
        # .transform_filter(selector)
    )
    if col_var is not None:
        if col_var == "planning_year":
            chart = chart.encode(column=alt.Column("y").title("Planning Year"))
        else:
            chart = chart.encode(
                column=alt.Column(VAR_ABBR_MAP[col_var])
                .title(title_case(col_var))
                .header(titleFontSize=20, labelFontSize=15)
            )
    if row_var is not None:
        chart = chart.encode(
            row=alt.Row(VAR_ABBR_MAP[row_var])
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    if dispatch is not None:
        hours = list(range(120))[::2]
        _dispatch = dispatch.query("hour.isin(@hours)")
        _dispatch = _dispatch.groupby(
            ["model", "planning_year", "resource_name", "hour"], as_index=False
        )["value"].sum()
        # _dispatch = _dispatch.query("value > 5")
        _dispatch = _dispatch.loc[
            _dispatch["resource_name"].isin(data["name"].unique())
        ]
        _dispatch["value"] = _dispatch["value"].astype(int)
        _dispatch["id"] = _dispatch["resource_name"].map(name_id_map)
        _dispatch = _dispatch.drop(columns=["resource_name"])
        _dispatch = _dispatch.rename(
            columns={"planning_year": "y", "hour": "h", "value": "v"}
        )
        timeseries = (
            alt.Chart(_dispatch)
            .mark_line()
            .encode(
                x=alt.X("h").title("Hour"),
                y=alt.Y("v:Q", impute=alt.ImputeParams(value=None)).title(
                    "Dispatch (MW)"
                ),
                color=alt.Color(color),
                # opacity=alt.condition(selector, alt.value(1), alt.value(0)),
                # tooltip=["resource_name"],
            )
            # .add_params(selection, selector)
            .transform_filter(selector)
            .interactive()
        )
        if col_var is not None:

            if col_var == "planning_year":
                timeseries = timeseries.encode(
                    column=alt.Column("y").title("Planning Year")
                )
            else:
                timeseries = timeseries.encode(
                    column=alt.Column(VAR_ABBR_MAP[col_var])
                    .title(title_case(col_var))
                    .header(titleFontSize=20, labelFontSize=15)
                )
        if row_var is not None:
            timeseries = timeseries.encode(
                row=alt.Row(VAR_ABBR_MAP[row_var])
                .title(title_case(row_var))
                .header(titleFontSize=20, labelFontSize=15)
            )

        chart = alt.vconcat(chart, timeseries)

    # Order data consistently for the small bar chart, too
    data = data.sort_values(["model", "y", "name"]) if {"model", "y", "name"}.issubset(data.columns) else data
    cap_factor = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x="mean(cf)",
            y=alt.Y("model"),
            column=alt.Column("y").title("Planning Year"),
            tooltip=["name", "cf", "v"],
            color="sum(v)",
        )
        .transform_filter(selector)
    )
    chart = alt.vconcat(chart, cap_factor)

    chart = configure_full_label_display(chart)
    return chart  # | timeseries

def chart_cap_factor_scatter_genx( 
    cap: pd.DataFrame,
    gen: pd.DataFrame,
    dispatch: pd.DataFrame = None,
    color="model",
    col_var=None,
    row_var=None,
    frac=None, # NOTE: frac value of less than 1.0 will make the plot sample vary each time.
    name_str_replace=None,
) -> alt.Chart:
    if name_str_replace is not None:
        for k, v in name_str_replace.items():
            gen["resource_name"] = gen["resource_name"].str.replace(k, v)
            cap["resource_name"] = cap["resource_name"].str.replace(k, v)
            if dispatch is not None:
                dispatch["resource_name"] = dispatch["resource_name"].str.replace(k, v)

    for hour in [2, 4, 6, 8]:
        cap["resource_name"] = cap["resource_name"].str.replace(f"_{hour}hour", "")
        gen["resource_name"] = gen["resource_name"].str.replace(f"_{hour}hour", "")
        if dispatch is not None:
            dispatch["resource_name"] = dispatch["resource_name"].str.replace(
                f"_{hour}hour", ""
            )

    merge_by = ["tech_type", "resource_name", "planning_year", "model"]
    group_by = ["resource_name", "planning_year", "model"]
    _tooltips = [
        alt.Tooltip("name").title("Resource"),
        alt.Tooltip(color),
    ]
    if col_var is not None:
        group_by.append(col_var)
        merge_by.append(col_var)
        # _tooltips.append(alt.Tooltip(col_var))
    if row_var is not None:
        _tooltips.append(alt.Tooltip(VAR_ABBR_MAP[row_var]).title(title_case(row_var)))
        merge_by.append(row_var)
        group_by.append(row_var)
    merge_by = list(set(merge_by))
    group_by = list(set(group_by))
    _cap = (
        cap.query("unit=='MW'")
        .groupby(
            merge_by+['capacity_factor'],
            # ["tech_type", "resource_name", "model", "planning_year"],
            as_index=False,
            sort=True,
        )["end_value"]
        .sum()
    )
    
    _cap = _cap.query("end_value >= 50")
    _gen = pd.merge(
        gen,
        _cap,
        # on=["tech_type", "resource_name", "model", "planning_year"],
        on=merge_by,
        how="left",
    )
    _gen = _gen.query("value >= 0")
    _gen.fillna({"end_value": 0, "capacity_factor": 0.0}, inplace=True)
    _gen["potential_gen"] = _gen["end_value"] * 8760
    data = _gen.groupby(group_by+['capacity_factor'], as_index=False, sort=True)[
        ["value", "potential_gen", "end_value"]
    ].sum()
    data = data.query("end_value >= 50").drop(columns=["potential_gen", "value"])
    # selection = alt.selection_point(fields=["model"], bind="legend")
    selector = alt.selection_point(fields=["id"])  # , "model", "planning_year"
    data["end_value"] = data["end_value"].astype(int)
    if frac:
        resources = data.sample(frac=frac)["resource_name"].unique()
        data = data.loc[data["resource_name"].isin(resources)]
    # Stable ID mapping: sort resource names
    name_id_map = {name: idx for idx, name in enumerate(sorted(data["resource_name"].unique()))}
    data["id"] = data["resource_name"].map(name_id_map)
    data = data.rename(
        columns={
            "planning_year": "y",
            "resource_name": "name",
            "capacity_factor": "cf",
            "end_value": "v",
        }
    )
    _tooltips.extend(
        [
            alt.Tooltip("cf", title="Capacity Factor"),
            alt.Tooltip("v", title="Capacity (MW)", format=",.0f"),
        ]
    )
    chart = (
        alt.Chart(data)
        .mark_point()
        .encode(
            x=alt.X("v").title("Capacity (MW)").scale(type="log"),
            y=alt.Y("cf").title("Capacity Factor"),
            color=color,
            shape=color,
            tooltip=_tooltips,
            opacity=alt.condition(selector, alt.value(1), alt.value(0.2)),
        )
        .add_params(selector)
        .properties(width=300, height=250)
        .interactive()
        # .transform_filter(selector)
    )
    if col_var is not None:
        if col_var == "planning_year":
            chart = chart.encode(column=alt.Column("y").title("Planning Year"))
        else:
            chart = chart.encode(
                column=alt.Column(VAR_ABBR_MAP[col_var])
                .title(title_case(col_var))
                .header(titleFontSize=20, labelFontSize=15)
            )
    if row_var is not None:
        chart = chart.encode(
            row=alt.Row(VAR_ABBR_MAP[row_var])
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    if dispatch is not None:
        hours = list(range(120))[::2]
        _dispatch = dispatch.query("hour.isin(@hours)")
        _dispatch = _dispatch.groupby(
            ["model", "planning_year", "resource_name", "hour"], as_index=False
        )["value"].sum()
        # _dispatch = _dispatch.query("value > 5")
        _dispatch = _dispatch.loc[
            _dispatch["resource_name"].isin(data["name"].unique())
        ]
        _dispatch["value"] = _dispatch["value"].astype(int)
        _dispatch["id"] = _dispatch["resource_name"].map(name_id_map)
        _dispatch = _dispatch.drop(columns=["resource_name"])
        _dispatch = _dispatch.rename(
            columns={"planning_year": "y", "hour": "h", "value": "v"}
        )
        timeseries = (
            alt.Chart(_dispatch)
            .mark_line()
            .encode(
                x=alt.X("h").title("Hour"),
                y=alt.Y("v:Q", impute=alt.ImputeParams(value=None)).title(
                    "Dispatch (MW)"
                ),
                color=alt.Color(color),
                # opacity=alt.condition(selector, alt.value(1), alt.value(0)),
                # tooltip=["resource_name"],
            )
            # .add_params(selection, selector)
            .transform_filter(selector)
            .interactive()
        )
        if col_var is not None:

            if col_var == "planning_year":
                timeseries = timeseries.encode(
                    column=alt.Column("y").title("Planning Year")
                )
            else:
                timeseries = timeseries.encode(
                    column=alt.Column(VAR_ABBR_MAP[col_var])
                    .title(title_case(col_var))
                    .header(titleFontSize=20, labelFontSize=15)
                )
        if row_var is not None:
            timeseries = timeseries.encode(
                row=alt.Row(VAR_ABBR_MAP[row_var])
                .title(title_case(row_var))
                .header(titleFontSize=20, labelFontSize=15)
            )

        chart = alt.vconcat(chart, timeseries)

    # # Order data consistently for the small bar chart, too
    # data = data.sort_values(["model", "y", "name"]) if {"model", "y", "name"}.issubset(data.columns) else data
    # cap_factor = (
    #     alt.Chart(data)
    #     .mark_bar()
    #     .encode(
    #         x="mean(cf)",
    #         y=alt.Y("model"),
    #         column=alt.Column("y").title("Planning Year"),
    #         tooltip=["name", "cf", "v"],
    #         color="sum(v)",
    #     )
    #     .transform_filter(selector)
    # )
    # chart = alt.vconcat(chart, cap_factor)

    chart = configure_full_label_display(chart)
    return chart  # | timeseries

def chart_cost_mwh(
    op_costs: pd.DataFrame,
    x_var="model",
    col_var=None,
    row_var=None,
    order=None,
    width=alt.Step(40),
    height=200,
) -> alt.Chart:
    if op_costs.empty:
        return None

    if (Path.cwd() / "annual_demand.csv").exists():
        demand = pd.read_csv(Path.cwd() / "annual_demand.csv")
        # demand.loc[:, "agg_zone"] = demand.loc[:, "zone"].map(rev_region_map)
        op_group = ["planning_year", "model"]
        if "case" in op_costs.columns:
            op_group.append("case")
        data = pd.merge(
            op_costs.groupby(op_group, as_index=False)["Total"].sum(),
            demand.groupby(["planning_year"], as_index=False)["annual_demand"].sum(),
            on=["planning_year"],
        )
        data["cost_mwh"] = data["Total"] / data["annual_demand"]
    else:
        demand = None
    data = data.rename(columns=VAR_ABBR_MAP)
    base = (
        alt.Chart()
        .mark_bar()
        .encode(
            x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var)),
            y=alt.Y("cost_mwh").title("$/MWh"),
            # column="planning_year"
        )
    )

    text = (
        alt.Chart()
        .mark_text(dy=-5, fontSize=12)
        .encode(
            x=alt.X(VAR_ABBR_MAP[x_var]).sort(order).title(title_case(x_var)),
            y=alt.Y("cost_mwh").title("$/MWh"),
            text=alt.Text("cost_mwh", format=".1f"),
        )
    )

    chart = alt.layer(
        base,
        text,
        data=data,
    ).properties(width=width, height=height)

    chart = config_chart_row_col(chart, row_var, col_var, x_var)
    return chart


# def agg_region_map():
#     gdf.loc[:, "agg_zone"] = gdf.loc[:, "zone"].map(rev_region_map)
#     background = (
#         alt.Chart(gdf)
#         .mark_geoshape(
#             stroke="lightgray",
#             # fill="lightgray",
#         )
#         .encode(color=alt.Color("agg_zone").title("Regions"))
#         .project(type="albersUsa")
#         .configure_legend(titleFontSize=28, labelFontSize=24)
#         .properties(width=900, height=700)
#     )
