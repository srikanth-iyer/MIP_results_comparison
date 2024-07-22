import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import altair as alt
import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# alt.data_transformers.enable("vegafusion")

try:
    pd.options.mode.copy_on_write = True
except:
    pass

region_map = {
    "WECC": ["BASN", "CANO", "CASO", "NWPP", "SRSG", "RMRG"],
    "TRE": ["TRE", "TRE_WEST"],
    "SPP": ["SPPC", "SPPN", "SPPS"],
    "MISO": ["MISC", "MISE", "MISS", "MISW", "SRCE"],
    "PJM": ["PJMC", "PJMW", "PJME", "PJMD"],
    "SOU": ["SRSE", "SRCA", "FRCC"],
    "NE": ["ISNE", "NYUP", "NYCW"],
}

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
    "nuclear_nuclear": "Nuclear",
    "nuclear_1": "Nuclear",
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
for k, v in sort_nested_dict(TECH_MAP).items():
    if k in EXISTING_TECH_MAP.keys():
        _TECH_MAP[k] = (v, True)
    else:
        _TECH_MAP[k] = (v, False)
_TECH_MAP = sort_nested_dict(_TECH_MAP)


def tech_to_type(df: pd.DataFrame) -> pd.DataFrame:
    # Create dictionaries to map unique resource names to their tech types and existence
    tech_type_map = {}
    existence_map = {}

    # Get unique resource names
    unique_resource_names = sorted(
        df["resource_name"].unique().tolist(), key=lambda x: len(str(x[0]))
    )

    # Iterate over each unique resource name and determine its type and existence
    for resource_name in unique_resource_names:
        for tech, (tech_type, existing) in _TECH_MAP.items():
            if tech.lower() in resource_name.lower():
                tech_type_map[resource_name] = tech_type
                existence_map[resource_name] = existing
                break
        else:  # If no tech match, set to default
            tech_type_map[resource_name] = "Not Specified"
            existence_map[resource_name] = False

    # Special case for 'unserved_load'
    if "unserved_load" in tech_type_map:
        tech_type_map["unserved_load"] = "Other"

    # Map the tech_type and existence to the DataFrame using vectorized operations
    df["tech_type"] = df["resource_name"].map(tech_type_map)
    df["existing"] = df["resource_name"].map(existence_map)
    df["new_build"] = ~df["existing"]

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
    "emissions": ["model", "zone", "agg_zone", "planning_year", "unit", "value"],
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


def load_data(data_path: Path, fn: str) -> pd.DataFrame:
    df_list = []
    fn = f"{fn.split('.')[0]}.*"
    for f in data_path.rglob(fn):
        if not ("output" in f.parts[-2] or "Results" in f.parts[-2]):
            # print(f.parts[-2])
            _df = pd.read_csv(f)  # , engine="pyarrow")
            # , skip_blank_lines=True, engine="pyarrow" na_filter=False, )
            _df = _df.dropna(how="all")
            df_list.append(_df)
    if not df_list:
        return pd.DataFrame(columns=DATA_COLS[fn.split(".")[0]])
    df = pd.concat(df_list, ignore_index=True)
    if "resource_name" in df.columns:
        df = tech_to_type(df)
        df = df.query("~tech_type.str.contains('Other')")
    if "line_name" in df.columns:
        df = fix_tx_line_names(df)
    if "zone" in df.columns:
        df.loc[:, "agg_zone"] = df.loc[:, "zone"].map(rev_region_map)
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
    group_cols = ["planning_year", "model", "agg_zone", "zone", "tech_type"]
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


def load_genx_operations_data(
    data_path: Path,
    fn: str,
    period_dict={
        "p1": 2027,
        "p2": 2030,
        "p3": 2035,
        "p4": 2040,
        "p5": 2045,
        "p6": 2050,
    },
    hourly_data: bool = False,
) -> pd.DataFrame:
    df_list = []
    nrows = None
    if hourly_data:
        nrows = 5
    files = list(data_path.rglob(fn))
    periods = find_periods(files)
    # if any("p6" in p for p in periods):
    #     period_dict = (
    #         {"p1": 2027, "p2": 2030, "p3": 2035, "p4": 2040, "p5": 2045, "p6": 2050},
    #     )
    if not files:
        return pd.DataFrame()
    df_list = Parallel(n_jobs=1)(
        delayed(_load_op_data)(f, hourly_data, nrows, period_dict) for f in files
    )
    # model_part = -3
    # _df = pd.read_csv(f, nrows=nrows)  # , dtype_backend="pyarrow")
    # if hourly_data:
    #     if fn == "nse.csv":
    #         _df = total_from_nse_hourly_data(_df)
    #     elif "Resource" in _df.columns:
    #         _df = total_from_resource_op_hourly_data(_df)
    #     else:
    #         raise ValueError(f"There is no hourly data function for file {fn}")
    # if "Results_p" in str(f):
    #     period = period_dict[f.parent.stem.split("_")[-1]]
    #     _df.loc[:, "planning_year"] = period
    #     model_part = -4
    # elif "Inputs_p" in str(f):
    #     period = period_dict[f.parents[1].stem.split("_")[-1]]
    #     _df.loc[:, "planning_year"] = period
    #     model_part = -5
    # model = f.parts[model_part].split("_")[0]
    # _df.loc[:, "model"] = model
    # df_list.append(_df)
    if not df_list:
        return pd.DataFrame(columns=DATA_COLS.get(fn.split(".")[0], []))
    df = pd.concat(df_list, ignore_index=True)
    if fn == "costs.csv":
        try:
            df = add_genx_op_network_cost(df, data_path).pipe(calc_op_percent_total)
            df = append_npv_cost(df)
        except FileNotFoundError:
            pass
    if "Resource" in df.columns:
        df = df.rename(columns={"Resource": "resource_name"}).pipe(tech_to_type)
        try:
            df.loc[:, "zone"] = df["resource_name"].str.split("_").str[0]
        except:
            df.loc[:, "zone"] = df["resource_name"].str.split("_").list[0]
        df.loc[df["resource_name"].str.contains("TRE_WEST"), "zone"] = "TRE_WEST"
    return df


@lru_cache
def load_op_vom(period: str, load_base: bool = True) -> pd.DataFrame:
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
    vom = pd.read_csv(path, usecols=["Resource", "Var_OM_Cost_per_MWh"])
    return vom


def _load_op_data(
    f: Path,
    hourly_data: bool,
    nrows=None,
    period_dict={
        "p1": 2027,
        "p2": 2030,
        "p3": 2035,
        "p4": 2040,
        "p5": 2045,
        "p6": 2050,
    },
) -> pd.DataFrame:
    fn = f.name
    model_part = -3
    _df = pd.read_csv(f, nrows=nrows)  # , dtype_backend="pyarrow")
    if hourly_data:
        if fn == "nse.csv":
            _df = total_from_nse_hourly_data(_df)
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
    model = f.parts[model_part].split("_")[0]
    _df.loc[:, "model"] = model
    if fn == "costs.csv":
        annual_gen = pd.read_csv(f.parent / "capacityfactor.csv")
        base_vom = load_op_vom(period_str, load_base=True)
        current_vom = load_op_vom(period_str, load_base=False)
        annual_gen_base = pd.merge(annual_gen, base_vom, on="Resource")
        annual_gen_base["var_om"] = (
            annual_gen_base["AnnualSum"] * annual_gen_base["Var_OM_Cost_per_MWh"]
        )
        total_vom_base = annual_gen_base["var_om"].sum()
        annual_gen_current = pd.merge(annual_gen, current_vom, on="Resource")
        annual_gen_current["var_om"] = (
            annual_gen_current["AnnualSum"] * annual_gen_current["Var_OM_Cost_per_MWh"]
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
        _df = pd.concat([_df, s])
        if "current" in str(f) or (_df["Total"] < 0).any():
            _df.loc[_df["Costs"] == "cVar", "Total"] -= total_vom_current
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
    data = pd.DataFrame(
        {
            "zone": df.iloc[0, 1:-1],
            "value": df.iloc[1, 1:-1],
        }
    ).reset_index(drop=True)
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
        "p1": 2027,
        "p2": 2030,
        "p3": 2035,
        "p4": 2040,
        "p5": 2045,
        "p6": 2050,
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


def reverse_line_name(s: str) -> str:
    segments = s.split("_to_")
    return segments[-1] + "_to_" + segments[0]


def fix_tx_line_names(df: pd.DataFrame) -> pd.DataFrame:
    line_count = df.groupby("line_name", as_index=False)["model"].count()
    median_count = line_count["model"].median()
    reversed_lines = line_count.query("model < @median_count")

    for idx, row in reversed_lines.iterrows():
        df.loc[:, "line_name"] = df["line_name"].str.replace(
            row["line_name"], reverse_line_name(row["line_name"])
        )

    return df


def title_case(s: str) -> str:
    return s.replace("_", " ").title()


def chart_total_cap(
    cap: pd.DataFrame,
    x_var="model",
    col_var=None,
    row_var="planning_year",
    order=None,
    width=350,
    height=250,
) -> alt.Chart:
    group_by = ["tech_type", x_var]
    _tooltips = [
        alt.Tooltip("tech_type", title="Technology"),
        alt.Tooltip("end_value", title="Capacity (GW)", format=",.0f"),
        alt.Tooltip(x_var),
    ]
    if col_var is not None:
        group_by.append(col_var)
        _tooltips.append(alt.Tooltip(col_var))
    if row_var is not None:
        group_by.append(row_var)
        _tooltips.append(alt.Tooltip(row_var))
    # if "new_build" in cap.columns:
    #     group_by.append("new_build")
    #     _tooltips.append(alt.Tooltip("new_build").title("New Build"))
    group_by = [c for c in group_by if c in cap.columns]
    cap_data = cap.groupby(group_by, as_index=False)["end_value"].sum()
    cap_data["end_value"] /= 1000
    chart = (
        alt.Chart(cap_data)
        .mark_bar()
        .encode(
            x=alt.X(x_var).sort(order).title(title_case(x_var)),
            y=alt.Y("sum(end_value)").title("Capacity (GW)"),
            color=alt.Color("tech_type")
            .scale(scheme="tableau20")
            .title(title_case("tech_type")),
            # column="zone",
            # row=alt.Row(row_var)
            # .title(title_case(row_var))
            # .header(labelFontSize=15, titleFontSize=20),
            tooltip=_tooltips,
        )
        .properties(width=width, height=height)
    )
    # if "new_build" in cap_data.columns:
    #     chart = chart.encode(
    #         opacity=alt.Opacity(
    #             "new_build",
    #             sort="descending",
    #             scale=alt.Scale(domain=[False, True], range=[0.6, 1]),
    #         ).title("New Build")
    #     )
    if col_var is not None:
        chart = chart.encode(
            column=alt.Column(col_var)
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    if row_var is not None:
        chart = chart.encode(
            row=alt.Row(row_var)
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
        titleFontSize=20, labelFontSize=16
    )
    return chart


def chart_regional_cap(
    cap: pd.DataFrame,
    group_by=["agg_zone", "tech_type", "model", "planning_year"],
    x_var="model",
    row_var="planning_year",
    order=None,
) -> alt.Chart:
    data = cap.groupby(group_by, as_index=False)["end_value"].sum()
    data["end_value"] /= 1000
    data = data.rename(columns={"agg_zone": "Region"})
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(x_var).sort(order).title(title_case(x_var)),
            y=alt.Y("end_value").title("Capacity (GW)"),
            color=alt.Color("tech_type")
            .scale(scheme="tableau20")
            .title(title_case("tech_type")),
            column=alt.Column("Region").header(labelFontSize=15, titleFontSize=20),
            row=alt.Row(row_var)
            .title(title_case(row_var))
            .header(labelFontSize=15, titleFontSize=20),
            tooltip=[
                alt.Tooltip("tech_type", title="Technology"),
                alt.Tooltip("end_value", title="Capacity (GW)", format=",.0f"),
                alt.Tooltip("Region"),
                alt.Tooltip(row_var, title=title_case(row_var)),
                alt.Tooltip(x_var, title=title_case(x_var)),
            ],
        )
        .properties(width=150, height=250)
    )
    chart = (
        chart.configure_axis(labelFontSize=15, titleFontSize=15)
        .configure_axis(labelFontSize=15, titleFontSize=15)
        .configure_legend(titleFontSize=20, labelFontSize=16)
    )
    return chart


def chart_total_gen(
    gen: pd.DataFrame,
    cap: pd.DataFrame = None,
    x_var="model",
    col_var=None,
    row_var="planning_year",
    order=None,
    width=350,
    height=250,
) -> alt.Chart:
    if gen.empty:
        return None
    merge_by = ["tech_type", "resource_name", x_var, "planning_year"]
    group_by = ["tech_type", x_var, "planning_year"]
    _tooltips = [
        alt.Tooltip("tech_type", title="Technology"),
        alt.Tooltip("value", title="Generation (TWh)", format=",.0f"),
    ]
    # if "new_build" in gen.columns:
    #     merge_by.append("new_build")
    #     group_by.append("new_build")
    #     _tooltips.append(alt.Tooltip("new_build", title="New Build"))
    if col_var is not None:
        group_by.append(col_var)
        merge_by.append(col_var)
        _tooltips.append(alt.Tooltip(col_var))
    if row_var is not None:
        _tooltips.append(alt.Tooltip(row_var))
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
                alt.Tooltip("end_value", title="Capacity (MW)", format=",.0f"),
            ]
        )

    else:
        data = gen.groupby(group_by, as_index=False)["value"].sum()

    if (Path.cwd() / "annual_demand.csv").exists():
        demand = pd.read_csv(Path.cwd() / "annual_demand.csv")
        demand.loc[:, "agg_zone"] = demand.loc[:, "zone"].map(rev_region_map)
        data = pd.merge(
            data,
            demand.groupby(["planning_year"], as_index=False)["annual_demand"].sum(),
            on=["planning_year"],
        )
        data["annual_demand"] /= 1000000
    else:
        demand = None
    data["value"] /= 1000000
    chart = (
        alt.Chart()  # data)
        .mark_bar()
        .encode(
            x=alt.X(x_var).sort(order).title(title_case(x_var)),
            y=alt.Y("value").title("Generation (TWh)"),
            color=alt.Color("tech_type")
            .scale(scheme="tableau20")
            .title(title_case("tech_type")),
            # column="zone",
            # row="planning_year:O",
            tooltip=_tooltips,
        )
        .properties(width=width, height=height)
    )
    # if "new_build" in data.columns:
    #     chart = chart.encode(
    #         opacity=alt.Opacity(
    #             "new_build",
    #             sort="descending",
    #             scale=alt.Scale(domain=[False, True], range=[0.6, 1]),
    #         ).title("New Build")
    #     )
    if demand is not None:
        line = (
            alt.Chart()
            .mark_rule()
            .encode(
                y=alt.Y("annual_demand"),
            )  # column="agg_zone", row="planning_year")
            # .properties(width=150, height=250)
        )
        chart = alt.layer(chart, line, data=data)  # .facet(
        #     column="agg_zone", row="planning_year"
        # )
    if col_var is not None and row_var is not None:
        chart = chart.facet(
            column=alt.Column(col_var)
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15),
            row=alt.Row(row_var)
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15),
        )
    elif row_var is not None:
        chart = chart.facet(
            row=alt.Row(row_var)
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    elif col_var is not None:
        chart = chart.facet(
            column=alt.Column(col_var)
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15)
        )

    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
        titleFontSize=20, labelFontSize=16
    )
    return chart


def chart_regional_gen(gen: pd.DataFrame, cap: pd.DataFrame = None) -> alt.Chart:
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
            alt.Tooltip("tech_type", title="Technology"),
            alt.Tooltip("value", title="Generation (TWh)", format=",.0f"),
            alt.Tooltip("capacity_factor", title="Capacity Factor"),
        ]
    else:
        data = gen.groupby(
            ["agg_zone", "tech_type", "model", "planning_year"], as_index=False
        )["value"].sum()
        _tooltips = [
            alt.Tooltip("tech_type", title="Technology"),
            alt.Tooltip("value", title="Generation (MWh)", format=",.0f"),
        ]
    if (Path.cwd() / "annual_demand.csv").exists():
        demand = pd.read_csv(Path.cwd() / "annual_demand.csv")
        demand.loc[:, "agg_zone"] = demand.loc[:, "zone"].map(rev_region_map)
        data = pd.merge(
            data,
            demand.groupby(["agg_zone", "planning_year"], as_index=False)[
                "annual_demand"
            ].sum(),
            on=["agg_zone", "planning_year"],
        )
        data["annual_demand"] /= 1000000
    else:
        demand = None
    data["value"] /= 1000000
    chart = (
        alt.Chart()
        .mark_bar()
        .encode(
            x=alt.X("model").title("Model"),
            y=alt.Y("value").title("Generation (TWh)"),
            color=alt.Color("tech_type")
            .scale(scheme="tableau20")
            .title(title_case("tech_type")),
            # column="agg_zone",
            # row="planning_year:O",
            tooltip=_tooltips,
        )
        .properties(width=150, height=250)
    )

    if demand is not None:
        line = (
            alt.Chart()
            .mark_rule()
            .encode(
                y=alt.Y("annual_demand"),
            )  # column="agg_zone", row="planning_year")
            .properties(width=150, height=250)
        )
        chart = alt.layer(chart, line, data=data).facet(
            column=alt.Column("agg_zone")
            .title("Region")
            .header(titleFontSize=20, labelFontSize=15),
            row=alt.Row("planning_year")
            .title(title_case("planning_year"))
            .header(titleFontSize=20, labelFontSize=15),
        )
    # chart = chart.encode(column="agg_zone:N", row="planning_year:O")
    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
        titleFontSize=20, labelFontSize=16
    )
    return chart


def chart_tx_expansion(
    data: pd.DataFrame,
    x_var="model",
    facet_col="line_name",
    n_cols=10,
    col_var=None,
    row_var=None,
    order=None,
    height=200,
    width=alt.Step(20),
) -> alt.Chart:
    if data.empty:
        return None
    _tooltip = [
        alt.Tooltip("sum(value)", format=",.0f", title="Period GW"),
        alt.Tooltip("planning_year", title=title_case("planning_year")),
    ]
    data["line_name"] = data["line_name"].str.replace("_to_", " | ")
    if order is None:
        order = sorted(data[x_var].unique())
    if col_var is None or row_var is None:
        first_year = data["planning_year"].min()
        idx_cols = [c for c in [x_var, facet_col, col_var, row_var] if c is not None]
        data = data.set_index(idx_cols)
        first_data = data.query("planning_year == @first_year")
        df_list = []
        for year in data["planning_year"].unique():
            _df = data.query("planning_year == @year")
            if year == first_year:
                _df["line_growth"] = 0
                # df_list.append(_df)
            else:
                try:
                    _df["line_growth"] = (_df["value"] / first_data["value"]).fillna(0)
                except:
                    _df["line_growth"] = 0
            df_list.append(_df)
        data = pd.concat(df_list).reset_index()
        if not (col_var is None and row_var is None):
            _tooltip.append(alt.Tooltip("line_growth", format=".1%"))

    if x_var == "case":
        _tooltip.append(
            alt.Tooltip("case"),
        )
    data["value"] /= 1000
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            # xOffset="model:N",
            x=alt.X(x_var).sort(order),
            y=alt.Y("sum(value)").title("Transmission (GW)"),
            color=alt.Color("model:N").title(title_case("model")),
            opacity=alt.Opacity("planning_year:O", sort="descending").title(
                title_case("planning_year")
            ),
            # facet=alt.Facet("line_name", columns=n_cols),
            order=alt.Order(
                # Sort the segments of the bars by this field
                "planning_year",
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
            facet=alt.Facet(facet_col, columns=n_cols)
            .title(title_case(facet_col))
            .header(titleFontSize=20, labelFontSize=15)
        )
    elif col_var is None and row_var is None:
        text = (
            alt.Chart()
            .mark_text(dy=-5, fontSize=14)
            .encode(
                x=alt.X(x_var).sort(order).title(title_case(x_var)),
                y="sum(value):Q",
                text=alt.Text("sum(value):Q", format=".0f"),
            )
        )
        chart = alt.layer(chart, text, data=data).properties(width=width)
    if col_var is not None:
        chart = chart.encode(
            column=alt.Column(col_var)
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    if row_var is not None:
        chart = chart.encode(
            row=alt.Row(row_var)
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
        titleFontSize=20, labelFontSize=16
    )
    if all([i is None for i in [facet_col, row_var, col_var]]):
        chart = (
            chart.properties(height=400, width=300)
            .configure_axis(labelFontSize=16, titleFontSize=18)
            .configure_legend(titleFontSize=20, labelFontSize=16)
        )
    return chart


def chart_emissions_intensity(
    emiss,
    gen,
    x_var="Region",
    x_offset=None,
    col_var="model",
    row_var="planning_year",
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

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(x_var).title(title_case(x_var)),
            y=alt.Y("emissions_intensity").title("kg/MWh"),
        )
        .properties(height=150)
    )
    # if col_var is not None and row_var is not None:
    #     text = (
    #         alt.Chart()
    #         .mark_text(dy=-5, fontSize=14)
    #         .encode(
    #             x=alt.X(x_var).sort(order).title(title_case(x_var)),
    #             y="sum(value):Q",
    #             text=alt.Text("sum(value):Q", format=".0f"),
    #         )
    #     )
    #     chart = alt.layer(chart, text, data=data).properties(width=width)
    if col_var is not None:
        chart = chart.encode(
            column=alt.Column(col_var)
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    if row_var is not None:
        chart = chart.encode(
            row=alt.Row(row_var)
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
        titleFontSize=20, labelFontSize=16
    )

    return chart


def chart_emissions(
    emiss: pd.DataFrame, x_var="model", col_var=None, order=None, co2_limit=True
) -> alt.Chart:
    if emiss.empty:
        return None
    _tooltips = [
        alt.Tooltip("sum(value)", format=",.0f", title="Million Tonnes"),
        alt.Tooltip("Region"),
    ]
    emiss["Region"] = emiss["zone"].map(rev_region_map)
    group_by = ["Region", x_var, "planning_year"]
    if col_var is not None:
        group_by.append(col_var)

    data = emiss.groupby(group_by, as_index=False)["value"].sum()
    if col_var is not None:
        _tooltips.append(alt.Tooltip(col_var))
    if order is None:
        order = sorted(data[x_var].unique())
    data["value"] /= 1e6
    data["limit"] = 0
    data.loc[data["planning_year"] == 2027, "limit"] = 873
    data.loc[data["planning_year"] == 2030, "limit"] = 186
    data.loc[data["planning_year"] == 2035, "limit"] = 130
    data.loc[data["planning_year"] == 2040, "limit"] = 86.7
    data.loc[data["planning_year"] == 2045, "limit"] = 43.3
    base = (
        alt.Chart()
        .mark_bar()
        .encode(
            x=alt.X(x_var).sort(order).title(title_case(x_var)),
            y=alt.Y("sum(value)").title("CO2 (Million Tonnes)"),
            color=alt.Color("Region"),  # .scale(scheme="tableau20"),
            # column="agg_zone",
            # row="planning_year:O",
            tooltip=_tooltips,
        )
        # .properties(width=350, height=250)
        # .resolve_scale(y="independent")
    )
    text = (
        alt.Chart()
        .mark_text(dy=-5)
        .encode(
            x=alt.X(x_var).sort(order).title(title_case(x_var)),
            y="sum(value):Q",
            text=alt.Text("sum(value):Q", format=".0f"),
        )
    )  # .properties(width=350, height=250)
    if co2_limit:
        size = 2
    else:
        size = 0
    line = (
        alt.Chart()
        .mark_rule(size=size)
        .encode(
            y=alt.Y("limit"),
        )  # column="agg_zone", row="planning_year")
        # .properties(width=150, height=250)
    )
    if col_var is None:
        chart = (
            alt.layer(base, text, line, data=data)
            .properties(width=350, height=250)
            .facet(
                row=alt.Row("planning_year:O")
                .title(title_case("planning_year"))
                .header(titleFontSize=20, labelFontSize=15)
            )
        )
    else:
        chart = (
            alt.layer(base, text, line, data=data)
            .properties(width=350, height=250)
            .facet(
                row=alt.Row("planning_year:O")
                .title(title_case("planning_year"))
                .header(titleFontSize=20, labelFontSize=15),
                column=alt.Column(col_var)
                .title(title_case(col_var))
                .header(titleFontSize=20, labelFontSize=15),
            )
        )
    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
        titleFontSize=20, labelFontSize=16
    )
    return chart


def chart_dispatch(data: pd.DataFrame) -> alt.Chart:
    data = data.rename(
        columns={
            "model": "m",
            "tech_type": "tt",
            "agg_zone": "az",
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
            x=alt.X("h").title("Hour"),
            y=alt.Y("v").title("Dispatch (GW)"),
            color=alt.Color("m").legend(title="Model"),
            row=alt.Row("tt")
            .title("Tech Type")
            .header(titleFontSize=20, labelFontSize=15),
            column=alt.Column("az")
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
    return chart


def chart_wind_dispatch(data: pd.DataFrame) -> alt.Chart:
    data = data.rename(
        columns={
            "model": "m",
            "tech_type": "tt",
            "zone": "z",
            "hour": "h",
            "value": "v",
        }
    )
    data = data.drop(columns=["tech_type"], errors="ignore")
    data["v"] /= 1000
    selection = alt.selection_point(fields=["model"], bind="legend")
    if "cluster" in data.columns:
        chart = (
            alt.Chart(data)
            .mark_line()
            .encode(
                x=alt.X("h").title("Hour"),
                y=alt.Y("v").title("Dispatch (GW)"),
                color=alt.Color("m").legend(title="Model"),
                strokeDash="cluster",
                facet=alt.Facet("z", columns=5)
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
                facet=alt.Facet("z", columns=5)
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
    return chart


# Calculate NPV for each cost category
def calculate_npv(df, period_len, discount_rate, base_year):
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
        2027: 4,
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
    data, x_var="model", col_var=None, row_var=None, order=None
) -> alt.Chart:
    _tooltip = [alt.Tooltip("Total", format=",.0f")]
    chart_cols = ["Costs", "Total", x_var]

    if "percent_total" in data.columns:
        _tooltip.append(alt.Tooltip("percent_total:Q", format=".1%"))
        chart_cols.append("percent_total")
    if col_var is not None:
        _tooltip.append(alt.Tooltip(col_var))
        _tooltip.append(alt.Tooltip("Costs"))
        chart_cols.append(col_var)
    if row_var is not None:
        _tooltip.append(alt.Tooltip(row_var))
        chart_cols.append(row_var)

    base = (
        alt.Chart()
        .mark_bar()
        .encode(
            # xOffset="model:N",
            x=alt.X(x_var).sort(order).title(title_case(x_var)),
            y=alt.Y("Total").title("Costs (Billion $)"),
            color="Costs:N",
            tooltip=_tooltip,
        )
    )

    text = (
        alt.Chart()
        .mark_text(dy=-5, fontSize=11)
        .encode(
            x=alt.X(x_var).sort(order).title(title_case(x_var)),
            y="sum(Total):Q",
            text=alt.Text("sum(Total):Q", format=".0f"),
        )
    )

    chart = alt.layer(
        base,
        text,
        data=data[chart_cols].query("Total!=0 and Costs != 'cTotal'"),
    ).properties(width=alt.Step(40), height=250)

    if row_var is None and col_var is None:
        return chart
    elif row_var is None and col_var is not None:
        chart = chart.facet(
            column=alt.Column(col_var)
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    elif col_var is None and row_var is not None:
        chart = chart.facet(
            row=alt.Row(row_var)
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    else:
        chart = chart.facet(
            row=alt.Row(row_var)
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15),
            column=alt.Column(col_var)
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15),
        )

    return chart


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

    chart = single_op_cost_chart(data, x_var, col_var, row_var, order)
    if not npv_costs.empty:
        npv_data = npv_costs.copy()
        npv_data["Total"] /= 1e9
        npv_chart = single_op_cost_chart(npv_data, x_var, col_var, row_var, order)
        if col_var == "planning_year":
            final_chart = chart | npv_chart
        elif row_var == "planning_year":
            final_chart = chart & npv_chart
        else:
            final_chart = chart
    final_chart = (
        final_chart.configure_axis(labelFontSize=15, titleFontSize=15)
        .configure_axis(labelFontSize=15, titleFontSize=15)
        .configure_legend(titleFontSize=20, labelFontSize=16)
    )
    return final_chart


def chart_op_nse(
    op_nse: pd.DataFrame, x_var="model", col_var=None, row_var=None, order=None
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
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            # xOffset="model:N",
            x=alt.X(x_var).sort(order).title(title_case(x_var)),
            y=alt.Y("sum(value)").title("Annual non-served GWh"),
            color=alt.Color("model:N").title(title_case("model")),
            tooltip=alt.Tooltip("sum(value)", format=",.0f", title="NSE"),
        )
        .properties(width=alt.Step(40), height=250)
    )
    if col_var is not None:
        chart = chart.encode(
            column=alt.Column(col_var)
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    if row_var is not None:
        chart = chart.encode(
            row=alt.Row(row_var)
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
        titleFontSize=20, labelFontSize=16
    )
    return chart


def chart_op_emiss(
    op_emiss: pd.DataFrame,
    x_var="model",
    color="tech_type",
    col_var=None,
    row_var=None,
    order=None,
) -> alt.Chart:
    if op_emiss.empty:
        return None
    op_emiss["Region"] = op_emiss["zone"].map(rev_region_map)
    if (
        col_var is None
        and "planning_year" in op_emiss.columns
        and row_var != "planning_year"
    ):
        col_var = "planning_year"
    _tooltip = [
        alt.Tooltip("value", format=",.0f", title="Emissions"),
        alt.Tooltip(color),
    ]
    by = [color, x_var]

    color_scale = "category10"
    if op_emiss[color].nunique() > 10:
        color_scale = "tableau20"
    if col_var is not None:
        _tooltip.append(alt.Tooltip(col_var))
        by.append(col_var)
    if row_var is not None:
        _tooltip.append(alt.Tooltip(row_var))
        by.append(row_var)

    by = list(set(by))
    data = op_emiss.groupby(by, as_index=False)["value"].sum().query("value>0")
    data["value"] /= 1e6
    base = (
        alt.Chart()
        .mark_bar()
        .encode(
            # xOffset="model:N",
            x=alt.X(x_var).sort(order).title(title_case(x_var)),
            y=alt.Y("value").title("CO2 (Million Tonnes)"),
            color=alt.Color(color).scale(scheme=color_scale).title(title_case(color)),
            tooltip=_tooltip,
        )
    )

    text = (
        alt.Chart()
        .mark_text(dy=-5, fontSize=11)
        .encode(
            x=alt.X(x_var).sort(order).title(title_case(x_var)),
            y="sum(value):Q",
            text=alt.Text("sum(value):Q", format=".0f"),
        )
    )

    chart = alt.layer(
        base,
        text,
        data=data,
    ).properties(width=alt.Step(40), height=250)

    if row_var is None and col_var is None:
        return chart
    elif row_var is None and col_var is not None:
        chart = chart.facet(
            column=alt.Column(col_var)
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    elif col_var is None and row_var is not None:
        chart = chart.facet(
            row=alt.Row(row_var)
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    else:
        chart = chart.facet(
            row=alt.Row(row_var)
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15),
            column=alt.Column(col_var)
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15),
        )
    chart = (
        chart.configure_axis(labelFontSize=15, titleFontSize=15)
        .configure_axis(labelFontSize=15, titleFontSize=15)
        .configure_legend(titleFontSize=20, labelFontSize=16)
    )
    return chart


gdf = gpd.read_file("conus_26z_latlon_simple.geojson")
gdf = gdf.rename(columns={"model_region": "zone"})


def chart_tx_map(tx_exp: pd.DataFrame, gdf: gpd.GeoDataFrame) -> alt.Chart:
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["lon"] = gdf.geometry.centroid.x
    tx_exp["lat1"] = tx_exp["start_region"].map(gdf.set_index("zone")["lat"])
    tx_exp["lon1"] = tx_exp["start_region"].map(gdf.set_index("zone")["lon"])
    tx_exp["lat2"] = tx_exp["dest_region"].map(gdf.set_index("zone")["lat"])
    tx_exp["lon2"] = tx_exp["dest_region"].map(gdf.set_index("zone")["lon"])

    model_figs = []
    data = tx_exp.copy()
    # data["value"] /= 1000
    for model in tx_exp.model.unique():
        background = (
            alt.Chart(gdf, title=f"{model}")
            .mark_geoshape(
                stroke="white",
                fill="lightgray",
            )
            .project(type="albersUsa")
            .properties(height=325, width=400)
        )
        lines = (
            alt.Chart(
                data.query("planning_year >= 2025 and model==@model and value > 0")
            )
            .mark_rule()
            .encode(
                latitude="lat1",
                longitude="lon1",
                latitude2="lat2",
                longitude2="lon2",
                strokeWidth="sum(value)",
                color=alt.Color("sum(value):Q")
                .scale(scheme="plasma")
                .title("Expansion (MW)"),
                tooltip=[
                    alt.Tooltip("line_name"),
                    alt.Tooltip("sum(value)", title="Expansion (MW)"),
                ],
            )
            .project(type="albersUsa")
        )

        model_figs.append(background + lines)
    chart = alt.vconcat(
        alt.hconcat(*model_figs[:2]), alt.hconcat(*model_figs[2:])
    ).configure_concat(spacing=-50)
    chart = (
        chart.configure_axis(labelFontSize=15, titleFontSize=15)
        .configure_legend(titleFontSize=20, labelFontSize=18)
        .configure_title(fontSize=20, dy=35)
    )
    return chart


def chart_tx_scenario_map(
    tx_exp: pd.DataFrame, gdf: gpd.GeoDataFrame, order=list
) -> alt.Chart:
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["lon"] = gdf.geometry.centroid.x
    tx_exp["lat1"] = tx_exp["start_region"].map(gdf.set_index("zone")["lat"])
    tx_exp["lon1"] = tx_exp["start_region"].map(gdf.set_index("zone")["lon"])
    tx_exp["lat2"] = tx_exp["dest_region"].map(gdf.set_index("zone")["lat"])
    tx_exp["lon2"] = tx_exp["dest_region"].map(gdf.set_index("zone")["lon"])

    model_figs = []
    for model in tx_exp.model.unique():
        scenario_figs = []
        for scenario in order:
            background = (
                alt.Chart(gdf, title=f"{model}_{scenario}")
                .mark_geoshape(
                    stroke="white",
                    fill="lightgray",
                )
                .project(type="albersUsa")
                .properties(height=325, width=400)
            )
            lines = (
                alt.Chart(
                    tx_exp.query(
                        "planning_year >= 2025 and model==@model and value > 0 and case == @scenario"
                    )
                )
                .mark_rule()
                .encode(
                    latitude="lat1",
                    longitude="lon1",
                    latitude2="lat2",
                    longitude2="lon2",
                    strokeWidth="sum(value)",
                    color=alt.Color("sum(value):Q")
                    .scale(scheme="plasma")
                    .title("Expansion (MW)"),
                    tooltip=[
                        alt.Tooltip("line_name"),
                        alt.Tooltip("sum(value)", title="Expansion (MW)"),
                    ],
                )
                .project(type="albersUsa")
            )

            scenario_figs.append(background + lines)

        model_figs.append(alt.hconcat(*scenario_figs))
    chart = alt.vconcat(*model_figs)
    chart = chart.configure_title(fontSize=20, dy=35).configure_legend(
        titleFontSize=20, labelFontSize=18
    )
    return chart


def chart_cap_factor_scatter(
    cap: pd.DataFrame,
    gen: pd.DataFrame,
    dispatch: pd.DataFrame = None,
    color="model",
    col_var=None,
    row_var=None,
    frac=None,
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
        _tooltips.append(alt.Tooltip(row_var))
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

    data = _gen.groupby(group_by, as_index=False)[
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

    name_id_map = {name: idx for idx, name in enumerate(data["resource_name"].unique())}
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
                column=alt.Column(col_var)
                .title(title_case(col_var))
                .header(titleFontSize=20, labelFontSize=15)
            )
    if row_var is not None:
        chart = chart.encode(
            row=alt.Row(row_var)
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
                    column=alt.Column(col_var)
                    .title(title_case(col_var))
                    .header(titleFontSize=20, labelFontSize=15)
                )
        if row_var is not None:
            timeseries = timeseries.encode(
                row=alt.Row(row_var)
                .title(title_case(row_var))
                .header(titleFontSize=20, labelFontSize=15)
            )

        chart = alt.vconcat(chart, timeseries)

    return chart  # | timeseries


def chart_cost_mwh(
    op_costs: pd.DataFrame, x_var="model", col_var=None, row_var=None, order=None
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
    base = (
        alt.Chart()
        .mark_bar()
        .encode(
            x=alt.X(x_var).sort(order).title(title_case(x_var)),
            y=alt.Y("cost_mwh").title("$/MWh"),
            # column="planning_year"
        )
    )

    text = (
        alt.Chart()
        .mark_text(dy=-5, fontSize=12)
        .encode(
            x=alt.X(x_var).sort(order).title(title_case(x_var)),
            y=alt.Y("cost_mwh").title("$/MWh"),
            text=alt.Text("cost_mwh", format=".1f"),
        )
    )

    chart = alt.layer(
        base,
        text,
        data=data,
    ).properties(width=alt.Step(40), height=250)

    if row_var is None and col_var is None:
        return chart
    elif row_var is None and col_var is not None:
        chart = chart.facet(
            column=alt.Column(col_var)
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    elif col_var is None and row_var is not None:
        chart = chart.facet(
            row=alt.Row(row_var)
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15)
        )
    else:
        chart = chart.facet(
            row=alt.Row(row_var)
            .title(title_case(row_var))
            .header(titleFontSize=20, labelFontSize=15),
            column=alt.Column(col_var)
            .title(title_case(col_var))
            .header(titleFontSize=20, labelFontSize=15),
        )
    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
        titleFontSize=28, labelFontSize=24
    )
    return chart


def agg_region_map():
    gdf.loc[:, "agg_zone"] = gdf.loc[:, "zone"].map(rev_region_map)
    background = (
        alt.Chart(gdf)
        .mark_geoshape(
            stroke="lightgray",
            # fill="lightgray",
        )
        .encode(color=alt.Color("agg_zone").title("Regions"))
        .project(type="albersUsa")
        .configure_legend(titleFontSize=28, labelFontSize=24)
        .properties(width=900, height=700)
    )
