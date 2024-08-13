"Recalculate GenX annuities using a single financial lifetime and WACC"

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd

# from powergenome.financials import investment_cost_calculator
import typer
from typing_extensions import Annotated

ListLike = Union[list, set, pd.Series, np.array]


def investment_cost_calculator(
    capex: Union[ListLike, float],
    wacc: Union[ListLike, float],
    cap_rec_years: Union[ListLike, int],
    compound_method: str = "discrete",
) -> np.array:
    """Calculate annualized investment cost using either discrete or continuous compounding.

    Parameters
    ----------
    capex : Union[LIST_LIKE, float]
        Single or list-like capital costs for one or more resources
    wacc : Union[LIST_LIKE, float]
        Weighted average cost of capital. Can be a single value or one value for each resource.
        Should be the same length as capex or a single value.
    cap_rec_years : Union[LIST_LIKE, int]
        Capital recovery years or the financial lifetime of each asset. Should be the same
        length as capex or a single value.
    compound_method : str, optional
        The method to compound interest. Either "discrete" or "continuous", by default
        "discrete"

    Returns
    -------
    np.array
        An annual investment cost for each capital cost

    Raises
    ------
    TypeError
        A list-like type of WACC or capital recovery years was provided for only a single
        capex
    ValueError
        The capex and WACC or capital recovery years are both list-like but not the same
        length
    ValueError
        One of the inputs contains a nan value
    ValueError
        The compounding_method argument must be either "discrete" or "continuous"
    """
    # Data checks
    for var, name in zip([wacc, cap_rec_years], ["wacc", "capital recovery years"]):
        if np.isscalar(capex):
            if not np.isscalar(var):
                raise TypeError(
                    f"Multiple {name} values were provided for only a single resource capex "
                    "when calculating annualized inventment costs. Only a single value "
                    "should be provided with only a single resource capex."
                )
        else:
            if not np.isscalar(var) and len(var) != len(capex):
                raise ValueError(
                    f"The number of {name} values ({len(var)}) and the number of resource "
                    f"capex values ({len(capex)}) should be the same but they are not."
                )

    # Convert everything to arrays and do the calculations.
    vars = [capex, wacc, cap_rec_years]
    dtypes = [float, float, int]
    for idx, (var, dtype) in enumerate(zip(vars, dtypes)):
        vars[idx] = np.asarray(var, dtype=dtype)
    capex, wacc, cap_rec_years = vars
    # capex = np.asarray(capex, dtype=float)
    # wacc = np.asarray(wacc, dtype=float)
    # cap_rec_years = np.asarray(cap_rec_years, dtype=int)

    for var, name in zip(
        [capex, wacc, cap_rec_years], ["capex", "wacc", "capital recovery years"]
    ):
        if np.isnan(var).any() or pd.isnull(var).any():
            raise ValueError(f"Investment variable {name} costs contains nan values")

    if compound_method.lower() == "discrete":
        inv_cost = _discrete_inv_cost_calc(
            capex=capex, wacc=wacc, cap_rec_years=cap_rec_years
        )
    elif "cont" in compound_method.lower():
        inv_cost = _continuous_inv_cost_calc(
            capex=capex, wacc=wacc, cap_rec_years=cap_rec_years
        )
    else:
        raise ValueError(
            f"'{compound_method}' is not a valid compounding method for converting capex "
            "into annual investment costs. Valid methods are 'discrete' or 'continuous'."
        )

    return inv_cost


def _continuous_inv_cost_calc(
    capex: Union[np.array, float],
    wacc: Union[np.array, float],
    cap_rec_years: Union[np.array, int],
) -> np.array:
    """Calculate annualized investment cost using continuous compounding.

    Parameters
    ----------
    capex : Union[LIST_LIKE, float]
        Single or list-like capital costs for one or more resources
    wacc : Union[LIST_LIKE, float]
        Weighted average cost of capital. Can be a single value or one value for each resource.
        Should be the same length as capex or a single value.
    cap_rec_years : Union[LIST_LIKE, int]
        Capital recovery years or the financial lifetime of each asset. Should be the same
        length as capex or a single value.

    Returns
    -------
    np.array
        An annual investment cost for each capital cost
    """
    inv_cost = capex * (
        np.exp(wacc * cap_rec_years)
        * (np.exp(wacc) - 1)
        / (np.exp(wacc * cap_rec_years) - 1)
    )

    return inv_cost


def _discrete_inv_cost_calc(
    capex: Union[np.array, float],
    wacc: Union[np.array, float],
    cap_rec_years: Union[np.array, int],
) -> np.array:
    """Calculate annualized investment cost using discrete compounding.

    Parameters
    ----------
    capex : Union[LIST_LIKE, float]
        Single or list-like capital costs for one or more resources
    wacc : Union[LIST_LIKE, float]
        Weighted average cost of capital. Can be a single value or one value for each resource.
        Should be the same length as capex or a single value.
    cap_rec_years : Union[LIST_LIKE, int]
        Capital recovery years or the financial lifetime of each asset. Should be the same
        length as capex or a single value.

    Returns
    -------
    np.array
        An annual investment cost for each capital cost
    """
    inv_cost = capex * wacc / (1 - (1 + wacc) ** -cap_rec_years)

    return inv_cost


def load_gen_data(fn: Path) -> pd.DataFrame:
    return pd.read_csv(fn, keep_default_na=False)


def load_network_data(fn: Path) -> pd.DataFrame:
    return pd.read_csv(fn)


def calc_gen_annuities(
    df: pd.DataFrame, wacc: float = None, capex_mod: Dict[str, float] = None
) -> pd.DataFrame:
    if capex_mod:
        for tech, value in capex_mod.items():
            for col in ["capex_mw", "capex_mwh"]:
                df.loc[df["Resource"].str.contains(tech, case=False), col] *= value
    if "WACC" in df.columns:
        wacc_col = "WACC"
    elif "wacc_real" in df.columns:
        wacc_col = "wacc_real"
    else:
        raise KeyError("Could not find either 'WACC' or 'wacc_real' column in df")
    if wacc is not None:
        df[wacc_col] = wacc
    if "old_Inv_Cost_per_MWyr" not in df.columns:
        df["old_Inv_Cost_per_MWyr"] = df["Inv_Cost_per_MWyr"]
        df["old_Inv_Cost_per_MWhyr"] = df["Inv_Cost_per_MWhyr"]
    if "old_capex_mw" not in df.columns:
        df["old_capex_mw"] = df["capex_mw"].copy()
        df["old_capex_mwh"] = df["capex_mwh"].copy()
    if "Capital_Recovery_Period" in df.columns:
        crp_col = "Capital_Recovery_Period"
    elif "cap_recovery_years" in df.columns:
        crp_col = "cap_recovery_years"
    else:
        raise KeyError("No recognized capital recovery period column found")
    capex_cols = ["co2_pipeline_capex_mw", "spur_capex", "interconnect_capex_mw"]
    capex_cols = [c for c in capex_cols if c in df.columns]
    df.loc[df["New_Build"] == 1, "Inv_Cost_per_MWyr"] = (
        (
            investment_cost_calculator(
                (
                    df.loc[df["New_Build"] == 1, "capex_mw"]
                    - df.loc[df["New_Build"] == 1, "co2_pipeline_capex_mw"]
                )
                * df.loc[df["New_Build"] == 1, "regional_cost_multiplier"],
                wacc=df.loc[df["New_Build"] == 1, wacc_col],
                cap_rec_years=df.loc[df["New_Build"] == 1, crp_col],
                compound_method="discrete",
            )
            + investment_cost_calculator(
                df.loc[df["New_Build"] == 1, capex_cols].sum(axis=1),
                wacc=df.loc[df["New_Build"] == 1, wacc_col],
                cap_rec_years=df.loc[df["New_Build"] == 1, crp_col],
                compound_method="discrete",
            )
        )
        .round(0)
        .astype(int)
    )
    df.loc[df["New_Build"] == 1, "Inv_Cost_per_MWhyr"] = (
        investment_cost_calculator(
            df.loc[df["New_Build"] == 1, "capex_mwh"]
            * df.loc[df["New_Build"] == 1, "regional_cost_multiplier"],
            wacc=df.loc[df["New_Build"] == 1, wacc_col],
            cap_rec_years=df.loc[df["New_Build"] == 1, crp_col],
            compound_method="discrete",
        )
        .round(0)
        .astype(int)
    )

    return df


def calc_network_annuities(df: pd.DataFrame, wacc: float = None) -> pd.DataFrame:
    if "old_Line_Reinforcement_Cost_per_MWyr" not in df.columns:
        df["old_Line_Reinforcement_Cost_per_MWyr"] = df.loc[
            :, "Line_Reinforcement_Cost_per_MWyr"
        ]
    if wacc:
        df["WACC"] = wacc

    df.loc[:, "Line_Reinforcement_Cost_per_MWyr"] = investment_cost_calculator(
        df["Line_Reinforcement_Cost_per_MW"],
        wacc=wacc,
        cap_rec_years=df["Capital_Recovery_Period"],
        compound_method="discrete",
    )

    return df


# def main(case_id: Annotated[List[str], typer.Option()], wacc: float = None):
#     cwd = Path.cwd()
#     fns = list(cwd.rglob("Generators_data.csv"))
#     for fn in fns:
#         for c in case_id:
#             if f"{c}_{fn.parts[-4]}" in fn.parts:
#                 print(fn)
#                 df = load_gen_data(fn)
#                 df = calc_annuities(df, wacc)
#                 df.to_csv(fn, index=False)


def main(
    folder: Annotated[List[str], typer.Option()],
    gen_wacc: float = None,
    network_wacc: float = 0.05,
):
    cwd = Path.cwd()
    # fns = []
    for f in folder:

        # fns.extend(list((cwd / f).rglob("Generators_data.csv")))
        for fn in (cwd / f).rglob("Generators_data.csv"):
            capex_mod = {}
            if "current_policies" in str(fn) and not (
                "p5" in str(fn) or "p6" in str(fn)
            ):
                capex_mod = {"OffShoreWind": 0.63, "Battery": 0.63, "Nuclear": 0.63}
            # for c in case_id:
            # if f"{c}_{fn.parts[-4]}" in fn.parts:
            print(fn)
            df = load_gen_data(fn)
            df["Fuel"] = df["Fuel"].fillna("None")
            df = calc_gen_annuities(df, gen_wacc, capex_mod)
            df.to_csv(fn, index=False)

        for fn in (cwd / f).rglob("Network.csv"):
            df = load_network_data(fn)
            df = calc_network_annuities(df, network_wacc)
            df.to_csv(fn, index=False)


if __name__ == "__main__":
    typer.run(main)
