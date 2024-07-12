"Copy annuities from base to current policies"

from pathlib import Path

import pandas as pd

CWD = Path.cwd()

base_gen_data_files = list((CWD / "base_52_week_commit").rglob("Generators_data.csv"))

for f in base_gen_data_files:
    dest_path = Path(
        str(f).replace("base_52_week_commit", "current_policies_52_week_commit_no_itc")
    )

    assert dest_path.exists()

    base_df = pd.read_csv(f, na_filter=False)
    dest_df = pd.read_csv(dest_path, na_filter=False)

    for kind in ["MW", "MWh"]:
        dest_df[f"policy_Inv_Cost_per_{kind}yr"] = dest_df.loc[
            :, f"Inv_Cost_per_{kind}yr"
        ]
        dest_df[f"Inv_Cost_per_{kind}yr"] = base_df.loc[:, f"Inv_Cost_per_{kind}yr"]

    dest_df.to_csv(dest_path, index=False)
