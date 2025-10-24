import gzip
import os

import pandas as pd


def gz_to_csv(gz_file_path: str) -> str | None:
    """Convert a gzipped CSV into an uncompressed CSV alongside the source file."""

    if not os.path.exists(gz_file_path):
        raise FileNotFoundError(f"File not found: {gz_file_path}")

    csv_file_path = gz_file_path.replace('.gz', '.csv')

    try:
        with gzip.open(gz_file_path, 'rt', encoding='utf-8') as gz_file:
            df = pd.read_csv(gz_file)
            print(df.head())
        df.to_csv(csv_file_path, index=False)
        print(f"Successfully converted {gz_file_path} to {csv_file_path}")
        return csv_file_path
    except Exception as e:
        print(f"Error converting file: {e}")
        return None


if __name__ == "__main__":
    gz_file = "C:\\Users\\Sriki\\MIP_results_comparison-1\\20-week-genx\\p4_Mod_Elect_Low_RE_results_summary\\dispatch.csv.gz"
    csv_output = gz_to_csv(gz_file)
    if csv_output:
        print(f"CSV file created at: {csv_output}")
