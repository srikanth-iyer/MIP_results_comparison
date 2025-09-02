import gzip
import pandas as pd
import os

def gz_to_csv(gz_file_path):
    """
    Convert a gzipped file to CSV format and save it in the same location.
    
    Args:
        gz_file_path (str): Path to the .gz file
    
    Returns:
        str: Path to the created CSV file
    """
    # Check if file exists
    if not os.path.exists(gz_file_path):
        raise FileNotFoundError(f"File not found: {gz_file_path}")
    
    # Generate output CSV path
    csv_file_path = gz_file_path.replace('.gz', '.csv')
    
    # Read the gzipped file and convert to CSV
    try:
        with gzip.open(gz_file_path, 'rt', encoding='utf-8') as gz_file:
            # Assuming the gz file contains CSV data
            df = pd.read_csv(gz_file)
            print(df.head())
        # Save as CSV
        df.to_csv(csv_file_path, index=False)
        print(f"Successfully converted {gz_file_path} to {csv_file_path}")
        return csv_file_path
        
    except Exception as e:
        print(f"Error converting file: {e}")
        return None

  

if __name__ == "__main__":
    # Sample usage
    gz_file = "C:\\Users\\Sriki\\MIP_results_comparison-1\\20-week-foresight\\GenX_results_summary\\dispatch.csv.gz"  # Replace with your actual .gz file path
    csv_output = gz_to_csv(gz_file)
    if csv_output:
        print(f"CSV file created at: {csv_output}")