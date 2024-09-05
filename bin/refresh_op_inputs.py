import hashlib
import os
import subprocess
from pathlib import Path

from joblib import Parallel, delayed


def md5(fnames):
    """Calculate MD5 hash of a list of files."""
    hash_md5 = hashlib.md5()
    for fname in fnames:
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()


def read_hash(done_file):
    """Read the hash stored in a .done file."""
    if Path(done_file).exists():
        with open(done_file, "r") as file:
            return file.read().strip()
    return None


def check_md5(input_files, done_file):
    """Check if the MD5 hash of input files matches the hash in the .done file."""
    if Path(done_file).exists():
        old_hash = read_hash(done_file)
        new_hash = md5(input_files)
        if old_hash == new_hash:
            return True
    return False


def _check_and_run(dir_path: Path):

    resource_capacity_file = dir_path / "resource_capacity.csv"
    transmission_expansion_file = dir_path / "transmission_expansion.csv"
    folder = dir_path.parts[-2]
    model = dir_path.parts[-1].split("_")[0]
    done_file = dir_path.parent / f"{model}_results_summary.done"

    # Check if both files exist
    if not resource_capacity_file.exists() or not transmission_expansion_file.exists():
        print(f"Missing required files in {dir_path}")
        return None

    # Define input files and auxiliary scripts
    input_files = [str(resource_capacity_file), str(transmission_expansion_file)]
    script_files = [
        "results_to_genx_inputs.py",
        "results_to_genx_inputs.sh",
    ]
    all_files_to_check = input_files + script_files

    short_path = "/".join(dir_path.parts[-2:])
    print(f"Checking {short_path}...")

    if check_md5(all_files_to_check, done_file):
        print(f"No changes detected in {input_files}. Skipping processing.")
    else:
        # Run the processing script
        print(f"Changes detected. Running script for {short_path}.")
        try:

            # bash_line = f"bash results_to_genx_inputs.sh {folder} {model}"
            # print(bash_line)
            subprocess.run(
                [
                    "bash",
                    "results_to_genx_inputs.sh",
                    folder,
                    model,
                ],
                check=True,
            )
            # Write the new MD5 hash to the .done file
            new_hash = md5(all_files_to_check)
            with open(done_file, "w") as f:
                f.write(new_hash)
            print(f"Updated .done file for {dir_path.parts[-2]}.")
        except subprocess.CalledProcessError as e:
            print(f"Error while processing {dir_path}: {e}")


def main():
    # Define the directories and models you're working with
    directories = [
        d
        for d in Path.cwd().parent.rglob("*_results_summary")
        if "26z" not in str(d) and ("full" in d.parts[-2] or "20-week" in d.parts[-2])
    ]

    Parallel(n_jobs=-1)(delayed(_check_and_run)(dir_path) for dir_path in directories)


if __name__ == "__main__":
    main()
