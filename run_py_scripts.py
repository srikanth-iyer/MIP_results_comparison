from pathlib import Path
import subprocess
import os
import re
from joblib import Parallel, delayed

CWD = Path.cwd()


def run_script(f: Path):
    os.chdir(f.parent)
    print(f)
    subprocess.run([f"python {str(f)}"], shell=True)
    os.chdir(CWD)


def run_scripts_in_dir(f: Path):
    scripts = (CWD / f).glob("figures.py")
    for s in scripts:
        os.chdir(s.parent)
        print(s)
        run_script(s)
        os.chdir(CWD)


def git_changed_files():
    """
    Retrieves the names of files that have been modified or added in the last git commit.

    Returns:
    - A list of file names that have been modified or added in the last git commit.
    """
    output = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "HEAD~1"])
    files = output.decode().splitlines()
    files = [f.rstrip() for f in files]
    return files


def figure_scripts(mod_folders):
    scripts = list(CWD.rglob("figures.py"))
    scripts = [s for s in scripts if s.parts[-2] in mod_folders]
    return scripts


def main():
    # Get modified or added files from the last git commit
    files = git_changed_files()
    folders = [f.split("/")[0] for f in files]
    scripts = figure_scripts(folders)
    run = False
    for f in files:
        if ".py" in f or ".csv" in f:
            run = True

    if run and scripts:
        print("Running figure generation scripts")

        # Run scripts in the parent directory of the modified files
        Parallel(n_jobs=-1)(delayed(run_script)(Path(f)) for f in scripts)
    else:
        print("No modified .py or .csv files")


if __name__ == "__main__":
    main()
