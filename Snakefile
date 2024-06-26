from pathlib import Path
import hashlib

# from rules.common import md5, check_md5

def md5(fnames: [str]) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    for fname in fnames:
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()

def read_hash(done_file: str) -> str:
    if Path(done_file).exists():
        with open(done_file, 'r') as file:
            return file.read().strip()

def check_md5(input_file: [str], done_file: Path) -> bool:
    """Check if the MD5 hash of the input file matches the hash stored in the .done file."""
    if done_file.exists():
        old_hash = read_hash(done_file)
        new_hash = md5(input_file)
        if old_hash == new_hash:
            return True
    return False


def my_inputs(wildcards):
    folders = Path.cwd().rglob("*_results_summary")
    files = []
    for f in folders:
        if "26z" not in str(f) and ("full" in f.parts[-2] or "short" in f.parts[-2] or "20-week" in f.parts[-2]):
            if (f / "resource_capacity.csv").exists() and (f / "transmission_expansion.csv").exists():
                files.append(f"{f.parts[-2]}/{f.parts[-1]}.done")
    return files

rule all:
    input:
        my_inputs


rule process_results_summary:
    input:
        input_files=[
            "{dir}/{model}_results_summary/resource_capacity.csv",
            "{dir}/{model}_results_summary/transmission_expansion.csv",
        ]
    output:
        # touch("{dir}/{model}_results_summary.done"),
        done_file="{dir}/{model}_results_summary.done"
    params:
        dir=lambda wildcards: wildcards.dir,
        model=lambda wildcards: wildcards.model
    run:
        # Check if the .done file exists and compare MD5 hashes
        # Hash both the input data file and the python script
        hash_files = input.input_files + [str(Path.cwd() / "bin" / "results_to_genx_inputs.py")]
        print(hash_files)
        print(read_hash(output.done_file))
        print(md5(hash_files))
        if check_md5(hash_files, Path(output.done_file)):
            print(f"No changes detected in {input.input_files}. Skipping processing.")
        else:
            # Run the processing script
            shell("""
                echo "Running script on {input_file}"
                cd bin
                bash results_to_genx_inputs.sh {dir} {model}
                cd ..
                """,
                input_file=input.input_files[0], dir=params.dir, model=params.model)

            # Write the new MD5 hash to the .done file
            new_hash = md5(hash_files)
            with open(output.done_file, 'w') as f:
                f.write(new_hash)
