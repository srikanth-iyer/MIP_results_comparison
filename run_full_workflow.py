"""One-click helper to export GenX scenarios and rebuild the Quarto dashboard.

Usage:
    python run_full_workflow.py [options]

Options overview:
    --input-folder PATH      Source folder that holds raw GenX scenario outputs. Defaults
                             to ./genx_scenarios_inputs under the repo root.
    --output-folder PATH     Destination for formatted scenario tables. Defaults to
                             ./genx-scenarios; created automatically when missing.
    --quarto PATH            Quarto CLI executable or full path. Useful if Quarto is not
                             on PATH (e.g., "C:/Program Files/Quarto/bin/quarto.exe").
    --render-target TARGET   QMD file or project directory to render. Defaults to
                             genx-notebooks/genx-results-scenarios.qmd; set to "." to
                             render the whole project.
    --render-arg ARG         Additional argument that is forwarded directly to
                             `quarto render`. Repeat this flag to append multiple args
                             in order (e.g., --render-arg "--execute").
    --skip-quarto            Skip the Quarto step entirely; only run data exports.
    --verbose                Turn on verbose logging inside the export pipeline for
                             troubleshooting missing files or warnings.
    --debug-overwrites       Allow generated files to be overwritten instead of failing
                             fast when outputs already exist.

Example:
    python run_full_workflow.py --verbose --render-arg "--execute" --render-arg "--profile prod"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from reformat_data_for_plot.format_genx_result_for_plotting import (
    export_all_genx_scenarios,
)

# Default mapping between scenario period keys and planning years.
SCENARIO_TO_YEAR_MAP = {
    "p1": 2030,
    "p2": 2035,
    "p3": 2040,
    "p4": 2050,
}


def log_step(message: str) -> None:
    """Print a highlighted progress message."""
    separator = "-" * 36
    print(f"\n\n{separator}\n{message}\n{separator}\n\n", flush=True)


def log_detail(message: str) -> None:
    """Print a sub-message indented beneath the current step."""
    print(f"     {message}", flush=True)


def ensure_directory(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")


def run_export(
    input_root: Path,
    output_root: Path,
    *,
    verbose: bool,
    debug_overwrites: bool,
) -> dict[str, Path]:
    log_step(f"[2/3] EXPORTING GENX SCENARIOS from \n{input_root} -> {output_root}")
    results = export_all_genx_scenarios(
        input_root,
        output_root,
        scenario_to_year_map=SCENARIO_TO_YEAR_MAP,
        debug_overwrites=debug_overwrites,
        verbose=verbose,
    )
    if results:
        exported = ", ".join(sorted(results.keys()))
        log_detail(f"Export complete for scenarios: {exported}")
    else:
        log_detail("No scenarios exported (check input folder contents)")
    return results


def run_quarto_render(
    quarto_cmd: str,
    project_root: Path,
    render_target: str | None,
    *,
    extra_args: list[str] | None = None,
) -> None:
    resolved_quarto = shutil.which(quarto_cmd)
    if resolved_quarto is None:
        raise FileNotFoundError(
            "Quarto executable not found. Set --quarto to an explicit path or ensure it is on PATH."
        )

    command = [resolved_quarto, "render"]
    if render_target:
        command.append(render_target)
    if extra_args:
        command.extend(extra_args)

    log_step(f"[3/3] RUNNING QUARTO \n{' '.join(command)} (cwd={project_root})")
    completed = subprocess.run(command, cwd=project_root, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Quarto render failed. Check the Quarto logs above for details."
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GenX export pipeline and rebuild the Quarto dashboard in one command.",
    )

    parser.add_argument(
        "--input-folder",
        type=Path,
        help="Path to source GenX scenario folders (default: ./genx_scenarios_inputs).",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        help="Destination for processed scenarios (default: ./genx-scenarios).",
    )
    parser.add_argument(
        "--quarto",
        default="quarto",
        help="Quarto executable or path (default: 'quarto').",
    )
    parser.add_argument(
        "--render-target",
        default="genx-notebooks/genx-results-scenarios.qmd",
        help="QMD file or project directory to render (default: genx-notebooks/genx-results-scenarios.qmd).",
    )
    parser.add_argument(
        "--render-arg",
        action="append",
        dest="render_args",
        help="Additional argument passed directly to 'quarto render'. Repeat for multiple flags.",
    )
    parser.add_argument(
        "--skip-quarto",
        action="store_true",
        help="Skip the Quarto render step (useful for debugging export only).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging inside the export pipeline.",
    )
    parser.add_argument(
        "--debug-overwrites",
        action="store_true",
        help="Allow overwriting generated files inside the export pipeline.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])
    repo_root = Path(__file__).resolve().parent

    input_root = (args.input_folder or (repo_root / "genx_scenarios_inputs")).resolve()
    output_root = (args.output_folder or (repo_root / "genx-scenarios")).resolve()

    try:
        log_step("[1/3] VALIDATING REQUIRED FOLDERS")
        ensure_directory(input_root, "Input root")
        if not output_root.exists():
            output_root.mkdir(parents=True, exist_ok=True)
            log_detail(f"Created output folder: {output_root}")
        else:
            ensure_directory(output_root, "Output root")

        run_export(
            input_root,
            output_root,
            verbose=args.verbose,
            debug_overwrites=args.debug_overwrites,
        )

        if args.skip_quarto:
            log_step("[3/3] Skipping Quarto render (requested)")
        else:
            render_target = args.render_target or None
            run_quarto_render(
                args.quarto,
                repo_root,
                render_target,
                extra_args=args.render_args,
            )

        log_step("Workflow completed successfully.")
        return 0
    except KeyboardInterrupt:
        log_step("Workflow interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001
        log_step(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
