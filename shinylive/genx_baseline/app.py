from __future__ import annotations

import functools
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Dict, List

import altair as alt
import pandas as pd
from shiny import App, reactive, render, ui
from pandas.api.types import is_numeric_dtype

# Location of the compiled GenX baseline result CSV files
DATA_DIR = Path(__file__).resolve().parents[2] / "compiled_results" / "all"

if not DATA_DIR.exists():  # pragma: no cover - defensive guard for deployment
    raise FileNotFoundError(
        "Unable to locate compiled GenX baseline data in 'compiled_results/all'."
    )

# The subset of cases that form the baseline collection showcased in the app.
BASELINE_CASES: Sequence[str] = (
    "20-week-foresight",
    "20-week-myopic",
    "full-base-50",
    "full-base-200",
    "full-base-200-no-ccs",
    "full-base-200-retire",
    "full-base-200-commit",
    "full-base-200-tx-0",
    "full-base-200-tx-15",
    "full-base-200-tx-50",
    "full-base-1000",
)

FILTER_DEFINITIONS: Sequence[Dict[str, object]] = (
    {
        "key": "case",
        "label": "Case(s)",
        "columns": ("case", "case_name"),
        "default_pool": BASELINE_CASES,
    },
    {
        "key": "planning_year",
        "label": "Planning year(s)",
        "columns": ("planning_year", "year", "period"),
    },
    {
        "key": "zone",
        "label": "Zone(s)",
        "columns": ("zone", "region", "agg_zone"),
    },
    {
        "key": "scenario",
        "label": "Scenario(s)",
        "columns": ("scenario", "configuration", "child_scenario"),
    },
    {
        "key": "technology",
        "label": "Technology / Resource",
        "columns": ("technology", "tech_type", "resource", "fuel"),
    },
)


@functools.lru_cache(maxsize=None)
def available_datasets() -> List[str]:
    """Return the CSV files that can be explored in the application."""

    return sorted(p.name for p in DATA_DIR.glob("*.csv"))


@functools.lru_cache(maxsize=None)
def load_dataset(name: str) -> pd.DataFrame:
    """Load and cache an individual CSV as a DataFrame."""

    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{name}' is not available in {DATA_DIR}.")
    return pd.read_csv(path)


def _coerce_iterable(value: Iterable[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


DEFAULT_DATASET = "avg_total_gen.csv" if (DATA_DIR / "avg_total_gen.csv").exists() else None


app_ui = ui.page_fluid(
    ui.h2("GenX baseline explorer"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_select(
                "dataset",
                "Dataset",
                choices=available_datasets(),
                selected=DEFAULT_DATASET or available_datasets()[0],
            ),
            ui.tags.hr(),
            ui.strong("Filters"),
            *(ui.output_ui(f"filter_{definition['key']}") for definition in FILTER_DEFINITIONS),
        ),
        ui.panel_main(
            ui.layout_columns(
                ui.value_box(
                    "Rows",
                    ui.output_text("row_count"),
                    showcase=ui.span(class_="bi bi-table"),
                ),
                ui.value_box(
                    "Cases",
                    ui.output_text("case_count"),
                    showcase=ui.span(class_="bi bi-diagram-3"),
                ),
                ui.value_box(
                    "Planning years",
                    ui.output_text("year_count"),
                    showcase=ui.span(class_="bi bi-calendar3"),
                ),
            ),
            ui.card(
                ui.card_header("Visualization"),
                ui.output_plot("dataset_plot", height="450px"),
            ),
            ui.card(
                ui.card_header("Tabular data"),
                ui.output_table("dataset_table"),
            ),
        ),
    ),
)


def _format_title(column: str) -> str:
    return column.replace("_", " ").strip().capitalize()


def _determine_value_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "value",
        "end_value",
        "total",
        "amount",
        "generation",
        "capacity",
        "emissions",
    ]
    for candidate in candidates:
        if candidate in df.columns and is_numeric_dtype(df[candidate]):
            return candidate

    ignored = {"min", "max", "lower", "upper", "count"}
    for column in df.columns:
        if column in ignored:
            continue
        if is_numeric_dtype(df[column]):
            return column
    return None


def _choose_color_column(df: pd.DataFrame, exclude: Sequence[str]) -> str | None:
    for column in (
        "case",
        "case_name",
        "technology",
        "tech_type",
        "resource",
        "fuel",
        "scenario",
        "configuration",
        "zone",
        "region",
    ):
        if column in df.columns and column not in exclude:
            return column
    return None


def _filter_dataframe(df: pd.DataFrame, column: str, values: Iterable[str]) -> pd.DataFrame:
    values = list(values)
    if not values or column not in df.columns:
        return df
    # Compare as strings to avoid dtype mismatches.
    string_values = {str(v) for v in values}
    return df[df[column].astype(str).isin(string_values)]


def server(input, output, session):
    @reactive.Calc
    def dataset_name() -> str:
        return input.dataset()

    @reactive.Calc
    def dataset_df() -> pd.DataFrame:
        df = load_dataset(dataset_name()).copy()
        if "case" in df.columns:
            mask = df["case"].isin(BASELINE_CASES)
            if mask.any():
                df = df[mask]
        return df

    @reactive.Calc
    def filter_columns() -> Dict[str, str]:
        df = dataset_df()
        mapping: Dict[str, str] = {}
        for definition in FILTER_DEFINITIONS:
            columns = tuple(definition["columns"])  # type: ignore[index]
            for column in columns:
                if column in df.columns:
                    mapping[str(definition["key"])] = column  # type: ignore[index]
                    break
        return mapping

    def get_input_values(name: str) -> List[str]:
        try:
            value = getattr(input, name)()
        except Exception:  # noqa: BLE001 - fall back to empty when input is undefined
            return []
        return _coerce_iterable(value)

    def create_filter_ui(definition: Dict[str, object]) -> ui.TagChild | str:
        key = str(definition["key"])
        column = filter_columns().get(key)
        if column is None:
            return ""

        df = dataset_df()
        if column not in df.columns:
            return ""

        values = [
            str(v)
            for v in sorted({val for val in df[column].dropna().tolist()}, key=lambda x: str(x))
        ]
        if not values:
            return ""

        default_pool = definition.get("default_pool")
        default_selection: List[str] = []
        if isinstance(default_pool, Sequence) and not isinstance(default_pool, (str, bytes)):
            default_selection = [str(val) for val in default_pool if str(val) in values]
        if not default_selection:
            default_selection = values

        current_value = get_input_values(key)
        if current_value:
            selected = [val for val in current_value if val in values] or default_selection
        else:
            selected = default_selection

        return ui.input_selectize(
            key,
            str(definition["label"]),
            choices=values,
            selected=selected,
            multiple=True,
        )

    def register_filter_renderer(definition: Dict[str, object]) -> None:
        key = str(definition["key"])

        @output(id=f"filter_{key}")  # type: ignore[misc]
        @render.ui
        def _render_filter() -> ui.TagChild | str:
            return create_filter_ui(definition)

    for definition in FILTER_DEFINITIONS:
        register_filter_renderer(definition)

    @reactive.Calc
    def filtered_df() -> pd.DataFrame:
        df = dataset_df()
        for definition in FILTER_DEFINITIONS:
            key = str(definition["key"])
            column = filter_columns().get(key)
            if column:
                df = _filter_dataframe(df, column, get_input_values(key))
        return df

    @output
    @render.text
    def row_count() -> str:
        return f"{len(filtered_df()):,}"

    @output
    @render.text
    def case_count() -> str:
        df = filtered_df()
        column = filter_columns().get("case")
        if not column or column not in df.columns:
            return "–"
        return f"{df[column].nunique():,}"

    @output
    @render.text
    def year_count() -> str:
        df = filtered_df()
        column = filter_columns().get("planning_year")
        if not column or column not in df.columns:
            return "–"
        return f"{df[column].nunique():,}"

    @output
    @render.plot
    def dataset_plot():
        df = filtered_df()
        if df.empty:
            return alt.Chart(pd.DataFrame({"text": ["No data for the selected filters."]})).mark_text(
                align="center", baseline="middle", fontSize=18
            ).encode(text="text", x=alt.value(0), y=alt.value(0))

        plot_df = df.copy()
        value_column = _determine_value_column(plot_df)
        if value_column is None:
            return alt.Chart(pd.DataFrame({"text": ["This dataset does not contain numeric values to plot."]})).mark_text(
                align="center", baseline="middle", fontSize=18
            ).encode(text="text", x=alt.value(0), y=alt.value(0))

        preferred_axes = [
            ("planning_year", "O"),
            ("year", "O"),
            ("period", "O"),
            ("zone", "N"),
            ("region", "N"),
            ("agg_zone", "N"),
            ("technology", "N"),
            ("tech_type", "N"),
            ("resource", "N"),
            ("fuel", "N"),
        ]

        x_column = None
        x_type = "N"
        for column, encoding in preferred_axes:
            if column in plot_df.columns and column != value_column:
                x_column = column
                x_type = encoding
                break

        if x_column is None:
            for column in plot_df.columns:
                if column == value_column:
                    continue
                if not is_numeric_dtype(plot_df[column]):
                    x_column = column
                    x_type = "N"
                    break

        if x_column is None:
            plot_df = plot_df.reset_index(drop=False).rename(columns={"index": "row_index"})
            x_column = "row_index"
            x_type = "O"

        mark = "line" if x_column in {"planning_year", "year", "period"} else "bar"
        color_column = _choose_color_column(plot_df, exclude=[x_column, value_column])

        tooltip_fields = []
        for column in plot_df.columns:
            if column == value_column:
                tooltip_fields.append(f"{column}:Q")
            elif is_numeric_dtype(plot_df[column]):
                tooltip_fields.append(f"{column}:Q")
            else:
                tooltip_fields.append(f"{column}:N")

        chart = alt.Chart(plot_df)
        if mark == "line":
            chart = chart.mark_line(point=True)
        else:
            chart = chart.mark_bar()

        encoding = chart.encode(
            x=alt.X(f"{x_column}:{x_type}", title=_format_title(x_column)),
            y=alt.Y(f"{value_column}:Q", title=_format_title(value_column)),
            tooltip=tooltip_fields,
        )

        if color_column:
            encoding = encoding.encode(color=alt.Color(f"{color_column}:N", title=_format_title(color_column)))
        else:
            encoding = encoding.encode(color=alt.value("#1f77b4"))

        return encoding

    @output
    @render.table
    def dataset_table():
        df = filtered_df()
        if df.empty:
            base = dataset_df()
            columns: List[str] = list(base.columns) if not base.empty else []
            return pd.DataFrame(columns=columns)
        return df.reset_index(drop=True)


app = App(app_ui, server)
