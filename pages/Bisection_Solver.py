
import math
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sympy import lambdify, latex, sympify, symbols

from components.navigation import navbar
from utilities.ui import load_css


# =============================================================================
# Consistent numerical display formatting
# =============================================================================
_SUPERSCRIPT_TRANSLATION = str.maketrans(
    "0123456789-+",
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺",
)


def format_scientific_power(
    value: float | int | None,
    decimals: int = 3,
    unavailable: str = "—",
) -> str:
    """Format scientific notation as a coefficient multiplied by 10ⁿ."""

    if value is None:
        return unavailable
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    if number == 0.0:
        return f"{0.0:.{decimals}f}"

    exponent = int(math.floor(math.log10(abs(number))))
    mantissa = number / (10.0 ** exponent)
    exponent_text = str(exponent).translate(_SUPERSCRIPT_TRANSLATION)
    return f"{mantissa:.{decimals}f} × 10{exponent_text}"


def format_display_number(
    value: float | int | None,
    decimals: int = 3,
    unavailable: str = "—",
) -> str:
    """Show three decimals and use × 10ⁿ when fixed notation loses meaning."""

    if value is None:
        return unavailable
    number = float(value)
    if not math.isfinite(number):
        return str(number)

    magnitude = abs(number)
    if magnitude != 0.0 and (magnitude < 10.0 ** (-decimals) or magnitude >= 1.0e6):
        return format_scientific_power(number, decimals, unavailable)
    return f"{number:.{decimals}f}"

from openpyxl.chart import BarChart, LineChart, Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# =========================================================
# Page configuration
# =========================================================

EXCEL_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def style_excel_workbook(workbook) -> None:
    """Apply consistent formatting to every worksheet in an exported report."""

    header_fill = PatternFill("solid", fgColor="0D3151")
    header_font = Font(color="FFFFFF", bold=True)

    for worksheet in workbook.worksheets:
        if worksheet.max_row >= 1 and worksheet.max_column >= 1:
            worksheet.freeze_panes = "A2"
            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
            if worksheet.max_row > 1:
                worksheet.auto_filter.ref = worksheet.dimensions

        for column_index in range(1, worksheet.max_column + 1):
            column_letter = get_column_letter(column_index)
            maximum_length = 0
            for cell in worksheet[column_letter]:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                maximum_length = max(
                    maximum_length,
                    len(str(cell.value)) if cell.value is not None else 0,
                )
                if isinstance(cell.value, float):
                    cell.number_format = "0.000000"
            worksheet.column_dimensions[column_letter].width = min(
                max(maximum_length + 2, 12),
                55,
            )

st.set_page_config(
    page_title="Bisection Solver | Numerical Methods",
    page_icon="📘",
    layout="wide",
)


# =========================================================
# CSS and theme
# =========================================================
load_css()


navbar(active_page="solver")


# =========================================================
# Solver helpers
# =========================================================
def create_function(equation_text):
    """Convert an equation into a SymPy expression and NumPy function."""
    x = symbols("x")
    expression = sympify(equation_text)

    other_variables = expression.free_symbols - {x}

    if other_variables:
        raise ValueError("Only x can be used as a variable.")

    function = lambdify(
        x,
        expression,
        modules=["numpy"],
    )

    return expression, function


def evaluate_function(function, value):
    """Evaluate a function and confirm that its result is finite."""
    result = float(function(value))

    if not np.isfinite(result):
        raise ValueError("The function returned a non-finite value.")

    return result


def solve_by_bisection(
    equation_text,
    left_endpoint,
    right_endpoint,
    tolerance,
    max_iterations,
):
    """Solve f(x) = 0 using the Bisection Method."""

    if not equation_text.strip():
        return {
            "status": "error",
            "message": "Enter a function before solving.",
        }

    if left_endpoint >= right_endpoint:
        return {
            "status": "error",
            "message": "The left endpoint a must be smaller than b.",
        }

    if tolerance <= 0:
        return {
            "status": "error",
            "message": "Tolerance must be greater than zero.",
        }

    if max_iterations < 1:
        return {
            "status": "error",
            "message": "Maximum iterations must be at least 1.",
        }

    try:
        expression, function = create_function(equation_text)
    except Exception:
        return {
            "status": "error",
            "message": (
                "Invalid function format. "
                "Example: x**3 - x - 2"
            ),
        }

    try:
        function_a = evaluate_function(
            function,
            left_endpoint,
        )

        function_b = evaluate_function(
            function,
            right_endpoint,
        )

    except Exception:
        return {
            "status": "error",
            "message": (
                "The function could not be evaluated at a or b. "
                "Check the function and interval."
            ),
        }

    original_a = left_endpoint
    original_b = right_endpoint

    if abs(function_a) <= tolerance:
        return {
            "status": "success",
            "converged": True,
            "root": left_endpoint,
            "iterations": 0,
            "history": [],
            "function": function,
            "expression": expression,
            "equation": equation_text,
            "initial_interval": (original_a, original_b),
            "final_interval": (left_endpoint, left_endpoint),
            "final_error": 0.0,
            "function_a": function_a,
            "function_b": function_b,
            "message": "The left endpoint is already a root.",
        }

    if abs(function_b) <= tolerance:
        return {
            "status": "success",
            "converged": True,
            "root": right_endpoint,
            "iterations": 0,
            "history": [],
            "function": function,
            "expression": expression,
            "equation": equation_text,
            "initial_interval": (original_a, original_b),
            "final_interval": (right_endpoint, right_endpoint),
            "final_error": 0.0,
            "function_a": function_a,
            "function_b": function_b,
            "message": "The right endpoint is already a root.",
        }

    if function_a * function_b > 0:
        return {
            "status": "error",
            "message": (
                "The interval is invalid. "
                "f(a) and f(b) must have opposite signs."
            ),
        }

    history = []
    midpoint = None
    final_error = None
    converged = False

    a = left_endpoint
    b = right_endpoint
    f_a = function_a
    f_b = function_b

    for iteration in range(1, max_iterations + 1):
        midpoint = (a + b) / 2
        f_midpoint = evaluate_function(function, midpoint)

        error_bound = abs(b - a) / 2

        if abs(f_midpoint) <= tolerance:
            action = "Root found"

        elif f_a * f_midpoint < 0:
            action = "Set b = c"

        else:
            action = "Set a = c"

        history.append(
            {
                "Iteration": iteration,
                "a": a,
                "b": b,
                "c": midpoint,
                "f(a)": f_a,
                "f(b)": f_b,
                "f(c)": f_midpoint,
                "Error Bound": error_bound,
                "Next Action": action,
            }
        )

        final_error = error_bound

        if (
            abs(f_midpoint) <= tolerance
            or error_bound <= tolerance
        ):
            converged = True
            break

        if f_a * f_midpoint < 0:
            b = midpoint
            f_b = f_midpoint

        else:
            a = midpoint
            f_a = f_midpoint

    if converged:
        message = "Root found successfully."

    else:
        message = (
            "Maximum iterations reached. "
            "The final approximation is shown."
        )

    return {
        "status": "success",
        "converged": converged,
        "root": midpoint,
        "iterations": len(history),
        "history": history,
        "function": function,
        "expression": expression,
        "equation": equation_text,
        "initial_interval": (original_a, original_b),
        "final_interval": (a, b),
        "final_error": final_error,
        "function_a": function_a,
        "function_b": function_b,
        "message": message,
    }



def create_excel_report(result, dataframe):
    """Create one organized Bisection workbook with final answers, tables, and graphs."""

    output = BytesIO()
    summary_dataframe = pd.DataFrame(
        {
            "Property": [
                "Method",
                "Equation",
                "Initial a",
                "Initial b",
                "f(a)",
                "f(b)",
                "Converged",
                "Approximate Root",
                "f(Root)",
                "Iterations",
                "Final Error Bound",
                "Final Interval a",
                "Final Interval b",
                "Message",
            ],
            "Value": [
                "Bisection Method",
                result["equation"],
                result["initial_interval"][0],
                result["initial_interval"][1],
                result["function_a"],
                result["function_b"],
                "Yes" if result["converged"] else "No",
                result["root"],
                float(result["function"](result["root"])),
                result["iterations"],
                result["final_error"],
                result["final_interval"][0],
                result["final_interval"][1],
                result["message"],
            ],
        }
    )

    initial_a, initial_b = result["initial_interval"]
    graph_x = np.linspace(initial_a, initial_b, 400)
    with np.errstate(all="ignore"):
        graph_y = np.asarray(result["function"](graph_x), dtype=float)
    if graph_y.ndim == 0:
        graph_y = np.full_like(graph_x, float(graph_y))
    graph_y[~np.isfinite(graph_y)] = np.nan

    max_rows = max(len(graph_x), len(dataframe))
    plot_data = pd.DataFrame(index=range(max_rows))
    plot_data["Function x"] = pd.Series(graph_x)
    plot_data["f(x)"] = pd.Series(graph_y)
    plot_data["Iteration"] = pd.Series(dataframe.get("Iteration", pd.Series(dtype=float)))
    plot_data["Midpoint c"] = pd.Series(dataframe.get("c", pd.Series(dtype=float)))
    plot_data["Error Bound"] = pd.Series(dataframe.get("Error Bound", pd.Series(dtype=float)))

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_dataframe.to_excel(writer, sheet_name="Summary", index=False)
        dataframe.to_excel(writer, sheet_name="Iterations", index=False)
        plot_data.to_excel(writer, sheet_name="Plot Data", index=False)

        workbook = writer.book
        plot_sheet = workbook["Plot Data"]
        plot_sheet["H1"] = "Graphs based on the report data"
        plot_sheet["H1"].font = Font(bold=True, size=14)

        function_chart = ScatterChart()
        function_chart.title = "Function on the Initial Interval"
        function_chart.x_axis.title = "x"
        function_chart.y_axis.title = "f(x)"
        function_chart.height = 8
        function_chart.width = 15
        function_chart.series.append(
            Series(
                Reference(plot_sheet, min_col=2, min_row=2, max_row=len(graph_x) + 1),
                Reference(plot_sheet, min_col=1, min_row=2, max_row=len(graph_x) + 1),
                title="f(x)",
            )
        )
        plot_sheet.add_chart(function_chart, "H3")

        if len(dataframe) > 0:
            approximation_chart = LineChart()
            approximation_chart.title = "Root Approximation"
            approximation_chart.x_axis.title = "Iteration"
            approximation_chart.y_axis.title = "Midpoint"
            approximation_chart.height = 8
            approximation_chart.width = 15
            approximation_chart.add_data(
                Reference(plot_sheet, min_col=4, min_row=1, max_row=len(dataframe) + 1),
                titles_from_data=True,
            )
            approximation_chart.set_categories(
                Reference(plot_sheet, min_col=3, min_row=2, max_row=len(dataframe) + 1)
            )
            plot_sheet.add_chart(approximation_chart, "H20")

            error_chart = LineChart()
            error_chart.title = "Error-Bound Convergence"
            error_chart.x_axis.title = "Iteration"
            error_chart.y_axis.title = "Error bound"
            error_chart.height = 8
            error_chart.width = 15
            error_chart.add_data(
                Reference(plot_sheet, min_col=5, min_row=1, max_row=len(dataframe) + 1),
                titles_from_data=True,
            )
            error_chart.set_categories(
                Reference(plot_sheet, min_col=3, min_row=2, max_row=len(dataframe) + 1)
            )
            plot_sheet.add_chart(error_chart, "X3")

        style_excel_workbook(workbook)
        summary_sheet = workbook["Summary"]
        for row in range(2, summary_sheet.max_row + 1):
            if isinstance(summary_sheet.cell(row, 2).value, float):
                summary_sheet.cell(row, 2).number_format = "0.000"

    output.seek(0)
    return finalize_excel_report_with_visible_charts(output.getvalue())



def create_empty_dataframe():
    """Create an empty iteration table."""
    return pd.DataFrame(
        columns=[
            "Iteration",
            "a",
            "b",
            "c",
            "f(a)",
            "f(b)",
            "f(c)",
            "Error Bound",
            "Next Action",
        ]
    )


# =========================================================
# Hero
# =========================================================

def finalize_excel_report_with_visible_charts(report_bytes: bytes) -> bytes:
    """Place existing workbook charts on Summary so they are immediately visible.

    The report data and worksheets are preserved. If Excel chart post-processing is
    unavailable for any reason, the original report is returned unchanged rather
    than risking a damaged workbook.
    """
    try:
        from io import BytesIO as _BytesIO
        from openpyxl import load_workbook as _load_workbook

        workbook = _load_workbook(_BytesIO(report_bytes))
        if "Summary" not in workbook.sheetnames:
            return report_bytes

        summary_sheet = workbook["Summary"]
        collected_charts = list(summary_sheet._charts)
        summary_sheet._charts = []

        for worksheet in workbook.worksheets:
            if worksheet is summary_sheet:
                continue
            if worksheet._charts:
                collected_charts.extend(list(worksheet._charts))
                worksheet._charts = []

        # Put every chart in the existing Summary worksheet, below one another.
        # D2 keeps the first chart visible beside the main results when the report opens.
        for chart_index, chart in enumerate(collected_charts):
            anchor_row = 2 + chart_index * 19
            summary_sheet.add_chart(chart, f"D{anchor_row}")

        workbook.active = workbook.sheetnames.index("Summary")
        output = _BytesIO()
        workbook.save(output)
        return output.getvalue()
    except Exception:
        return report_bytes


st.html(
    f"""
    <section class="solver-hero">
        <div>
            <div class="page-label">
                BISECTION METHOD TOOL
            </div>

            <h1>Bisection Solver</h1>

            <p>
                Enter a continuous function and a valid interval to find
                an approximate root using the Bisection Method. Review
                every iteration, analyze the error, and export the results.
            </p>

            <div class="method-actions">
                <a
                    href="/Bisection_Method"
                    target="_self"
                    class="btn-outline-ui"
                >
                    Review Lesson →
                </a>

                <a
                    href="/Bisection_Quiz"
                    target="_self"
                    class="btn-primary-ui"
                >
                    Take Quiz →
                </a>
            </div>
        </div>
    </section>
    """
)

# =========================================================
# Main content area
# =========================================================
left_margin, main_area, right_margin = st.columns(
    [0.035, 0.93, 0.035]
)


with main_area:

    # =====================================================
    # Instructions and conditions
    # =====================================================
    guide_column, conditions_column = st.columns(2)

    with guide_column:
        with st.container(border=True):
            st.subheader("How to Write the Function")

            st.markdown(
                """
                Enter the expression for **f(x)** without an equals sign.

                Use only **x** as the variable.

                **Writing rules**

                - Powers: write **x\\*\\*2**, not **x^2**
                - Functions: write **sin(x)**, not **sin x**
                - Multiplication: write **2\\*x**, not `2x`
                - Use parentheses when needed: **(x + 1)\\*\\*2**
                - Always write the first letter of mathematical functions in lowercase, such as **sin(x)**, **cos(x)**, **sqrt(x)**, and **log(x)**
                """
            )

    with conditions_column:
        with st.container(border=True):
            st.subheader("Before Solving")

            st.markdown(
                """
                Before using the **Bisection Method**:

                1. The function must be continuous on **[a, b]**.

                2. The endpoint values must have opposite signs, unless one endpoint is already a root.
                """
            )

            st.latex(r"f(a)\,f(b)<0")

            st.markdown("The midpoint is calculated using:")

            st.latex(r"c=\frac{a+b}{2}")

            st.info(
                "The method is guaranteed to converge when the function is "
                "continuous and the interval contains a sign change."
            )

    # =====================================================
    # Input and result
    # =====================================================
    input_column, result_column = st.columns(2)

    with input_column:
        with st.container(border=True):
            st.subheader("Input")

            equation = st.text_input(
                "Function f(x)",
                value="x**3 - x - 2",
                help=(
                    "Examples: x**3 - x - 2, "
                    "sin(x) - 0.5, sqrt(x) - 2"
                ),
            )

            endpoint_column1, endpoint_column2 = st.columns(2)

            with endpoint_column1:
                a = st.number_input(
                    "Left endpoint a",
                    value=1.0,
                )

            with endpoint_column2:
                b = st.number_input(
                    "Right endpoint b",
                    value=2.0,
                )

            settings_column1, settings_column2 = st.columns(2)

            with settings_column1:
                tolerance = st.number_input(
                    "Tolerance",
                    value=0.0001,
                    min_value=0.00000001,
                    format="%.8f",
                )

            with settings_column2:
                max_iterations = st.number_input(
                    "Maximum iterations",
                    value=100,
                    min_value=1,
                    step=1,
                )

            st.markdown("#### Endpoint Check")

            try:
                _, preview_function = create_function(equation)

                preview_f_a = evaluate_function(
                    preview_function,
                    a,
                )

                preview_f_b = evaluate_function(
                    preview_function,
                    b,
                )

                endpoint_result1, endpoint_result2 = st.columns(2)

                with endpoint_result1:
                    st.metric(
                        "f(a)",
                        f"{preview_f_a:.3f}",
                    )

                with endpoint_result2:
                    st.metric(
                        "f(b)",
                        f"{preview_f_b:.3f}",
                    )

                if a >= b:
                    st.warning(
                        "The value of a must be smaller than b."
                    )

                elif abs(preview_f_a) <= tolerance:
                    st.success(
                        "The left endpoint is already a root."
                    )

                elif abs(preview_f_b) <= tolerance:
                    st.success(
                        "The right endpoint is already a root."
                    )

                elif preview_f_a * preview_f_b < 0:
                    st.success(
                        "Valid interval: the endpoint values have "
                        "opposite signs."
                    )

                else:
                    st.warning(
                        "Invalid interval: f(a) and f(b) do not have "
                        "opposite signs."
                    )

            except Exception:
                st.warning(
                    "The function preview is unavailable. "
                    "Check the function format."
                )

            solve_column, reset_column = st.columns(2)

            with solve_column:
                solve_button = st.button(
                    "Solve",
                    type="primary",
                    use_container_width=True,
                )

            with reset_column:
                reset_button = st.button(
                    "Reset",
                    use_container_width=True,
                )

    if reset_button:
        st.session_state.pop(
            "bisection_result",
            None,
        )
        st.rerun()

    if solve_button:
        st.session_state.bisection_result = solve_by_bisection(
            equation_text=equation,
            left_endpoint=a,
            right_endpoint=b,
            tolerance=tolerance,
            max_iterations=int(max_iterations),
        )

        st.rerun()

    with result_column:
        with st.container(border=True):
            st.subheader("Final Result")

            if "bisection_result" not in st.session_state:
                st.info(
                    "Enter the function and interval, then click Solve."
                )

            else:
                result = st.session_state.bisection_result

                if result["status"] == "error":
                    st.error(result["message"])

                    st.markdown(
                        """
                        **Common mistakes**

                        - Writing **x^2** instead of **x\\*\\*2**
                        - Writing `2x` instead of **2\\*x**
                        - Writing **sin x** instead of **sin(x)**
                        - Choosing **a ≥ b**
                        - Choosing endpoints without a sign change
                        """
                    )

                else:
                    if result["converged"]:
                        st.success(result["message"])

                    else:
                        st.warning(result["message"])

                    st.markdown("**Function**")

                    mathematical_function = latex(
                        result["expression"]
                    )

                    st.latex(
                        rf"f(x) = {mathematical_function}"
                    )

                    metric_column1, metric_column2 = st.columns(2)

                    with metric_column1:
                        st.metric(
                            "Approximate Root",
                            f"{result['root']:.3f}",
                        )

                    with metric_column2:
                        st.metric(
                            "Iterations",
                            result["iterations"],
                        )

                    metric_column3, metric_column4 = st.columns(2)

                    with metric_column3:
                        st.metric(
                            "Final Error Bound",
                            f"{result['final_error']:.3f}",
                        )

                    with metric_column4:
                        st.metric(
                            "Converged",
                            "Yes"
                            if result["converged"]
                            else "No",
                        )

                    final_a, final_b = result["final_interval"]

                    st.info(
                        f"Final interval: "
                        f"[{final_a:.3f}, {final_b:.3f}]"
                    )

    # =====================================================
    # Results
    # =====================================================
    if (
        "bisection_result" in st.session_state
        and st.session_state.bisection_result["status"]
        == "success"
    ):
        result = st.session_state.bisection_result
        history = result["history"]

        with st.container(border=True):
            st.subheader("Iteration Table")

            if history:
                full_dataframe = pd.DataFrame(history)
                display_dataframe = full_dataframe.copy()

                numeric_columns = [
                    "a",
                    "b",
                    "c",
                    "f(a)",
                    "f(b)",
                    "f(c)",
                    "Error Bound",
                ]

                display_dataframe[numeric_columns] = (
                    display_dataframe[numeric_columns].round(3)
                )

                table_html = display_dataframe.to_html(
                    index=False,
                    classes="iteration-table",
                    border=0,
                )

                st.markdown(
                    f"""
                    <div class="iteration-table-wrapper">
                        {table_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Calculations use full precision. The displayed "
                    "values are rounded only for readability."
                )

            else:
                full_dataframe = create_empty_dataframe()
                display_dataframe = full_dataframe.copy()

                st.info(
                    "No iterations were required because an endpoint "
                    "was already a root."
                )

        # =================================================
        # Error analysis
        # =================================================
        with st.container(border=True):
            st.subheader("Error Analysis")

            error_metric1, error_metric2, error_metric3 = (
                st.columns(3)
            )

            with error_metric1:
                st.metric(
                    "Initial Error Bound",
                    (
                        f"{history[0]['Error Bound']:.3f}"
                        if history
                        else "0"
                    ),
                )

            with error_metric2:
                st.metric(
                    "Final Error Bound",
                    f"{result['final_error']:.3f}",
                )

            with error_metric3:
                st.metric(
                    "Requested Tolerance",
                    f"{tolerance:.3f}",
                )

            st.markdown(
                "The theoretical Bisection Method error bound is:"
            )

            st.latex(
                r"|x_{\mathrm{true}}-c_n|"
                r"\leq\frac{b_n-a_n}{2}"
            )

            if result["converged"]:
                st.success(
                    "The stopping condition was satisfied."
                )

            else:
                st.warning(
                    "The maximum number of iterations was reached "
                    "before the tolerance was satisfied."
                )

        # =================================================
        # Downloads
        # =================================================
        with st.container(border=True):
            st.subheader("Download Reports")

            st.write(
                "Download the iteration results as CSV or Excel."
            )
            st.subheader("Excel Report")
            excel_data = create_excel_report(result, display_dataframe)
            st.download_button(
                label="Download Complete Excel Report",
                data=excel_data,
                file_name="bisection_complete_report.xlsx",
                mime=EXCEL_MIME_TYPE,
                use_container_width=True,
                key="bisection_excel_report",
            )

        # =================================================
        # Graph data
        # =================================================
        function = result["function"]
        initial_a, initial_b = result["initial_interval"]

        iterations = [
            row["Iteration"]
            for row in history
        ]

        midpoint_values = [
            row["c"]
            for row in history
        ]

        error_bounds = [
            row["Error Bound"]
            for row in history
        ]

        with st.container(border=True):
            st.subheader("Graphs and Convergence Analysis")

            st.write(
                "Examine the function, root approximations, and error."
            )

        # =================================================
        # Three graphs in one row
        # =================================================
        function_column, root_column, error_column = st.columns(3)

        with function_column:
            with st.container(border=True):
                st.subheader("Function Graph")

                try:
                    x_values = np.linspace(
                        initial_a,
                        initial_b,
                        400,
                    )

                    y_values = function(x_values)

                    figure1, axis1 = plt.subplots(
                        figsize=(3.4, 2.6)
                    )

                    axis1.plot(
                        x_values,
                        y_values,
                        linewidth=2,
                        label="f(x)",
                    )

                    axis1.axhline(
                        0,
                        linewidth=1,
                    )

                    axis1.axvline(
                        initial_a,
                        linestyle="--",
                        linewidth=1,
                        label="a",
                    )

                    axis1.axvline(
                        initial_b,
                        linestyle="--",
                        linewidth=1,
                        label="b",
                    )

                    axis1.scatter(
                        result["root"],
                        function(result["root"]),
                        s=40,
                        label="Estimated root",
                        zorder=5,
                    )

                    axis1.set_xlabel("x")
                    axis1.set_ylabel("f(x)")
                    axis1.grid(True)
                    axis1.legend(fontsize=7)

                    st.pyplot(
                        figure1,
                        use_container_width=True,
                    )

                    plt.close(figure1)

                except Exception:
                    st.warning(
                        "The function graph could not be created."
                    )

        with root_column:
            with st.container(border=True):
                st.subheader("Root Convergence")

                if midpoint_values:
                    figure2, axis2 = plt.subplots(
                        figsize=(3.4, 2.6)
                    )

                    axis2.plot(
                        iterations,
                        midpoint_values,
                        marker="o",
                        markersize=4,
                        label="Midpoint",
                    )

                    axis2.axhline(
                        result["root"],
                        linestyle="--",
                        linewidth=1.2,
                        label="Final root",
                    )

                    axis2.set_xlabel("Iteration")
                    axis2.set_ylabel("Root")
                    axis2.grid(True)
                    axis2.legend(fontsize=7)

                    st.pyplot(
                        figure2,
                        use_container_width=True,
                    )

                    plt.close(figure2)

                else:
                    st.info(
                        "No convergence graph was required."
                    )

        with error_column:
            with st.container(border=True):
                st.subheader("Error Graph")

                if error_bounds:
                    figure3, axis3 = plt.subplots(
                        figsize=(3.4, 2.6)
                    )

                    axis3.plot(
                        iterations,
                        error_bounds,
                        marker="o",
                        markersize=4,
                    )

                    axis3.axhline(
                        tolerance,
                        linestyle="--",
                        linewidth=1.2,
                        label="Tolerance",
                    )

                    axis3.set_xlabel("Iteration")
                    axis3.set_ylabel("Error")
                    axis3.grid(True)
                    axis3.legend(fontsize=7)

                    st.pyplot(
                        figure3,
                        use_container_width=True,
                    )

                    plt.close(figure3)

                else:
                    st.info(
                        "Not enough data to display the error graph."
                    )

        # =================================================
        # Bottom navigation
        # =================================================
        with st.container(border=True):
            st.subheader("Continue Learning")

            navigation_column1, navigation_column2 = st.columns(2)

            with navigation_column1:
                if st.button(
                    "Review Bisection Lesson",
                    use_container_width=True,
                ):
                    st.switch_page(
                        "pages/Bisection_Method.py"
                    )

            with navigation_column2:
                if st.button(
                    "Back to Solver Menu",
                    use_container_width=True,
                ):
                    st.switch_page(
                        "pages/Numerical_Solver.py"
                    )


# =========================================================
# Footer
# =========================================================
st.html(
    """
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>Bisection Solver • Root Finding</div>
    </footer>
    """
)