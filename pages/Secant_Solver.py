CONFIG = {'method_id': 'secant',
 'title': 'Secant Solver',
 'label': 'SECANT METHOD TOOL',
 'description': 'Enter a nonlinear function and two starting values to approximate a root without evaluating '
                'an analytical derivative.',
 'lesson': 'Secant_Method',
 'quiz': 'Secant_Quiz',
 'footer': 'Secant Solver • Root Finding',
 'default_eq': 'x**3 - x - 2',
 'inputs': [('x0', 'Initial approximation x₀', 1.0), ('x1', 'Initial approximation x₁', 2.0)],
 'conditions': ['Two distinct starting values are required.',
                'Consecutive function values must not produce a zero denominator.',
                'Convergence is not guaranteed, so inspect the iteration history.'],
 'formula': 'x_{n+1}=x_n-\\frac{f(x_n)(x_n-x_{n-1})}{f(x_n)-f(x_{n-1})}'}


import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sympy import diff, lambdify, latex, sympify, symbols

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

from io import BytesIO
from openpyxl.chart import BarChart, LineChart, Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter



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
    page_title=f"{CONFIG['title']} | Numerical Methods",
    page_icon="📘",
    layout="wide",
)

load_css()
navbar(active_page="solver")


def parse_function(equation_text):
    if not equation_text.strip():
        raise ValueError("Enter a function before solving.")

    x = symbols("x")
    expression = sympify(equation_text)
    other_symbols = expression.free_symbols - {x}

    if other_symbols:
        raise ValueError("Only x can be used as a variable.")

    derivative_expression = diff(expression, x)
    function = lambdify(x, expression, modules=["numpy"])
    derivative = lambdify(x, derivative_expression, modules=["numpy"])

    return expression, derivative_expression, function, derivative


def safe_value(function, value):
    result = float(function(value))
    if not np.isfinite(result):
        raise ValueError("The function returned a non-finite value.")
    return result


def solve_root(equation_text, values, tolerance, max_iterations):
    if tolerance <= 0:
        return {"status": "error", "message": "Tolerance must be greater than zero."}

    if max_iterations < 1:
        return {"status": "error", "message": "Maximum iterations must be at least 1."}

    try:
        expression, derivative_expression, function, derivative = parse_function(
            equation_text
        )
    except Exception as error:
        return {
            "status": "error",
            "message": f"Invalid function: {error}",
        }

    history = []
    converged = False
    stopping_reason = "Maximum iterations reached."
    root = None

    try:
        if CONFIG["method_id"] == "secant":
            x_previous = float(values["x0"])
            x_current = float(values["x1"])

            if x_previous == x_current:
                raise ValueError("The two starting values must be different.")

            f_previous = safe_value(function, x_previous)
            f_current = safe_value(function, x_current)

            if abs(f_previous) <= tolerance:
                root = x_previous
                converged = True
                stopping_reason = "The first starting value is already a root."

            elif abs(f_current) <= tolerance:
                root = x_current
                converged = True
                stopping_reason = "The second starting value is already a root."

            else:
                for iteration in range(1, max_iterations + 1):
                    denominator = f_current - f_previous

                    if abs(denominator) <= 1e-14:
                        raise ValueError(
                            "The Secant denominator became zero or extremely small."
                        )

                    x_next = (
                        x_current
                        - f_current * (x_current - x_previous) / denominator
                    )
                    f_next = safe_value(function, x_next)
                    absolute_error = abs(x_next - x_current)
                    relative_error = (
                        absolute_error / max(abs(x_next), 1e-15)
                    )

                    history.append(
                        {
                            "Iteration": iteration,
                            "x(n-1)": x_previous,
                            "x(n)": x_current,
                            "x(n+1)": x_next,
                            "f(x(n+1))": f_next,
                            "Absolute Error": absolute_error,
                            "Relative Error": relative_error,
                            "Residual": abs(f_next),
                        }
                    )

                    root = x_next

                    if abs(f_next) <= tolerance:
                        converged = True
                        stopping_reason = "Residual tolerance reached."
                        break

                    if absolute_error <= tolerance:
                        converged = True
                        stopping_reason = "Step-size tolerance reached."
                        break

                    x_previous, x_current = x_current, x_next
                    f_previous, f_current = f_current, f_next

        else:
            x_current = float(values["x0"])
            multiplicity = int(values.get("multiplicity", 1))

            if CONFIG["method_id"] == "multiple" and multiplicity < 1:
                raise ValueError("Multiplicity must be a positive integer.")

            f_current = safe_value(function, x_current)

            if abs(f_current) <= tolerance:
                root = x_current
                converged = True
                stopping_reason = "The starting value is already a root."

            else:
                for iteration in range(1, max_iterations + 1):
                    derivative_value = safe_value(derivative, x_current)

                    if abs(derivative_value) <= 1e-14:
                        raise ValueError(
                            "The derivative became zero or extremely small."
                        )

                    correction_multiplier = (
                        multiplicity
                        if CONFIG["method_id"] == "multiple"
                        else 1
                    )

                    x_next = (
                        x_current
                        - correction_multiplier
                        * f_current
                        / derivative_value
                    )

                    f_next = safe_value(function, x_next)
                    absolute_error = abs(x_next - x_current)
                    relative_error = (
                        absolute_error / max(abs(x_next), 1e-15)
                    )

                    history.append(
                        {
                            "Iteration": iteration,
                            "x(n)": x_current,
                            "f(x(n))": f_current,
                            "f'(x(n))": derivative_value,
                            "x(n+1)": x_next,
                            "f(x(n+1))": f_next,
                            "Absolute Error": absolute_error,
                            "Relative Error": relative_error,
                            "Residual": abs(f_next),
                        }
                    )

                    root = x_next

                    if abs(f_next) <= tolerance:
                        converged = True
                        stopping_reason = "Residual tolerance reached."
                        break

                    if absolute_error <= tolerance:
                        converged = True
                        stopping_reason = "Step-size tolerance reached."
                        break

                    x_current = x_next
                    f_current = f_next

    except Exception as error:
        return {
            "status": "error",
            "message": str(error),
            "history": history,
            "expression": expression,
            "derivative_expression": derivative_expression,
        }

    if root is None:
        root = float(values.get("x1", values["x0"]))

    function_at_root = safe_value(function, root)
    final_error = (
        history[-1]["Absolute Error"]
        if history
        else 0.0
    )

    return {
        "status": "success",
        "converged": converged,
        "message": (
            "Root found successfully."
            if converged
            else "Maximum iterations reached; the last approximation is shown."
        ),
        "stopping_reason": stopping_reason,
        "root": root,
        "function_at_root": function_at_root,
        "iterations": len(history),
        "final_error": final_error,
        "history": history,
        "expression": expression,
        "derivative_expression": derivative_expression,
        "function": function,
        "initial_values": dict(values),
        "tolerance": tolerance,
        "max_iterations": max_iterations,
    }


def format_number(value):
    """Format final solver values consistently to three decimal places."""

    return format_display_number(value, unavailable="Not available")




def create_excel_report(result) -> bytes:
    """Create one organized workbook containing all root-finding output."""

    if result.get("status") != "success":
        raise ValueError("Only a successful calculation can be exported.")

    history = pd.DataFrame(result.get("history", []))
    if history.empty:
        history = pd.DataFrame(
            [{
                "Iteration": 0,
                "Approximation": result["root"],
                "Residual": abs(result["function_at_root"]),
                "Absolute Error": result["final_error"],
            }]
        )

    summary = pd.DataFrame(
        {
            "Property": [
                "Method",
                "Function",
                "Derivative",
                "Converged",
                "Approximate Root",
                "f(Root)",
                "Iterations",
                "Final Error",
                "Tolerance",
                "Maximum Iterations",
                "Stopping Reason",
                "Message",
            ],
            "Value": [
                CONFIG["title"],
                str(result["expression"]),
                str(result.get("derivative_expression", "Not used")),
                "Yes" if result["converged"] else "No",
                result["root"],
                result["function_at_root"],
                result["iterations"],
                result["final_error"],
                result["tolerance"],
                result["max_iterations"],
                result["stopping_reason"],
                result["message"],
            ],
        }
    )

    initial_values = pd.DataFrame(
        {
            "Input": list(result["initial_values"].keys()),
            "Value": list(result["initial_values"].values()),
        }
    )

    approximation_column = "x(n+1)" if "x(n+1)" in history.columns else "Approximation"
    approximation_values = history[approximation_column].to_numpy(dtype=float)
    starting_values = [
        float(value)
        for key, value in result["initial_values"].items()
        if str(key).startswith("x")
    ]
    all_points = starting_values + approximation_values.tolist() + [float(result["root"])]
    minimum_x = min(all_points)
    maximum_x = max(all_points)
    span = maximum_x - minimum_x
    padding = 1.0 if span == 0.0 else max(0.2 * span, 0.5)
    graph_x = np.linspace(minimum_x - padding, maximum_x + padding, 400)
    with np.errstate(all="ignore"):
        graph_y = np.asarray(result["function"](graph_x), dtype=float)
    if graph_y.ndim == 0:
        graph_y = np.full_like(graph_x, float(graph_y))
    graph_y[~np.isfinite(graph_y)] = np.nan

    max_rows = max(len(graph_x), len(history))
    plot_data = pd.DataFrame(index=range(max_rows))
    plot_data["Function x"] = pd.Series(graph_x)
    plot_data["f(x)"] = pd.Series(graph_y)
    plot_data["Iteration"] = pd.Series(history["Iteration"].to_numpy())
    plot_data["Approximation"] = pd.Series(approximation_values)
    if "Absolute Error" in history.columns:
        plot_data["Absolute Error"] = pd.Series(history["Absolute Error"].to_numpy())
    else:
        plot_data["Absolute Error"] = np.nan
    if "Residual" in history.columns:
        plot_data["Residual"] = pd.Series(history["Residual"].to_numpy())
    else:
        plot_data["Residual"] = np.nan

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        initial_values.to_excel(writer, sheet_name="Inputs", index=False)
        history.to_excel(writer, sheet_name="Iterations", index=False)
        plot_data.to_excel(writer, sheet_name="Plot Data", index=False)

        workbook = writer.book
        plot_sheet = workbook["Plot Data"]
        plot_sheet["H1"] = "Graphs based on the report data"
        plot_sheet["H1"].font = Font(bold=True, size=14)

        function_chart = ScatterChart()
        function_chart.title = "Function and Estimated Root"
        function_chart.x_axis.title = "x"
        function_chart.y_axis.title = "f(x)"
        function_chart.height = 8
        function_chart.width = 15
        function_series = Series(
            Reference(plot_sheet, min_col=2, min_row=2, max_row=len(graph_x) + 1),
            Reference(plot_sheet, min_col=1, min_row=2, max_row=len(graph_x) + 1),
            title="f(x)",
        )
        function_chart.series.append(function_series)
        plot_sheet.add_chart(function_chart, "H3")

        if len(history) > 0:
            convergence_chart = LineChart()
            convergence_chart.title = "Root Approximation by Iteration"
            convergence_chart.x_axis.title = "Iteration"
            convergence_chart.y_axis.title = "Approximation"
            convergence_chart.height = 8
            convergence_chart.width = 15
            convergence_chart.add_data(
                Reference(plot_sheet, min_col=4, min_row=1, max_row=len(history) + 1),
                titles_from_data=True,
            )
            convergence_chart.set_categories(
                Reference(plot_sheet, min_col=3, min_row=2, max_row=len(history) + 1)
            )
            plot_sheet.add_chart(convergence_chart, "H20")

            error_chart = LineChart()
            error_chart.title = "Error and Residual Convergence"
            error_chart.x_axis.title = "Iteration"
            error_chart.y_axis.title = "Magnitude"
            error_chart.height = 8
            error_chart.width = 15
            error_chart.add_data(
                Reference(plot_sheet, min_col=5, max_col=6, min_row=1, max_row=len(history) + 1),
                titles_from_data=True,
            )
            error_chart.set_categories(
                Reference(plot_sheet, min_col=3, min_row=2, max_row=len(history) + 1)
            )
            plot_sheet.add_chart(error_chart, "X3")

        style_excel_workbook(workbook)
        summary_sheet = workbook["Summary"]
        for row in range(2, summary_sheet.max_row + 1):
            if isinstance(summary_sheet.cell(row, 2).value, float):
                summary_sheet.cell(row, 2).number_format = "0.000"

    output.seek(0)
    return finalize_excel_report_with_visible_charts(output.getvalue())


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
            <div class="page-label">{CONFIG['label']}</div>
            <h1>{CONFIG['title']}</h1>
            <p>{CONFIG['description']}</p>
            <div class="method-actions">
                <a href="/{CONFIG['lesson']}" target="_self" class="btn-outline-ui">Review Lesson →</a>
                <a href="/{CONFIG['quiz']}" target="_self" class="btn-primary-ui">Take Quiz →</a>
            </div>
        </div>
    </section>
    """
)

left_margin, main_area, right_margin = st.columns([0.035, 0.93, 0.035])

with main_area:
    guide_column, conditions_column = st.columns(2)

    with guide_column:
        with st.container(border=True):
            st.subheader("How to Write the Function")
            st.markdown(
                """
                Enter **f(x)** without an equals sign and use only **x** as the variable.

                - Powers: `x**2`, not `x^2`
                - Multiplication: `2*x`, not `2x`
                - Functions: `sin(x)`, `cos(x)`, `sqrt(x)`, `exp(x)`, `log(x)`
                - Use parentheses whenever the order of operations could be unclear
                """
            )
            st.markdown("**Method formula**")
            st.latex(CONFIG["formula"])

    with conditions_column:
        with st.container(border=True):
            st.subheader("Before Solving")
            for condition in CONFIG["conditions"]:
                st.markdown(f"- {condition}")
            st.info(
                "Use the iteration table and graphs to verify that the method is behaving sensibly."
            )

    input_column, result_column = st.columns(2)

    with input_column:
        with st.container(border=True):
            st.subheader("Input")

            equation = st.text_input(
                "Function f(x)",
                value=CONFIG["default_eq"],
                key=f"{CONFIG['method_id']}_equation",
            )

            input_values = {}

            for field_name, label, default_value in CONFIG["inputs"]:
                if field_name == "multiplicity":
                    input_values[field_name] = st.number_input(
                        label,
                        min_value=1,
                        value=int(default_value),
                        step=1,
                        key=f"{CONFIG['method_id']}_{field_name}",
                    )
                else:
                    input_values[field_name] = st.number_input(
                        label,
                        value=float(default_value),
                        format="%.10f",
                        key=f"{CONFIG['method_id']}_{field_name}",
                    )

            settings_column1, settings_column2 = st.columns(2)

            with settings_column1:
                tolerance = st.number_input(
                    "Tolerance",
                    min_value=1e-12,
                    value=1e-6,
                    format="%.10f",
                    key=f"{CONFIG['method_id']}_tolerance",
                )

            with settings_column2:
                max_iterations = st.number_input(
                    "Maximum iterations",
                    min_value=1,
                    value=100,
                    step=1,
                    key=f"{CONFIG['method_id']}_max_iterations",
                )

            st.markdown("**Equation Preview**")

            try:
                preview_expression, preview_derivative, _, _ = parse_function(
                    equation
                )
                st.latex(r"f(x)=" + latex(preview_expression))
                if CONFIG["method_id"] != "secant":
                    st.latex(r"f'(x)=" + latex(preview_derivative))
            except Exception as error:
                st.warning(str(error))

            solve_column, reset_column = st.columns(2)

            with solve_column:
                solve_button = st.button(
                    "Solve",
                    type="primary",
                    use_container_width=True,
                    key=f"{CONFIG['method_id']}_solve",
                )

            with reset_column:
                reset_button = st.button(
                    "Reset",
                    use_container_width=True,
                    key=f"{CONFIG['method_id']}_reset",
                )

            if reset_button:
                st.session_state.pop(
                    f"{CONFIG['method_id']}_result",
                    None,
                )
                st.rerun()

            if solve_button:
                st.session_state[f"{CONFIG['method_id']}_result"] = solve_root(
                    equation_text=equation,
                    values=input_values,
                    tolerance=float(tolerance),
                    max_iterations=int(max_iterations),
                )
                st.rerun()

    with result_column:
        with st.container(border=True):
            st.subheader("Final Result")

            result = st.session_state.get(
                f"{CONFIG['method_id']}_result"
            )

            if result is None:
                st.info("Enter values and click Solve to display the result.")

            elif result["status"] == "error":
                st.error(result["message"])

            else:
                if result["converged"]:
                    st.success(result["message"])
                else:
                    st.warning(result["message"])

                metric_column1, metric_column2 = st.columns(2)

                with metric_column1:
                    st.metric(
                        "Approximate Root",
                        format_number(result["root"]),
                    )

                with metric_column2:
                    st.metric(
                        "Iterations",
                        result["iterations"],
                    )

                metric_column3, metric_column4 = st.columns(2)

                with metric_column3:
                    st.metric(
                        "|f(root)|",
                        format_number(abs(result["function_at_root"])),
                    )

                with metric_column4:
                    st.metric(
                        "Final Error",
                        format_number(result["final_error"]),
                    )

                st.info(
                    f"Stopping reason: {result['stopping_reason']}"
                )


    result = st.session_state.get(f"{CONFIG['method_id']}_result")

    if result and result["status"] == "success":
        history = result["history"]
        dataframe = pd.DataFrame(history)

        with st.container(border=True):
            st.subheader("Iteration Table")

            if dataframe.empty:
                st.info(
                    "No iterations were required because the starting value was already a root."
                )
            else:
                display_dataframe = dataframe.copy()
                numeric_columns = display_dataframe.select_dtypes(
                    include=[np.number]
                ).columns
                display_dataframe[numeric_columns] = (
                    display_dataframe[numeric_columns].round(3)
                )
                st.dataframe(
                    display_dataframe,
                    use_container_width=True,
                    hide_index=True,
                )
                st.subheader("Excel Report")
                excel_report = create_excel_report(result)
                st.download_button(
                    "Download Complete Excel Report",
                    data=excel_report,
                    file_name=f"{CONFIG['method_id']}_complete_report.xlsx",
                    mime=EXCEL_MIME_TYPE,
                    use_container_width=True,
                    key=f"{CONFIG['method_id']}_excel_report",
                )

        graph_column, approximation_column, error_column = st.columns(3)

        with graph_column:
            with st.container(border=True):
                st.subheader("Function Graph")

                try:
                    points = []
                    for field_name, _, _ in CONFIG["inputs"]:
                        if field_name.startswith("x"):
                            points.append(
                                float(result["initial_values"][field_name])
                            )
                    if history:
                        points.extend(
                            float(item["x(n+1)"])
                            for item in history
                        )
                    points.append(float(result["root"]))

                    x_min = min(points)
                    x_max = max(points)
                    span = max(x_max - x_min, 1.0)
                    x_values = np.linspace(
                        x_min - 0.75 * span,
                        x_max + 0.75 * span,
                        500,
                    )
                    y_values = np.asarray(
                        result["function"](x_values),
                        dtype=float,
                    )

                    figure, axis = plt.subplots(figsize=(4.2, 3.0))
                    axis.plot(x_values, y_values, linewidth=2)
                    axis.axhline(0, linewidth=1)
                    axis.axvline(result["root"], linestyle="--", linewidth=1)
                    axis.scatter(
                        [result["root"]],
                        [result["function_at_root"]],
                        s=50,
                    )
                    axis.set_xlabel("x")
                    axis.set_ylabel("f(x)")
                    axis.grid(True)
                    st.pyplot(figure, use_container_width=True)
                    plt.close(figure)
                except Exception:
                    st.warning("The function graph could not be drawn.")

        with approximation_column:
            with st.container(border=True):
                st.subheader("Root Approximation")

                if history:
                    approximation_values = [
                        item["x(n+1)"]
                        for item in history
                    ]
                    iterations = list(
                        range(1, len(approximation_values) + 1)
                    )

                    figure, axis = plt.subplots(figsize=(4.2, 3.0))
                    axis.plot(
                        iterations,
                        approximation_values,
                        marker="o",
                    )
                    axis.axhline(
                        result["root"],
                        linestyle="--",
                        linewidth=1,
                    )
                    axis.set_xlabel("Iteration")
                    axis.set_ylabel("Approximation")
                    axis.grid(True)
                    st.pyplot(figure, use_container_width=True)
                    plt.close(figure)
                else:
                    st.info("No iteration history is available.")

        with error_column:
            with st.container(border=True):
                st.subheader("Error Analysis")

                if history:
                    errors = [
                        max(item["Absolute Error"], 1e-18)
                        for item in history
                    ]
                    residuals = [
                        max(item["Residual"], 1e-18)
                        for item in history
                    ]
                    iterations = list(range(1, len(errors) + 1))

                    figure, axis = plt.subplots(figsize=(4.2, 3.0))
                    axis.semilogy(
                        iterations,
                        errors,
                        marker="o",
                        label="Absolute Error",
                    )
                    axis.semilogy(
                        iterations,
                        residuals,
                        marker="o",
                        label="Residual",
                    )
                    axis.set_xlabel("Iteration")
                    axis.set_ylabel("Log Scale")
                    axis.grid(True)
                    axis.legend()
                    st.pyplot(figure, use_container_width=True)
                    plt.close(figure)
                else:
                    st.info("No error history is available.")

        with st.container(border=True):
            st.subheader("Continue Learning")
            navigation_column1, navigation_column2 = st.columns(2)

            with navigation_column1:
                if st.button(
                    "Review Lesson",
                    use_container_width=True,
                    key=f"{CONFIG['method_id']}_lesson_button",
                ):
                    st.switch_page(f"pages/{CONFIG['lesson']}.py")

            with navigation_column2:
                if st.button(
                    "Back to Solver Menu",
                    use_container_width=True,
                    key=f"{CONFIG['method_id']}_menu_button",
                ):
                    st.switch_page("pages/Numerical_Solver.py")


st.html(
    f"""
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>{CONFIG['footer']}</div>
    </footer>
    """
)
