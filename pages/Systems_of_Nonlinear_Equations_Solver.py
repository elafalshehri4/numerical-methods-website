
from __future__ import annotations

import hashlib
import html
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp

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
    page_title="Systems of Nonlinear Equations Solver | Numerical Methods",
    page_icon="🧮",
    layout="wide",
)

load_css()
navbar(active_page="solver")

st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius:18px!important;
        border:1px solid rgba(15,61,62,.10)!important;
        box-shadow:0 10px 24px rgba(15,61,62,.06)!important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    """
    <section class="solver-hero">
        <div>
            <div class="page-label">TWO-VARIABLE NEWTON SOLVER</div>
            <h1>Systems of Nonlinear Equations Solver</h1>
            <p>
                Enter two nonlinear equations in x and y, choose an initial guess,
                and inspect the residual vector, Jacobian, Newton correction, and convergence history.
            </p>
            <div class="method-actions">
                <a href="/Systems_of_Nonlinear_Equations" target="_self" class="btn-outline-ui">Review Lesson →</a>
                <a href="/Systems_of_Nonlinear_Equations_Quiz" target="_self" class="btn-primary-ui">Take Quiz →</a>
            </div>
        </div>
    </section>
    """
)

X_SYMBOL, Y_SYMBOL = sp.symbols("x y", real=True)
ALLOWED_NAMES = {
    "x": X_SYMBOL,
    "y": Y_SYMBOL,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
}


def parse_equation(text: str) -> sp.Expr:
    cleaned = str(text).strip().replace("^", "**")
    if not cleaned:
        raise ValueError("Both equations are required.")
    if "=" in cleaned:
        left, right = cleaned.split("=", 1)
        cleaned = f"({left})-({right})"
    try:
        expression = sp.sympify(cleaned, locals=ALLOWED_NAMES)
    except Exception as error:
        raise ValueError(
            "Invalid equation. Enter an expression equal to zero, such as x**2 + y**2 - 4."
        ) from error
    unexpected = expression.free_symbols.difference({X_SYMBOL, Y_SYMBOL})
    if unexpected:
        names = ", ".join(sorted(str(symbol) for symbol in unexpected))
        raise ValueError(f"Unexpected symbol(s): {names}.")
    if expression.has(sp.I, sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("An equation contains a non-real or undefined value.")
    return expression


def signature(*values) -> str:
    payload = repr(tuple(str(value) for value in values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def finite_vector(values, label: str) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{label} produced complex values.")
    array = np.asarray(array, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} produced NaN or infinity.")
    return array


def solve_system(
    first_text: str,
    second_text: str,
    x0: float,
    y0: float,
    tolerance: float,
    max_iterations: int,
) -> dict:
    if tolerance <= 0 or not math.isfinite(tolerance):
        raise ValueError("Tolerance must be a positive finite number.")
    if max_iterations <= 0:
        raise ValueError("Maximum iterations must be positive.")

    f1 = parse_equation(first_text)
    f2 = parse_equation(second_text)
    functions = sp.Matrix([f1, f2])
    jacobian_expression = functions.jacobian([X_SYMBOL, Y_SYMBOL])

    residual_function = sp.lambdify((X_SYMBOL, Y_SYMBOL), functions, modules="numpy")
    jacobian_function = sp.lambdify((X_SYMBOL, Y_SYMBOL), jacobian_expression, modules="numpy")

    current = np.array([float(x0), float(y0)], dtype=float)
    history = []

    for iteration in range(max_iterations + 1):
        residual = finite_vector(
            residual_function(current[0], current[1]),
            "Residual vector",
        )
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))

        if iteration == 0:
            correction = np.array([np.nan, np.nan])
            correction_norm = np.nan
            determinant = np.nan
        else:
            correction = last_correction
            correction_norm = float(np.linalg.norm(correction, ord=np.inf))
            determinant = last_determinant

        history.append({
            "Iteration": iteration,
            "x": current[0],
            "y": current[1],
            "f1": residual[0],
            "f2": residual[1],
            "Residual ∞-Norm": residual_norm,
            "Δx": correction[0],
            "Δy": correction[1],
            "Correction ∞-Norm": correction_norm,
            "Jacobian Determinant": determinant,
        })

        if residual_norm <= tolerance:
            return {
                "success": True,
                "message": "Residual tolerance reached.",
                "solution": current.copy(),
                "history": pd.DataFrame(history),
                "f1": str(f1),
                "f2": str(f2),
                "jacobian": str(jacobian_expression),
            }

        if iteration == max_iterations:
            break

        jacobian = np.asarray(
            jacobian_function(current[0], current[1]),
            dtype=float,
        ).reshape(2, 2)
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("The Jacobian contains NaN or infinity.")
        determinant = float(np.linalg.det(jacobian))
        if abs(determinant) <= 1.0e-14:
            raise ValueError(
                "The Jacobian is singular or numerically near singular at the current iterate."
            )

        try:
            correction = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError as error:
            raise ValueError("The Newton correction could not be calculated.") from error
        correction = finite_vector(correction, "Newton correction")
        next_value = current + correction
        if not np.all(np.isfinite(next_value)):
            raise ValueError("The updated iterate is not finite.")

        last_correction = correction
        last_determinant = determinant
        current = next_value

        if np.linalg.norm(correction, ord=np.inf) <= tolerance:
            final_residual = finite_vector(
                residual_function(current[0], current[1]),
                "Final residual",
            )
            history.append({
                "Iteration": iteration + 1,
                "x": current[0],
                "y": current[1],
                "f1": final_residual[0],
                "f2": final_residual[1],
                "Residual ∞-Norm": float(np.linalg.norm(final_residual, ord=np.inf)),
                "Δx": correction[0],
                "Δy": correction[1],
                "Correction ∞-Norm": float(np.linalg.norm(correction, ord=np.inf)),
                "Jacobian Determinant": determinant,
            })
            return {
                "success": True,
                "message": "Correction tolerance reached.",
                "solution": current.copy(),
                "history": pd.DataFrame(history),
                "f1": str(f1),
                "f2": str(f2),
                "jacobian": str(jacobian_expression),
            }

    return {
        "success": False,
        "message": "Maximum iterations reached before convergence.",
        "solution": current.copy(),
        "history": pd.DataFrame(history),
        "f1": str(f1),
        "f2": str(f2),
        "jacobian": str(jacobian_expression),
    }



def create_excel_report(
    result: dict,
    equation_1: str,
    equation_2: str,
    initial_x: float,
    initial_y: float,
    tolerance: float,
    max_iterations: int,
) -> bytes:
    """Create one workbook with the final solution, tables, and convergence graphs."""

    if "error" in result:
        raise ValueError("Only a completed calculation can be exported.")

    history = result["history"].copy()
    solution = np.asarray(result["solution"], dtype=float)
    final_row = history.iloc[-1]
    summary = pd.DataFrame(
        {
            "Property": [
                "Method",
                "First Equation",
                "Second Equation",
                "Initial x",
                "Initial y",
                "Tolerance",
                "Maximum Iterations",
                "Converged",
                "Final x",
                "Final y",
                "Final Residual Infinity Norm",
                "Iterations Recorded",
                "Stopping Message",
                "Jacobian",
            ],
            "Value": [
                "Newton Method for Two Nonlinear Equations",
                equation_1,
                equation_2,
                initial_x,
                initial_y,
                tolerance,
                max_iterations,
                "Yes" if result["success"] else "No",
                solution[0],
                solution[1],
                float(final_row["Residual ∞-Norm"]),
                len(history),
                result["message"],
                result["jacobian"],
            ],
        }
    )
    system_table = pd.DataFrame(
        {
            "Component": ["F1(x, y)", "F2(x, y)", "J(x, y)"],
            "Expression": [result["f1"], result["f2"], result["jacobian"]],
        }
    )
    final_solution = pd.DataFrame(
        {
            "Variable": ["x", "y"],
            "Calculated Value": solution,
        }
    )
    plot_data = history[
        ["Iteration", "x", "y", "Residual ∞-Norm", "Correction ∞-Norm"]
    ].copy()

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        system_table.to_excel(writer, sheet_name="Parsed System", index=False)
        final_solution.to_excel(writer, sheet_name="Final Solution", index=False)
        history.to_excel(writer, sheet_name="Iterations", index=False)
        plot_data.to_excel(writer, sheet_name="Plot Data", index=False)

        workbook = writer.book
        plot_sheet = workbook["Plot Data"]
        plot_sheet["G1"] = "Graphs based on the report data"
        plot_sheet["G1"].font = Font(bold=True, size=14)
        rows = len(plot_data) + 1

        iterate_chart = LineChart()
        iterate_chart.title = "x and y Iterates"
        iterate_chart.x_axis.title = "Iteration"
        iterate_chart.y_axis.title = "Value"
        iterate_chart.height = 8
        iterate_chart.width = 15
        iterate_chart.add_data(
            Reference(plot_sheet, min_col=2, max_col=3, min_row=1, max_row=rows),
            titles_from_data=True,
        )
        iterate_chart.set_categories(
            Reference(plot_sheet, min_col=1, min_row=2, max_row=rows)
        )
        plot_sheet.add_chart(iterate_chart, "G3")

        residual_chart = LineChart()
        residual_chart.title = "Residual and Correction Norms"
        residual_chart.x_axis.title = "Iteration"
        residual_chart.y_axis.title = "Magnitude"
        residual_chart.height = 8
        residual_chart.width = 15
        residual_chart.add_data(
            Reference(plot_sheet, min_col=4, max_col=5, min_row=1, max_row=rows),
            titles_from_data=True,
        )
        residual_chart.set_categories(
            Reference(plot_sheet, min_col=1, min_row=2, max_row=rows)
        )
        plot_sheet.add_chart(residual_chart, "G20")

        style_excel_workbook(workbook)
        summary_sheet = workbook["Summary"]
        for row in range(2, summary_sheet.max_row + 1):
            if isinstance(summary_sheet.cell(row, 2).value, float):
                summary_sheet.cell(row, 2).number_format = "0.000"

    output.seek(0)
    return finalize_excel_report_with_visible_charts(output.getvalue())

# Match the centered Bisection solver content width.
left_margin, main_area, right_margin = st.columns([0.035, 0.93, 0.035])
with main_area:
    st.markdown('<main class="solver-wrapper solver-streamlit-area">', unsafe_allow_html=True)
    
    guide_column, conditions_column = st.columns(2)

    with guide_column:
        with st.container(border=True):
            st.subheader('How to Write the Functions')
            st.markdown(
                """
            Enter each equation as an expression equal to zero, but type only the expression.

            - Use only **x** and **y** as variables.
            - Powers: write `x**2`, not **x^2**.
            - Multiplication: write `x*y`, not `xy`.
            - Use lowercase functions such as **sin(x)**, **cos(y)**, **exp(x)**, and **log(x)**.
            - Example pair: `x**2 + y**2 - 4` and `x - y`.
                """
            )

    with conditions_column:
        with st.container(border=True):
            st.subheader('Before Solving')
            st.markdown(
                """
            - Provide an initial guess reasonably close to the desired solution.
            - The Jacobian must be nonsingular at the iterates used by Newton’s method.
            - Tolerance must be positive and the maximum iteration count must be at least one.
            - Convergence is not guaranteed for every initial guess or nonlinear system.
            - Verify both the residual norm and the step norm before accepting the solution.

            **Newton update:** solve **J(xₖ)Δxₖ = −F(xₖ)**, then set **xₖ₊₁ = xₖ + Δxₖ**.
                """
            )

    input_column, result_column = st.columns([1.25, 1.0])
    
    with input_column:
        with st.container(border=True):
            st.subheader("Input")
            equation_1 = st.text_input(
                "First equation = 0",
                value="x**2 + y**2 - 4",
                key="nonlinear_system_eq1",
            )
            equation_2 = st.text_input(
                "Second equation = 0",
                value="x - y",
                key="nonlinear_system_eq2",
            )
            initial_columns = st.columns(2)
            with initial_columns[0]:
                x0 = st.number_input("Initial x", value=1.5, format="%.10g", key="nonlinear_system_x0")
            with initial_columns[1]:
                y0 = st.number_input("Initial y", value=1.0, format="%.10g", key="nonlinear_system_y0")
            control_columns = st.columns(2)
            with control_columns[0]:
                tolerance = st.number_input(
                    "Tolerance",
                    min_value=1.0e-14,
                    value=1.0e-8,
                    format="%.1e",
                    key="nonlinear_system_tolerance",
                )
            with control_columns[1]:
                max_iterations = st.number_input(
                    "Maximum iterations",
                    min_value=1,
                    max_value=200,
                    value=30,
                    step=1,
                    key="nonlinear_system_max_iterations",
                )
            solve_clicked = st.button(
                "Solve System",
                type="primary",
                use_container_width=True,
                key="nonlinear_system_solve",
            )
    
    current_signature = signature(
        equation_1,
        equation_2,
        x0,
        y0,
        tolerance,
        max_iterations,
    )
    
    if solve_clicked:
        try:
            st.session_state.nonlinear_system_result = solve_system(
                equation_1,
                equation_2,
                x0,
                y0,
                float(tolerance),
                int(max_iterations),
            )
        except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
            st.session_state.nonlinear_system_result = {"error": str(error)}
        st.session_state.nonlinear_system_signature = current_signature
        st.rerun()
    
    with result_column:
        with st.container(border=True):
            st.subheader("Final Result")
            stored = st.session_state.get("nonlinear_system_result")
            stored_signature = st.session_state.get("nonlinear_system_signature")
    
            if stored is None:
                st.info("Enter the equations and click Solve System.")
            elif stored_signature != current_signature:
                st.info("The inputs changed. Click Solve System to recalculate.")
            elif "error" in stored:
                st.error(stored["error"])
            else:
                if stored["success"]:
                    st.success(stored["message"])
                else:
                    st.warning(stored["message"])
                solution = stored["solution"]
                result_columns = st.columns(2)
                result_columns[0].metric("x", format_display_number(solution[0]))
                result_columns[1].metric("y", format_display_number(solution[1]))
                final_row = stored["history"].iloc[-1]
                st.metric("Final Residual ∞-Norm", format_display_number(final_row["Residual ∞-Norm"]))
    
    stored = st.session_state.get("nonlinear_system_result")
    if (
        stored
        and "error" not in stored
        and st.session_state.get("nonlinear_system_signature") == current_signature
    ):
        history = stored["history"]
    
        st.divider()
        st.subheader("Iteration History")
        st.dataframe(history.round(3), use_container_width=True, hide_index=True)
    
        with st.container(border=True):
            st.subheader("Parsed System and Jacobian")
            st.code(f"F1(x,y) = {stored['f1']}\nF2(x,y) = {stored['f2']}")
            st.code(f"J(x,y) = {stored['jacobian']}")
    
        st.subheader("Convergence Graph")
        figure, axis = plt.subplots(figsize=(10, 5.5))
        axis.semilogy(
            history["Iteration"],
            np.maximum(history["Residual ∞-Norm"], np.finfo(float).tiny),
            marker="o",
            label="Residual ∞-norm",
        )
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Residual norm")
        axis.grid(True, which="both")
        axis.legend()
        figure.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
        st.subheader("Excel Report")
        excel_report = create_excel_report(
            stored,
            equation_1,
            equation_2,
            float(x0),
            float(y0),
            float(tolerance),
            int(max_iterations),
        )
        st.download_button(
            "Download Complete Excel Report",
            data=excel_report,
            file_name="nonlinear_system_complete_report.xlsx",
            mime=EXCEL_MIME_TYPE,
            use_container_width=True,
            key="nonlinear_system_excel_report",
        )
    
        with st.container(border=True):
            st.subheader("Continue Learning")
            navigation_left, navigation_right = st.columns(2)
            with navigation_left:
                if st.button("Review Lesson", use_container_width=True, key="nonlinear_system_lesson_button"):
                    st.switch_page("pages/Systems_of_Nonlinear_Equations.py")
            with navigation_right:
                if st.button("Back to Solver Menu", use_container_width=True, key="nonlinear_system_menu_button"):
                    st.switch_page("pages/Numerical_Solver.py")
    
    st.markdown("</main>", unsafe_allow_html=True)

st.html(
    """
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>Systems of Nonlinear Equations • Newton Method</div>
    </footer>
    """
)
