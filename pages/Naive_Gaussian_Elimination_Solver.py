
import math
CONFIG = {'method_id': 'naive_gaussian',
 'title': 'Naive Gaussian Elimination Solver',
 'label': 'NAIVE GAUSSIAN ELIMINATION TOOL',
 'description': 'Enter a square linear system, perform forward elimination without pivoting, and inspect the '
                'triangular system and back-substitution result.',
 'lesson': 'Naive_Gaussian_Elimination',
 'quiz': 'Naive_Gaussian_Elimination_Quiz',
 'footer': 'Naive Gaussian Elimination Solver • Linear Systems',
 'conditions': ['The coefficient matrix must be square.',
                'Every diagonal pivot must be nonzero.',
                'Use a pivoted method when a pivot is zero, tiny, or the system is poorly scaled.'],
 'formula': 'm_{ik}=\\frac{a_{ik}}{a_{kk}},\\qquad R_i\\leftarrow R_i-m_{ik}R_k',
 'defaults': {'A': [[2.0, 1.0], [4.0, -6.0]], 'b': [5.0, -2.0]}}


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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


def augmented_text(matrix, vector):
    augmented = np.column_stack((matrix, vector))
    return np.array2string(
        augmented,
        precision=6,
        suppress_small=True,
        max_line_width=140,
    )


def solve_linear_system(matrix, vector):
    A = np.asarray(matrix, dtype=float).copy()
    b = np.asarray(vector, dtype=float).copy()

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return {"status": "error", "message": "A must be a square matrix."}

    n = A.shape[0]

    if b.shape != (n,):
        return {
            "status": "error",
            "message": "The right-hand side must contain one value per equation.",
        }

    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        return {"status": "error", "message": "All entries must be finite numbers."}

    original_A = A.copy()
    original_b = b.copy()
    history = []
    pivot_tolerance = 1e-12

    try:
        if CONFIG["method_id"] in {"naive_gaussian", "scaled_pivoting"}:
            scales = None

            if CONFIG["method_id"] == "scaled_pivoting":
                scales = np.max(np.abs(A), axis=1)

                if np.any(scales <= pivot_tolerance):
                    raise ValueError(
                        "At least one row has a zero scaling factor."
                    )

            for k in range(n - 1):
                if CONFIG["method_id"] == "scaled_pivoting":
                    ratios = np.abs(A[k:, k]) / scales[k:]
                    pivot_row = k + int(np.argmax(ratios))

                    if abs(A[pivot_row, k]) <= pivot_tolerance:
                        raise ValueError(
                            f"No usable pivot exists in column {k + 1}."
                        )

                    if pivot_row != k:
                        A[[k, pivot_row]] = A[[pivot_row, k]]
                        b[[k, pivot_row]] = b[[pivot_row, k]]
                        scales[[k, pivot_row]] = scales[[pivot_row, k]]

                        history.append(
                            {
                                "Step": len(history) + 1,
                                "Operation": (
                                    f"Swap row {k + 1} with row {pivot_row + 1}"
                                ),
                                "Pivot": A[k, k],
                                "Augmented Matrix": augmented_text(A, b),
                            }
                        )

                elif abs(A[k, k]) <= pivot_tolerance:
                    raise ValueError(
                        f"Zero or extremely small pivot at row {k + 1}. "
                        "Naive Gaussian Elimination cannot continue."
                    )

                for i in range(k + 1, n):
                    multiplier = A[i, k] / A[k, k]
                    A[i, k:] = A[i, k:] - multiplier * A[k, k:]
                    b[i] = b[i] - multiplier * b[k]
                    A[i, k] = 0.0

                    history.append(
                        {
                            "Step": len(history) + 1,
                            "Operation": (
                                f"R{i + 1} ← R{i + 1} − "
                                f"({multiplier:.6g})R{k + 1}"
                            ),
                            "Pivot": A[k, k],
                            "Augmented Matrix": augmented_text(A, b),
                        }
                    )

            if abs(A[-1, -1]) <= pivot_tolerance:
                raise ValueError(
                    "The final pivot is zero or extremely small."
                )

            solution = np.zeros(n)

            for i in range(n - 1, -1, -1):
                if abs(A[i, i]) <= pivot_tolerance:
                    raise ValueError(
                        f"Invalid pivot during back substitution at row {i + 1}."
                    )

                known_sum = np.dot(A[i, i + 1:], solution[i + 1:])
                solution[i] = (b[i] - known_sum) / A[i, i]

            final_matrix = A
            final_vector = b
            final_form_name = "Upper-Triangular Form"

        else:
            augmented = np.column_stack((A, b))

            for k in range(n):
                pivot_row = k + int(
                    np.argmax(np.abs(augmented[k:, k]))
                )

                if abs(augmented[pivot_row, k]) <= pivot_tolerance:
                    raise ValueError(
                        "The system does not have a unique solution."
                    )

                if pivot_row != k:
                    augmented[[k, pivot_row]] = augmented[[pivot_row, k]]
                    history.append(
                        {
                            "Step": len(history) + 1,
                            "Operation": (
                                f"Swap row {k + 1} with row {pivot_row + 1}"
                            ),
                            "Pivot": augmented[k, k],
                            "Augmented Matrix": np.array2string(
                                augmented,
                                precision=6,
                                suppress_small=True,
                                max_line_width=140,
                            ),
                        }
                    )

                pivot = augmented[k, k]
                augmented[k] = augmented[k] / pivot

                history.append(
                    {
                        "Step": len(history) + 1,
                        "Operation": f"Normalize row {k + 1}",
                        "Pivot": 1.0,
                        "Augmented Matrix": np.array2string(
                            augmented,
                            precision=6,
                            suppress_small=True,
                            max_line_width=140,
                        ),
                    }
                )

                for i in range(n):
                    if i == k:
                        continue

                    factor = augmented[i, k]

                    if abs(factor) <= pivot_tolerance:
                        augmented[i, k] = 0.0
                        continue

                    augmented[i] = augmented[i] - factor * augmented[k]
                    augmented[i, k] = 0.0

                    history.append(
                        {
                            "Step": len(history) + 1,
                            "Operation": (
                                f"R{i + 1} ← R{i + 1} − "
                                f"({factor:.6g})R{k + 1}"
                            ),
                            "Pivot": augmented[k, k],
                            "Augmented Matrix": np.array2string(
                                augmented,
                                precision=6,
                                suppress_small=True,
                                max_line_width=140,
                            ),
                        }
                    )

            solution = augmented[:, -1]
            final_matrix = augmented[:, :-1]
            final_vector = augmented[:, -1]
            final_form_name = "Reduced Row Echelon Form"

        residual_vector = original_A @ solution - original_b
        residual_norm = float(np.linalg.norm(residual_vector, ord=np.inf))
        determinant = float(np.linalg.det(original_A))

        return {
            "status": "success",
            "message": "The linear system was solved successfully.",
            "solution": solution,
            "history": history,
            "original_A": original_A,
            "original_b": original_b,
            "final_matrix": final_matrix,
            "final_vector": final_vector,
            "final_form_name": final_form_name,
            "residual_vector": residual_vector,
            "residual_norm": residual_norm,
            "determinant": determinant,
            "operations": len(history),
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error),
            "history": history,
        }


def create_default_table(size):
    default_A = np.eye(size) * 2.0
    default_b = np.ones(size)

    stored_A = np.asarray(CONFIG["defaults"]["A"], dtype=float)
    stored_b = np.asarray(CONFIG["defaults"]["b"], dtype=float)

    rows = min(size, stored_A.shape[0])
    columns = min(size, stored_A.shape[1])

    default_A[:rows, :columns] = stored_A[:rows, :columns]
    default_b[:rows] = stored_b[:rows]

    data = {
        f"x{column + 1}": default_A[:, column]
        for column in range(size)
    }
    data["b"] = default_b

    return pd.DataFrame(data)


def format_number(value):
    """Format final solver values consistently to three decimal places."""

    return format_display_number(value, unavailable="Not available")




def create_excel_report(result) -> bytes:
    """Create one organized workbook containing all linear-system results."""

    if result.get("status") != "success":
        raise ValueError("Only a successful calculation can be exported.")

    solution = np.asarray(result["solution"], dtype=float)
    residual = np.asarray(result["residual_vector"], dtype=float)
    size = len(solution)

    summary = pd.DataFrame(
        {
            "Property": [
                "Method",
                "Status",
                "Matrix Size",
                "Final Matrix Form",
                "Row Operations",
                "Determinant",
                "Residual Infinity Norm",
                "Solution Vector",
                "Message",
            ],
            "Value": [
                CONFIG["title"],
                result["status"].title(),
                f"{size} × {size}",
                result["final_form_name"],
                result["operations"],
                result["determinant"],
                result["residual_norm"],
                ", ".join(
                    f"x{index + 1} = {format_display_number(value)}"
                    for index, value in enumerate(solution)
                ),
                result["message"],
            ],
        }
    )

    input_system = pd.DataFrame(
        np.column_stack((result["original_A"], result["original_b"])),
        columns=[f"x{index + 1}" for index in range(size)] + ["b"],
        index=[f"Equation {index + 1}" for index in range(size)],
    ).reset_index(names="Equation")

    final_system = pd.DataFrame(
        np.column_stack((result["final_matrix"], result["final_vector"])),
        columns=[f"x{index + 1}" for index in range(size)] + ["Right-Hand Side"],
        index=[f"Row {index + 1}" for index in range(size)],
    ).reset_index(names="Row")

    solution_table = pd.DataFrame(
        {
            "Variable": [f"x{index + 1}" for index in range(size)],
            "Calculated Value": solution,
        }
    )
    residual_table = pd.DataFrame(
        {
            "Equation": [f"Equation {index + 1}" for index in range(size)],
            "Computed Ax": result["original_A"] @ solution,
            "Original b": result["original_b"],
            "Residual Ax − b": residual,
            "Absolute Residual": np.abs(residual),
        }
    )
    operations_table = pd.DataFrame(result["history"])
    if operations_table.empty:
        operations_table = pd.DataFrame(
            [{"Step": 0, "Operation": "No row operation was required."}]
        )

    chart_data = pd.DataFrame(
        {
            "Variable": [f"x{index + 1}" for index in range(size)],
            "Solution": solution,
            "Equation": [f"Equation {index + 1}" for index in range(size)],
            "Absolute Residual": np.abs(residual),
        }
    )

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        input_system.to_excel(writer, sheet_name="Input System", index=False)
        final_system.to_excel(writer, sheet_name="Final System", index=False)
        solution_table.to_excel(writer, sheet_name="Solution", index=False)
        residual_table.to_excel(writer, sheet_name="Residual Analysis", index=False)
        operations_table.to_excel(writer, sheet_name="Row Operations", index=False)
        chart_data.to_excel(writer, sheet_name="Chart Data", index=False)

        workbook = writer.book
        data_sheet = workbook["Chart Data"]
        data_sheet["F1"] = "Graphs based on the report data"
        data_sheet["F1"].font = Font(bold=True, size=14)

        solution_chart = BarChart()
        solution_chart.title = "Final Solution Values"
        solution_chart.y_axis.title = "Value"
        solution_chart.x_axis.title = "Variable"
        solution_chart.height = 8
        solution_chart.width = 15
        solution_chart.add_data(
            Reference(data_sheet, min_col=2, min_row=1, max_row=size + 1),
            titles_from_data=True,
        )
        solution_chart.set_categories(
            Reference(data_sheet, min_col=1, min_row=2, max_row=size + 1)
        )
        data_sheet.add_chart(solution_chart, "F3")

        residual_chart = BarChart()
        residual_chart.title = "Absolute Residual by Equation"
        residual_chart.y_axis.title = "Absolute residual"
        residual_chart.x_axis.title = "Equation"
        residual_chart.height = 8
        residual_chart.width = 15
        residual_chart.add_data(
            Reference(data_sheet, min_col=4, min_row=1, max_row=size + 1),
            titles_from_data=True,
        )
        residual_chart.set_categories(
            Reference(data_sheet, min_col=3, min_row=2, max_row=size + 1)
        )
        data_sheet.add_chart(residual_chart, "F20")

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
            st.subheader("How to Enter the System")
            st.markdown(
                """
                Choose the number of equations, then enter the coefficients of
                **A** and the right-hand side **b** in the table.

                Each row represents one equation and each `x` column represents
                one unknown coefficient.
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
                "After solving, inspect the residual and operation history rather than relying only on the displayed solution."
            )

    input_column, result_column = st.columns(2)

    with input_column:
        with st.container(border=True):
            st.subheader("Input")

            size = st.number_input(
                "Number of equations",
                min_value=2,
                max_value=6,
                value=2,
                step=1,
                key=f"{CONFIG['method_id']}_size",
            )

            input_table = st.data_editor(
                create_default_table(int(size)),
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key=f"{CONFIG['method_id']}_table_{int(size)}",
            )

            st.caption(
                "Columns x1 … xn form the coefficient matrix; the final column is b."
            )

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
                    "Reset Result",
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
                try:
                    matrix = input_table[
                        [f"x{i + 1}" for i in range(int(size))]
                    ].to_numpy(dtype=float)
                    vector = input_table["b"].to_numpy(dtype=float)

                    st.session_state[
                        f"{CONFIG['method_id']}_result"
                    ] = solve_linear_system(matrix, vector)

                except Exception as error:
                    st.session_state[
                        f"{CONFIG['method_id']}_result"
                    ] = {
                        "status": "error",
                        "message": f"Invalid table values: {error}",
                    }

                st.rerun()

    with result_column:
        with st.container(border=True):
            st.subheader("Final Result")

            result = st.session_state.get(
                f"{CONFIG['method_id']}_result"
            )

            if result is None:
                st.info("Enter the system and click Solve to display the result.")

            elif result["status"] == "error":
                st.error(result["message"])

            else:
                st.success(result["message"])

                metric_column1, metric_column2 = st.columns(2)

                with metric_column1:
                    st.metric(
                        "Residual Norm",
                        format_number(result["residual_norm"]),
                    )

                with metric_column2:
                    st.metric(
                        "Row Operations",
                        result["operations"],
                    )

                metric_column3, metric_column4 = st.columns(2)

                with metric_column3:
                    st.metric(
                        "System Size",
                        len(result["solution"]),
                    )

                with metric_column4:
                    st.metric(
                        "Determinant",
                        format_number(result["determinant"]),
                    )

                solution_frame = pd.DataFrame(
                    {
                        "Variable": [
                            f"x{i + 1}"
                            for i in range(len(result["solution"]))
                        ],
                        "Value": result["solution"],
                    }
                )

                st.markdown("**Solution Vector**")
                st.dataframe(
                    solution_frame.round(3),
                    use_container_width=True,
                    hide_index=True,
                )


    result = st.session_state.get(f"{CONFIG['method_id']}_result")

    if result and result["status"] == "success":
        with st.container(border=True):
            st.subheader(result["final_form_name"])
            st.code(
                augmented_text(
                    result["final_matrix"],
                    result["final_vector"],
                ),
                language="text",
            )

        with st.container(border=True):
            st.subheader("Operation History")

            if not result["history"]:
                st.info("No row operations were required.")
            else:
                history_frame = pd.DataFrame(
                    [
                        {
                            "Step": item["Step"],
                            "Operation": item["Operation"],
                            "Pivot": item["Pivot"],
                        }
                        for item in result["history"]
                    ]
                )

                st.dataframe(
                    history_frame,
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("View Intermediate Matrices", expanded=False):
                    for item in result["history"]:
                        st.markdown(
                            f"**Step {item['Step']}: {item['Operation']}**"
                        )
                        st.code(
                            item["Augmented Matrix"],
                            language="text",
                        )

                solution_download = pd.DataFrame(
                    {
                        "Variable": [
                            f"x{i + 1}"
                            for i in range(len(result["solution"]))
                        ],
                        "Value": result["solution"],
                    }
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

        matrix_column, solution_column, residual_column = st.columns(3)

        with matrix_column:
            with st.container(border=True):
                st.subheader("Coefficient Matrix")
                figure, axis = plt.subplots(figsize=(4.2, 3.0))
                image = axis.imshow(result["original_A"], aspect="auto")
                axis.set_xlabel("Column")
                axis.set_ylabel("Row")
                axis.set_xticks(range(result["original_A"].shape[1]))
                axis.set_yticks(range(result["original_A"].shape[0]))
                st.pyplot(figure, use_container_width=True)
                plt.close(figure)

        with solution_column:
            with st.container(border=True):
                st.subheader("Solution Values")
                figure, axis = plt.subplots(figsize=(4.2, 3.0))
                positions = np.arange(len(result["solution"]))
                axis.bar(positions, result["solution"])
                axis.set_xticks(
                    positions,
                    [f"x{i + 1}" for i in positions],
                )
                axis.set_ylabel("Value")
                axis.grid(True, axis="y")
                st.pyplot(figure, use_container_width=True)
                plt.close(figure)

        with residual_column:
            with st.container(border=True):
                st.subheader("Residual Components")
                figure, axis = plt.subplots(figsize=(4.2, 3.0))
                positions = np.arange(len(result["residual_vector"]))
                axis.bar(positions, np.abs(result["residual_vector"]))
                axis.set_xticks(
                    positions,
                    [f"Eq. {i + 1}" for i in positions],
                )
                axis.set_ylabel("|Ax − b|")
                axis.grid(True, axis="y")
                st.pyplot(figure, use_container_width=True)
                plt.close(figure)

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
