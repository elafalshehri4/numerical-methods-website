from __future__ import annotations

import base64
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
from openpyxl.chart import LineChart, Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational

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

# =============================================================================
# Constants
# =============================================================================
METHOD_NAME = "Richardson Extrapolation"
DISPLAY_DECIMALS = 3
DEFAULT_FUNCTION = "sin(x)"
DEFAULT_X_VALUE = 1.0
DEFAULT_STEP_SIZE = 0.2
DEFAULT_LEVELS = 5
MIN_LEVELS = 2
MAX_LEVELS = 9
ZERO_TOLERANCE = 1.0e-15
RELATIVE_ERROR_DENOMINATOR_TOLERANCE = 1.0e-15
REPORT_TIME_ZONE = "Asia/Riyadh"
EXCEL_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

BASE_METHOD_OPTIONS = {
    "Central difference — base error O(h²)": "central",
    "Forward difference — base error O(h)": "forward",
}

ALLOWED_FUNCTION_NAMES = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
}


# =============================================================================
# Structured data models
# =============================================================================
@dataclass(frozen=True)
class BaseApproximation:
    """One finite-difference approximation used by Richardson extrapolation."""

    level: int
    step_size: float
    x_value: float
    x_minus_h: float | None
    f_x_minus_h: float | None
    f_x: float
    x_plus_h: float
    f_x_plus_h: float
    numerator: float
    denominator: float
    approximation: float
    exact_derivative: float | None
    absolute_error: float | None
    relative_error: float | None
    substitution_text: str


@dataclass(frozen=True)
class ExtrapolationStep:
    """One computed cell in the triangular Richardson table."""

    row: int
    column: int
    step_size: float
    lower_order_estimate: float
    previous_row_estimate: float
    refinement_ratio: float
    base_order: int
    exponent: int
    denominator_factor: float
    correction: float
    extrapolated_estimate: float
    expected_order: int
    exact_derivative: float | None
    absolute_error: float | None
    relative_error: float | None
    formula_text: str
    substitution_text: str


@dataclass(frozen=True)
class RichardsonResult:
    """Complete solver result shared by Streamlit and Excel renderers."""

    status: str
    success: bool
    method: str
    message: str
    stopping_reason: str
    function_text: str
    function_expression: sp.Expr | None
    derivative_expression: sp.Expr | None
    derivative_expression_text: str
    base_method_key: str
    base_method_name: str
    base_formula_text: str
    base_order: int
    x_value: float | None
    initial_step_size: float | None
    refinement_ratio: float
    levels: int
    base_approximations: tuple[BaseApproximation, ...]
    extrapolation_steps: tuple[ExtrapolationStep, ...]
    richardson_table: tuple[tuple[float | None, ...], ...]
    exact_derivative: float | None
    primary_base_estimate: float | None
    final_estimate: float | None
    final_expected_order: int | None
    primary_absolute_error: float | None
    final_absolute_error: float | None
    final_relative_error: float | None
    error_improvement_factor: float | None
    latest_observed_order: float | None
    warnings: tuple[str, ...]
    input_signature: str
    execution_datetime: datetime


# =============================================================================
# General helpers
# =============================================================================
def image_to_base64(image_path: str | Path) -> str:
    """Return a file as Base64 text, or an empty string if unavailable."""

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def current_report_datetime() -> datetime:
    """Return a timezone-aware report timestamp."""

    return datetime.now(ZoneInfo(REPORT_TIME_ZONE))


def format_number(
    value: float | int | None,
    decimals: int = 3,
) -> str:
    """Format displayed values with three decimals and × 10ⁿ notation."""

    return format_display_number(value, decimals)



def scientific_number(value: float | int | None) -> str:
    """Format a value as a three-decimal coefficient multiplied by 10ⁿ."""

    return format_scientific_power(value)



def round_numeric_dataframe(
    dataframe: pd.DataFrame,
    decimals: int = DISPLAY_DECIMALS,
) -> pd.DataFrame:
    """Round numeric columns only for display."""

    rounded = dataframe.copy()
    numeric_columns = rounded.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        rounded[numeric_columns] = rounded[numeric_columns].round(decimals)
    return rounded


def create_input_signature(
    function_text: str,
    x_value: Any,
    step_size: Any,
    base_method_name: str,
    levels: Any,
) -> str:
    """Create a stable signature used to prevent stale Streamlit results."""

    payload = repr(
        (
            str(function_text).strip(),
            str(x_value),
            str(step_size),
            str(base_method_name),
            str(levels),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_float(raw_value: Any, value_name: str) -> float:
    """Convert one user input to a finite real floating-point value."""

    if raw_value is None:
        raise ValueError(f"{value_name} is required.")
    if isinstance(raw_value, str) and not raw_value.strip():
        raise ValueError(f"{value_name} is required.")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{value_name} must be a valid numerical value.") from error
    if not math.isfinite(value):
        raise ValueError(
            f"{value_name} must be finite; NaN and infinity are not allowed."
        )
    return value


def safe_relative_error(absolute_error: float, exact_value: float) -> float | None:
    """Calculate relative error safely when the exact derivative is nonzero."""

    if abs(exact_value) <= RELATIVE_ERROR_DENOMINATOR_TOLERANCE:
        return None
    return absolute_error / abs(exact_value)


def human_readable_expression(expression: sp.Expr | None) -> str:
    """Return a compact readable mathematical expression."""

    if expression is None:
        return "Not available"
    return str(sp.simplify(expression))


# =============================================================================
# Function parsing and safe evaluation
# =============================================================================
def parse_function(function_text: str) -> tuple[sp.Expr, sp.Symbol]:
    """Parse a safe, real, single-variable mathematical expression."""

    if not isinstance(function_text, str) or not function_text.strip():
        raise ValueError("Function f(x) is required.")

    x_symbol = sp.Symbol("x", real=True)
    local_dictionary = {"x": x_symbol, **ALLOWED_FUNCTION_NAMES}

    try:
        expression = sp.sympify(
            function_text.strip(),
            locals=local_dictionary,
            evaluate=True,
        )
    except (sp.SympifyError, SyntaxError, TypeError, ValueError) as error:
        raise ValueError(
            "The function format is invalid. Use valid Python/SymPy syntax, "
            "for example sin(x), exp(x), or x**3 - 2*x + 1."
        ) from error

    if isinstance(expression, Relational) or expression.is_Matrix:
        raise ValueError("Enter a scalar function expression, not an equation or matrix.")
    if expression.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        raise ValueError("The function contains an undefined or infinite constant.")
    if expression.atoms(AppliedUndef):
        raise ValueError("The function contains an unsupported function name.")

    unknown_symbols = expression.free_symbols - {x_symbol}
    if unknown_symbols:
        names = ", ".join(sorted(str(symbol) for symbol in unknown_symbols))
        raise ValueError(
            "Only the variable x is supported. Remove the following symbols: "
            f"{names}."
        )

    return sp.simplify(expression), x_symbol


def create_numeric_function(
    expression: sp.Expr,
    x_symbol: sp.Symbol,
) -> Callable[[Any], Any]:
    """Convert one SymPy expression to a NumPy-compatible function."""

    try:
        return sp.lambdify(x_symbol, expression, modules=["numpy"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The function could not be converted to a numerical function."
        ) from error


def evaluate_real_scalar(
    numeric_function: Callable[[Any], Any],
    x_value: float,
    value_name: str,
) -> float:
    """Evaluate a numerical function and require one finite real scalar."""

    try:
        with np.errstate(all="ignore"):
            raw_value = numeric_function(x_value)
    except (ArithmeticError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"The function could not be evaluated for {value_name} at "
            f"x = {format_number(x_value, 12)}."
        ) from error

    array = np.asarray(raw_value)
    if array.size != 1:
        raise ValueError(f"{value_name} did not produce a scalar value.")

    scalar = array.reshape(-1)[0]
    if np.iscomplexobj(scalar):
        complex_value = complex(scalar)
        if abs(complex_value.imag) > 1.0e-12:
            raise ValueError(
                f"{value_name} is complex at x = {format_number(x_value, 12)}."
            )
        scalar = complex_value.real

    try:
        value = float(scalar)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{value_name} is not a real numerical value.") from error

    if not math.isfinite(value):
        raise ValueError(
            f"{value_name} is undefined, NaN, or infinite at "
            f"x = {format_number(x_value, 12)}."
        )
    return value


def evaluate_real_array(
    numeric_function: Callable[[Any], Any],
    x_values: np.ndarray,
) -> np.ndarray:
    """Evaluate a function over an array for plotting, replacing invalid values."""

    try:
        with np.errstate(all="ignore"):
            raw_values = numeric_function(x_values)
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return np.full_like(x_values, np.nan, dtype=float)

    array = np.asarray(raw_values)
    if array.ndim == 0:
        array = np.full_like(x_values, array, dtype=np.complex128)
    try:
        array = np.broadcast_to(array, x_values.shape)
    except ValueError:
        return np.full_like(x_values, np.nan, dtype=float)

    if np.iscomplexobj(array):
        imaginary_part = np.abs(np.imag(array))
        real_values = np.real(array).astype(float)
        real_values[imaginary_part > 1.0e-12] = np.nan
    else:
        try:
            real_values = array.astype(float)
        except (TypeError, ValueError):
            return np.full_like(x_values, np.nan, dtype=float)

    real_values[~np.isfinite(real_values)] = np.nan
    return real_values


# =============================================================================
# Richardson Extrapolation algorithm
# =============================================================================
def base_method_metadata(method_key: str) -> tuple[str, int]:
    """Return the base finite-difference formula and its leading error order."""

    if method_key == "central":
        return (
            "D(h) = [f(x₀ + h) − f(x₀ − h)] / (2h)",
            2,
        )
    if method_key == "forward":
        return (
            "D(h) = [f(x₀ + h) − f(x₀)] / h",
            1,
        )
    raise ValueError("Unsupported base finite-difference method.")


def calculate_base_approximation(
    numeric_function: Callable[[Any], Any],
    x_value: float,
    step_size: float,
    method_key: str,
) -> dict[str, float | None | str]:
    """Calculate one base finite-difference derivative approximation manually."""

    x_plus_h = x_value + step_size
    if not math.isfinite(x_plus_h):
        raise ValueError("x₀ + h is not finite. Reduce the magnitude of x₀ or h.")

    f_x = evaluate_real_scalar(numeric_function, x_value, "f(x₀)")
    f_x_plus_h = evaluate_real_scalar(
        numeric_function,
        x_plus_h,
        "f(x₀ + h)",
    )

    if method_key == "forward":
        x_minus_h = None
        f_x_minus_h = None
        numerator = f_x_plus_h - f_x
        denominator = step_size
        approximation = numerator / denominator
        substitution_text = (
            f"[{format_number(f_x_plus_h, 12)} − "
            f"{format_number(f_x, 12)}] / "
            f"{format_number(step_size, 12)}"
        )
    elif method_key == "central":
        x_minus_h = x_value - step_size
        if not math.isfinite(x_minus_h):
            raise ValueError("x₀ − h is not finite. Reduce the magnitude of x₀ or h.")
        f_x_minus_h = evaluate_real_scalar(
            numeric_function,
            x_minus_h,
            "f(x₀ − h)",
        )
        numerator = f_x_plus_h - f_x_minus_h
        denominator = 2.0 * step_size
        approximation = numerator / denominator
        substitution_text = (
            f"[{format_number(f_x_plus_h, 12)} − "
            f"{format_number(f_x_minus_h, 12)}] / "
            f"[2({format_number(step_size, 12)})]"
        )
    else:
        raise ValueError("Unsupported base finite-difference method.")

    values_to_check = [numerator, denominator, approximation]
    if not all(math.isfinite(float(value)) for value in values_to_check):
        raise ValueError("The base finite-difference calculation is non-finite.")

    return {
        "x_minus_h": x_minus_h,
        "f_x_minus_h": f_x_minus_h,
        "f_x": f_x,
        "x_plus_h": x_plus_h,
        "f_x_plus_h": f_x_plus_h,
        "numerator": numerator,
        "denominator": denominator,
        "approximation": approximation,
        "substitution_text": substitution_text,
    }


def calculate_observed_order(
    previous_error: float | None,
    current_error: float | None,
    refinement_ratio: float = 2.0,
) -> float | None:
    """Estimate convergence order from consecutive positive exact errors."""

    if previous_error is None or current_error is None:
        return None
    if previous_error <= 0.0 or current_error <= 0.0:
        return None
    try:
        order = math.log(previous_error / current_error) / math.log(refinement_ratio)
    except (ValueError, ZeroDivisionError):
        return None
    return order if math.isfinite(order) else None


def build_richardson_table(
    base_values: Sequence[float],
    step_sizes: Sequence[float],
    base_order: int,
    exact_derivative: float | None,
    refinement_ratio: float = 2.0,
) -> tuple[list[list[float | None]], list[ExtrapolationStep]]:
    """Build the triangular Richardson extrapolation table manually.

    The recursion is

        R[i, j] = R[i, j-1]
                  + (R[i, j-1] - R[i-1, j-1])
                    / (r**(p*j) - 1)

    where ``r`` is the refinement ratio and ``p`` is the leading error order of
    the base finite-difference method.
    """

    levels = len(base_values)
    table: list[list[float | None]] = [
        [None for _ in range(levels)] for _ in range(levels)
    ]
    steps: list[ExtrapolationStep] = []

    for row, value in enumerate(base_values):
        table[row][0] = float(value)

    for row in range(1, levels):
        for column in range(1, row + 1):
            current_lower = table[row][column - 1]
            previous_lower = table[row - 1][column - 1]
            if current_lower is None or previous_lower is None:
                raise ValueError("The Richardson table contains a missing prerequisite.")

            exponent = base_order * column
            denominator_factor = refinement_ratio**exponent - 1.0
            if abs(denominator_factor) <= ZERO_TOLERANCE:
                raise ValueError("The Richardson extrapolation denominator is zero.")

            correction = (current_lower - previous_lower) / denominator_factor
            extrapolated = current_lower + correction

            if not all(
                math.isfinite(value)
                for value in (denominator_factor, correction, extrapolated)
            ):
                raise ValueError(
                    "Richardson extrapolation produced a non-finite numerical value."
                )

            table[row][column] = extrapolated
            absolute_error = (
                None
                if exact_derivative is None
                else abs(extrapolated - exact_derivative)
            )
            relative_error = (
                None
                if absolute_error is None or exact_derivative is None
                else safe_relative_error(absolute_error, exact_derivative)
            )
            expected_order = base_order * (column + 1)
            formula_text = (
                f"R[{row},{column}] = R[{row},{column - 1}] + "
                f"(R[{row},{column - 1}] − R[{row - 1},{column - 1}]) / "
                f"({refinement_ratio:g}^{exponent} − 1)"
            )
            substitution_text = (
                f"{format_number(current_lower, 12)} + "
                f"({format_number(current_lower, 12)} − "
                f"{format_number(previous_lower, 12)}) / "
                f"({format_number(refinement_ratio, 6)}^{exponent} − 1) "
                f"= {format_number(extrapolated, 12)}"
            )

            steps.append(
                ExtrapolationStep(
                    row=row,
                    column=column,
                    step_size=float(step_sizes[row]),
                    lower_order_estimate=float(current_lower),
                    previous_row_estimate=float(previous_lower),
                    refinement_ratio=float(refinement_ratio),
                    base_order=base_order,
                    exponent=exponent,
                    denominator_factor=float(denominator_factor),
                    correction=float(correction),
                    extrapolated_estimate=float(extrapolated),
                    expected_order=expected_order,
                    exact_derivative=exact_derivative,
                    absolute_error=absolute_error,
                    relative_error=relative_error,
                    formula_text=formula_text,
                    substitution_text=substitution_text,
                )
            )

    return table, steps


def solve_richardson_extrapolation(
    function_text: str,
    raw_x_value: Any,
    raw_step_size: Any,
    base_method_name: str,
    raw_levels: Any,
    input_signature: str,
) -> RichardsonResult:
    """Validate inputs and perform the complete Richardson workflow."""

    execution_datetime = current_report_datetime()
    refinement_ratio = 2.0

    try:
        x_value = safe_float(raw_x_value, "Evaluation point x₀")
        initial_step_size = safe_float(raw_step_size, "Initial step size h")
        if initial_step_size <= 0.0:
            raise ValueError("Initial step size h must be greater than zero.")
        if initial_step_size <= ZERO_TOLERANCE:
            raise ValueError(
                "Initial step size h is too close to machine precision. "
                "Choose a larger value."
            )

        try:
            levels = int(raw_levels)
        except (TypeError, ValueError) as error:
            raise ValueError("The number of Richardson levels must be an integer.") from error
        if not MIN_LEVELS <= levels <= MAX_LEVELS:
            raise ValueError(
                f"Richardson levels must be between {MIN_LEVELS} and {MAX_LEVELS}."
            )

        base_method_key = BASE_METHOD_OPTIONS.get(base_method_name)
        if base_method_key is None:
            raise ValueError("Select a valid base finite-difference method.")
        base_formula_text, base_order = base_method_metadata(base_method_key)

        expression, x_symbol = parse_function(function_text)
        numeric_function = create_numeric_function(expression, x_symbol)

        derivative_expression: sp.Expr | None
        exact_derivative: float | None
        derivative_warning: str | None = None
        try:
            derivative_expression = sp.simplify(sp.diff(expression, x_symbol))
            derivative_numeric = create_numeric_function(
                derivative_expression,
                x_symbol,
            )
            exact_derivative = evaluate_real_scalar(
                derivative_numeric,
                x_value,
                "the exact derivative f′(x₀)",
            )
        except (ValueError, TypeError, NotImplementedError) as error:
            derivative_expression = None
            exact_derivative = None
            derivative_warning = (
                "The exact symbolic derivative could not be evaluated at x₀. "
                "The Richardson estimate remains available, but exact-error "
                f"metrics are omitted. Details: {error}"
            )

        base_approximations: list[BaseApproximation] = []
        base_values: list[float] = []
        step_sizes: list[float] = []

        for level in range(levels):
            step_size = initial_step_size / (refinement_ratio**level)
            if step_size <= ZERO_TOLERANCE or not math.isfinite(step_size):
                raise ValueError(
                    "Step-size refinement became numerically unusable. "
                    "Reduce the number of Richardson levels."
                )

            calculation = calculate_base_approximation(
                numeric_function=numeric_function,
                x_value=x_value,
                step_size=step_size,
                method_key=base_method_key,
            )
            approximation = float(calculation["approximation"])
            absolute_error = (
                None
                if exact_derivative is None
                else abs(approximation - exact_derivative)
            )
            relative_error = (
                None
                if absolute_error is None or exact_derivative is None
                else safe_relative_error(absolute_error, exact_derivative)
            )

            base_approximations.append(
                BaseApproximation(
                    level=level,
                    step_size=step_size,
                    x_value=x_value,
                    x_minus_h=(
                        None
                        if calculation["x_minus_h"] is None
                        else float(calculation["x_minus_h"])
                    ),
                    f_x_minus_h=(
                        None
                        if calculation["f_x_minus_h"] is None
                        else float(calculation["f_x_minus_h"])
                    ),
                    f_x=float(calculation["f_x"]),
                    x_plus_h=float(calculation["x_plus_h"]),
                    f_x_plus_h=float(calculation["f_x_plus_h"]),
                    numerator=float(calculation["numerator"]),
                    denominator=float(calculation["denominator"]),
                    approximation=approximation,
                    exact_derivative=exact_derivative,
                    absolute_error=absolute_error,
                    relative_error=relative_error,
                    substitution_text=str(calculation["substitution_text"]),
                )
            )
            base_values.append(approximation)
            step_sizes.append(step_size)

        table, extrapolation_steps = build_richardson_table(
            base_values=base_values,
            step_sizes=step_sizes,
            base_order=base_order,
            exact_derivative=exact_derivative,
            refinement_ratio=refinement_ratio,
        )

        final_estimate = table[-1][-1]
        if final_estimate is None:
            raise ValueError("The final Richardson estimate could not be created.")

        primary_absolute_error = base_approximations[0].absolute_error
        final_absolute_error = (
            None
            if exact_derivative is None
            else abs(final_estimate - exact_derivative)
        )
        final_relative_error = (
            None
            if final_absolute_error is None or exact_derivative is None
            else safe_relative_error(final_absolute_error, exact_derivative)
        )

        if (
            primary_absolute_error is not None
            and final_absolute_error is not None
            and final_absolute_error > 0.0
        ):
            error_improvement_factor = primary_absolute_error / final_absolute_error
        elif primary_absolute_error is not None and final_absolute_error == 0.0:
            error_improvement_factor = math.inf
        else:
            error_improvement_factor = None

        diagonal_errors: list[float | None] = []
        for row in range(levels):
            diagonal_value = table[row][row]
            diagonal_errors.append(
                None
                if exact_derivative is None or diagonal_value is None
                else abs(diagonal_value - exact_derivative)
            )

        observed_orders: list[float] = []
        for index in range(1, len(diagonal_errors)):
            observed_order = calculate_observed_order(
                diagonal_errors[index - 1],
                diagonal_errors[index],
                refinement_ratio,
            )
            if observed_order is not None:
                observed_orders.append(observed_order)
        latest_observed_order = observed_orders[-1] if observed_orders else None

        warnings: list[str] = []
        if derivative_warning:
            warnings.append(derivative_warning)

        finest_step = step_sizes[-1]
        cancellation_scale = np.sqrt(np.finfo(float).eps) * max(1.0, abs(x_value))
        if finest_step < cancellation_scale:
            warnings.append(
                "The finest step is very small relative to x₀. Floating-point "
                "round-off and subtractive cancellation may dominate the table."
            )

        if (
            primary_absolute_error is not None
            and final_absolute_error is not None
            and final_absolute_error > primary_absolute_error
        ):
            warnings.append(
                "The final Richardson estimate is less accurate than the initial "
                "base estimate for this input. Reduce the number of levels or use "
                "a larger initial h because round-off or non-asymptotic behavior "
                "may be dominating."
            )

        diagonal_values = [table[row][row] for row in range(levels)]
        if len(diagonal_values) >= 3:
            final_change = abs(float(diagonal_values[-1]) - float(diagonal_values[-2]))
            previous_change = abs(float(diagonal_values[-2]) - float(diagonal_values[-3]))
            if final_change > previous_change and previous_change > 0.0:
                warnings.append(
                    "The diagonal Richardson estimates are no longer improving "
                    "monotonically. The final levels may be affected by round-off."
                )

        final_expected_order = base_order * levels

        return RichardsonResult(
            status="success",
            success=True,
            method=METHOD_NAME,
            message="Execution completed successfully.",
            stopping_reason=(
                "The requested base estimates and complete Richardson "
                "extrapolation table were calculated."
            ),
            function_text=function_text.strip(),
            function_expression=expression,
            derivative_expression=derivative_expression,
            derivative_expression_text=human_readable_expression(
                derivative_expression
            ),
            base_method_key=base_method_key,
            base_method_name=base_method_name,
            base_formula_text=base_formula_text,
            base_order=base_order,
            x_value=x_value,
            initial_step_size=initial_step_size,
            refinement_ratio=refinement_ratio,
            levels=levels,
            base_approximations=tuple(base_approximations),
            extrapolation_steps=tuple(extrapolation_steps),
            richardson_table=tuple(tuple(row) for row in table),
            exact_derivative=exact_derivative,
            primary_base_estimate=base_values[0],
            final_estimate=float(final_estimate),
            final_expected_order=final_expected_order,
            primary_absolute_error=primary_absolute_error,
            final_absolute_error=final_absolute_error,
            final_relative_error=final_relative_error,
            error_improvement_factor=error_improvement_factor,
            latest_observed_order=latest_observed_order,
            warnings=tuple(warnings),
            input_signature=input_signature,
            execution_datetime=execution_datetime,
        )

    except ValueError as error:
        base_method_key = BASE_METHOD_OPTIONS.get(base_method_name, "central")
        base_formula_text, base_order = base_method_metadata(base_method_key)
        return RichardsonResult(
            status="error",
            success=False,
            method=METHOD_NAME,
            message=str(error),
            stopping_reason=(
                "The calculation stopped during input validation, function "
                "evaluation, or Richardson table construction."
            ),
            function_text=str(function_text).strip(),
            function_expression=None,
            derivative_expression=None,
            derivative_expression_text="Not available",
            base_method_key=base_method_key,
            base_method_name=base_method_name,
            base_formula_text=base_formula_text,
            base_order=base_order,
            x_value=None,
            initial_step_size=None,
            refinement_ratio=refinement_ratio,
            levels=0,
            base_approximations=tuple(),
            extrapolation_steps=tuple(),
            richardson_table=tuple(),
            exact_derivative=None,
            primary_base_estimate=None,
            final_estimate=None,
            final_expected_order=None,
            primary_absolute_error=None,
            final_absolute_error=None,
            final_relative_error=None,
            error_improvement_factor=None,
            latest_observed_order=None,
            warnings=tuple(),
            input_signature=input_signature,
            execution_datetime=execution_datetime,
        )


# =============================================================================
# DataFrame builders
# =============================================================================
def create_base_approximation_dataframe(result: RichardsonResult) -> pd.DataFrame:
    """Build the base finite-difference approximation table."""

    rows: list[dict[str, Any]] = []
    for item in result.base_approximations:
        rows.append(
            {
                "Level": item.level,
                "h": item.step_size,
                "x0 - h": item.x_minus_h,
                "f(x0 - h)": item.f_x_minus_h,
                "x0": item.x_value,
                "f(x0)": item.f_x,
                "x0 + h": item.x_plus_h,
                "f(x0 + h)": item.f_x_plus_h,
                "Numerator": item.numerator,
                "Denominator": item.denominator,
                "Base Approximation R[i,0]": item.approximation,
                "Exact Derivative": item.exact_derivative,
                "Absolute Error": item.absolute_error,
                "Relative Error": item.relative_error,
            }
        )
    return pd.DataFrame(rows)


def create_richardson_table_dataframe(result: RichardsonResult) -> pd.DataFrame:
    """Build a rectangular DataFrame from the triangular Richardson table."""

    columns = [
        "h",
        *[
            f"R(i,{column}) — O(h^{result.base_order * (column + 1)})"
            for column in range(result.levels)
        ],
    ]
    rows: list[list[Any]] = []
    for row_index, table_row in enumerate(result.richardson_table):
        rows.append(
            [result.base_approximations[row_index].step_size, *table_row]
        )
    return pd.DataFrame(rows, columns=columns)


def create_extrapolation_steps_dataframe(result: RichardsonResult) -> pd.DataFrame:
    """Build the complete per-cell extrapolation history table."""

    rows: list[dict[str, Any]] = []
    for item in result.extrapolation_steps:
        rows.append(
            {
                "Row i": item.row,
                "Column j": item.column,
                "h_i": item.step_size,
                "R(i,j-1)": item.lower_order_estimate,
                "R(i-1,j-1)": item.previous_row_estimate,
                "Refinement Ratio": item.refinement_ratio,
                "Exponent p*j": item.exponent,
                "Denominator r^(p*j)-1": item.denominator_factor,
                "Correction": item.correction,
                "R(i,j)": item.extrapolated_estimate,
                "Expected Order": item.expected_order,
                "Exact Derivative": item.exact_derivative,
                "Absolute Error": item.absolute_error,
                "Relative Error": item.relative_error,
                "Formula": item.formula_text,
                "Substitution": item.substitution_text,
            }
        )
    return pd.DataFrame(rows)


def create_diagonal_analysis_dataframe(result: RichardsonResult) -> pd.DataFrame:
    """Build convergence analysis for the diagonal Richardson estimates."""

    rows: list[dict[str, Any]] = []
    previous_error: float | None = None
    previous_estimate: float | None = None

    for row in range(result.levels):
        estimate = result.richardson_table[row][row]
        if estimate is None:
            continue
        absolute_error = (
            None
            if result.exact_derivative is None
            else abs(float(estimate) - result.exact_derivative)
        )
        relative_error = (
            None
            if absolute_error is None or result.exact_derivative is None
            else safe_relative_error(absolute_error, result.exact_derivative)
        )
        successive_difference = (
            None
            if previous_estimate is None
            else abs(float(estimate) - previous_estimate)
        )
        observed_order = calculate_observed_order(
            previous_error,
            absolute_error,
            result.refinement_ratio,
        )

        rows.append(
            {
                "Level": row,
                "h": result.base_approximations[row].step_size,
                "Diagonal Estimate R(i,i)": float(estimate),
                "Expected Order": result.base_order * (row + 1),
                "Exact Derivative": result.exact_derivative,
                "Absolute Error": absolute_error,
                "Relative Error": relative_error,
                "Successive Difference": successive_difference,
                "Observed Order": observed_order,
            }
        )
        previous_estimate = float(estimate)
        previous_error = absolute_error

    return pd.DataFrame(rows)


def create_summary_dataframe(result: RichardsonResult) -> pd.DataFrame:
    """Build the summary sheet as property-value rows."""

    warnings = " | ".join(result.warnings) if result.warnings else "None"
    rows = [
        ("Method", result.method),
        ("Status", result.status),
        ("Function", result.function_text),
        ("Symbolic Function", human_readable_expression(result.function_expression)),
        ("Symbolic Derivative", result.derivative_expression_text),
        ("Base Method", result.base_method_name),
        ("Base Formula", result.base_formula_text),
        ("Base Error Order", f"O(h^{result.base_order})"),
        ("Evaluation Point x0", result.x_value),
        ("Initial Step Size h", result.initial_step_size),
        ("Refinement Ratio", result.refinement_ratio),
        ("Richardson Levels", result.levels),
        ("Initial Base Estimate", result.primary_base_estimate),
        ("Final Richardson Estimate", result.final_estimate),
        ("Exact Derivative", result.exact_derivative),
        ("Expected Final Order", result.final_expected_order),
        ("Initial Absolute Error", result.primary_absolute_error),
        ("Final Absolute Error", result.final_absolute_error),
        ("Final Relative Error", result.final_relative_error),
        ("Error Improvement Factor", result.error_improvement_factor),
        ("Latest Observed Order", result.latest_observed_order),
        ("Warnings", warnings),
        ("Stopping Reason", result.stopping_reason),
        ("Execution Date", result.execution_datetime.strftime("%Y-%m-%d %H:%M:%S %Z")),
    ]
    return pd.DataFrame(rows, columns=["Property", "Value"])


# =============================================================================
# Scientific plots
# =============================================================================
def create_function_plot(result: RichardsonResult) -> plt.Figure:
    """Plot the function, evaluation point, sampled points, and tangent line."""

    if not result.success or result.function_expression is None or result.x_value is None:
        raise ValueError("A successful result is required for the function graph.")

    x_symbol = sp.Symbol("x", real=True)
    numeric_function = create_numeric_function(result.function_expression, x_symbol)

    largest_h = result.initial_step_size or 1.0
    span = max(4.0 * largest_h, 1.0, 0.4 * abs(result.x_value))
    x_values = np.linspace(result.x_value - span, result.x_value + span, 700)
    y_values = evaluate_real_array(numeric_function, x_values)
    valid_mask = np.isfinite(y_values)
    if np.count_nonzero(valid_mask) < 10:
        raise ValueError("The function is not finite over a sufficient plotting range.")

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(x_values, y_values, linewidth=2, label=f"f(x) = {result.function_text}")
    axis.axhline(0.0, linewidth=1)
    axis.axvline(result.x_value, linestyle="--", linewidth=1, label="x₀")

    point_y = evaluate_real_scalar(numeric_function, result.x_value, "f(x₀)")
    axis.scatter([result.x_value], [point_y], s=80, zorder=5, label="Evaluation point")

    primary = result.base_approximations[0]
    sample_x = [primary.x_plus_h]
    sample_y = [primary.f_x_plus_h]
    if primary.x_minus_h is not None and primary.f_x_minus_h is not None:
        sample_x.insert(0, primary.x_minus_h)
        sample_y.insert(0, primary.f_x_minus_h)
    axis.scatter(sample_x, sample_y, s=65, zorder=5, label="Base sample points")

    tangent_x = np.linspace(result.x_value - 0.45 * span, result.x_value + 0.45 * span, 150)
    tangent_y = point_y + float(result.final_estimate) * (tangent_x - result.x_value)
    axis.plot(
        tangent_x,
        tangent_y,
        linestyle="--",
        linewidth=2,
        label=f"Tangent using Richardson ≈ {format_number(result.final_estimate, 8)}",
    )

    axis.set_title("Function and Richardson-Based Tangent Approximation")
    axis.set_xlabel("x")
    axis.set_ylabel("f(x)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def create_convergence_plot(result: RichardsonResult) -> plt.Figure:
    """Plot diagonal and base errors on a logarithmic scale."""

    analysis = create_diagonal_analysis_dataframe(result)
    base_df = create_base_approximation_dataframe(result)

    if result.exact_derivative is not None:
        diagonal_errors = analysis["Absolute Error"].to_numpy(dtype=float)
        base_errors = base_df["Absolute Error"].to_numpy(dtype=float)
        y_label = "Absolute Error (log scale)"
        title = "Richardson Extrapolation Error Convergence"
    else:
        diagonal_errors = analysis["Successive Difference"].to_numpy(dtype=float)
        base_values = base_df["Base Approximation R[i,0]"].to_numpy(dtype=float)
        base_errors = np.full_like(base_values, np.nan, dtype=float)
        if len(base_values) > 1:
            base_errors[1:] = np.abs(np.diff(base_values))
        y_label = "Successive Difference (log scale)"
        title = "Richardson Estimate Convergence"

    step_sizes = analysis["h"].to_numpy(dtype=float)
    valid_diagonal = np.isfinite(diagonal_errors) & (diagonal_errors > 0.0)
    valid_base = np.isfinite(base_errors) & (base_errors > 0.0)
    if not np.any(valid_diagonal) and not np.any(valid_base):
        raise ValueError("There are not enough positive error values to plot convergence.")

    figure, axis = plt.subplots(figsize=(10, 6))
    if np.any(valid_base):
        axis.loglog(
            step_sizes[valid_base],
            base_errors[valid_base],
            marker="o",
            linewidth=2,
            label="Base approximation",
        )
    if np.any(valid_diagonal):
        axis.loglog(
            step_sizes[valid_diagonal],
            diagonal_errors[valid_diagonal],
            marker="s",
            linewidth=2,
            label="Richardson diagonal",
        )

    axis.invert_xaxis()
    axis.set_title(title)
    axis.set_xlabel("Step Size h (log scale)")
    axis.set_ylabel(y_label)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


# =============================================================================
# Excel export
# =============================================================================
def serialize_warnings(warnings: Sequence[str]) -> str:
    """Serialize warnings for the Excel summary."""

    return "None" if not warnings else "\n".join(warnings)


def apply_excel_style(workbook: Any) -> None:
    """Apply consistent professional formatting to all workbook sheets."""

    header_fill = PatternFill("solid", fgColor="0D3151")
    header_font = Font(color="FFFFFF", bold=True)

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        worksheet.sheet_view.showGridLines = False

        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                if isinstance(cell.value, float):
                    cell.number_format = "0.000000000000E+00"

        for column_index, column_cells in enumerate(worksheet.columns, start=1):
            maximum_length = 0
            for cell in column_cells:
                value_length = len(str(cell.value)) if cell.value is not None else 0
                maximum_length = max(maximum_length, value_length)
            worksheet.column_dimensions[get_column_letter(column_index)].width = min(
                max(maximum_length + 2, 12),
                55,
            )


def create_excel_report(result: RichardsonResult) -> bytes:
    """Generate a formatted in-memory XLSX report."""

    if not result.success:
        raise ValueError("Only a successful result can be exported.")

    summary_df = create_summary_dataframe(result)
    base_df = create_base_approximation_dataframe(result)
    richardson_df = create_richardson_table_dataframe(result)
    steps_df = create_extrapolation_steps_dataframe(result)
    diagonal_df = create_diagonal_analysis_dataframe(result)

    formulas_df = pd.DataFrame(
        {
            "Item": [
                "Base formula",
                "Richardson recursion",
                "Refinement ratio",
                "Base order p",
                "Final expected order",
            ],
            "Expression": [
                result.base_formula_text,
                "R(i,j) = R(i,j-1) + [R(i,j-1)-R(i-1,j-1)] / [r^(p*j)-1]",
                result.refinement_ratio,
                result.base_order,
                result.final_expected_order,
            ],
        }
    )

    function_sample_df = pd.DataFrame(
        {
            "Level": [item.level for item in result.base_approximations],
            "h": [item.step_size for item in result.base_approximations],
            "x0-h": [item.x_minus_h for item in result.base_approximations],
            "f(x0-h)": [item.f_x_minus_h for item in result.base_approximations],
            "x0": [item.x_value for item in result.base_approximations],
            "f(x0)": [item.f_x for item in result.base_approximations],
            "x0+h": [item.x_plus_h for item in result.base_approximations],
            "f(x0+h)": [item.f_x_plus_h for item in result.base_approximations],
            "Substitution": [item.substitution_text for item in result.base_approximations],
        }
    )

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        formulas_df.to_excel(writer, sheet_name="Method Formulas", index=False)
        function_sample_df.to_excel(writer, sheet_name="Function Samples", index=False)
        base_df.to_excel(writer, sheet_name="Base Approximations", index=False)
        richardson_df.to_excel(writer, sheet_name="Richardson Table", index=False)
        steps_df.to_excel(writer, sheet_name="Extrapolation Steps", index=False)
        diagonal_df.to_excel(writer, sheet_name="Error Analysis", index=False)
        diagonal_df.to_excel(writer, sheet_name="Convergence Analysis", index=False)

        workbook = writer.book
        apply_excel_style(workbook)

        if not diagonal_df.empty:
            convergence_sheet = workbook["Convergence Analysis"]
            row_count = len(diagonal_df) + 1
            categories = Reference(
                convergence_sheet,
                min_col=1,
                min_row=2,
                max_row=row_count,
            )

            estimate_chart = LineChart()
            estimate_chart.title = "Richardson Diagonal Estimates"
            estimate_chart.x_axis.title = "Level"
            estimate_chart.y_axis.title = "Derivative Estimate"
            estimate_data = Reference(
                convergence_sheet,
                min_col=3,
                min_row=1,
                max_row=row_count,
            )
            estimate_chart.add_data(estimate_data, titles_from_data=True)
            estimate_chart.set_categories(categories)
            estimate_chart.height = 8
            estimate_chart.width = 16
            convergence_sheet.add_chart(estimate_chart, "K2")

            if "Absolute Error" in diagonal_df.columns:
                error_column = diagonal_df.columns.get_loc("Absolute Error") + 1
                error_chart = LineChart()
                error_chart.title = "Richardson Absolute Error"
                error_chart.x_axis.title = "Level"
                error_chart.y_axis.title = "Absolute Error"
                error_data = Reference(
                    convergence_sheet,
                    min_col=error_column,
                    min_row=1,
                    max_row=row_count,
                )
                error_chart.add_data(error_data, titles_from_data=True)
                error_chart.set_categories(categories)
                error_chart.height = 8
                error_chart.width = 16
                convergence_sheet.add_chart(error_chart, "K20")

    output.seek(0)
    return finalize_excel_report_with_visible_charts(output.getvalue())


# =============================================================================
# Streamlit result rendering
# =============================================================================

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


def render_final_result(result: RichardsonResult) -> None:
    """Render the compact final-result card."""

    if not result.success:
        st.error(result.message)
        st.caption(result.stopping_reason)
        return

    st.success(result.message)
    st.markdown(f"**Function:** `{result.function_text}`")
    st.markdown(f"**Base method:** {result.base_method_name}")

    metric_columns = st.columns(2)
    metric_columns[0].metric(
        "Final Richardson Estimate",
        format_number(result.final_estimate, 10),
    )
    metric_columns[1].metric(
        "Exact f′(x₀)",
        format_number(result.exact_derivative, 10),
    )

    detail_columns = st.columns(2)
    detail_columns[0].metric("Levels", str(result.levels))
    detail_columns[1].metric(
        "Expected Final Order",
        f"O(h^{result.final_expected_order})",
    )

    if result.final_absolute_error is not None:
        error_columns = st.columns(2)
        error_columns[0].metric(
            "Final Absolute Error",
            scientific_number(result.final_absolute_error),
        )
        error_columns[1].metric(
            "Improvement Factor",
            (
                "∞"
                if result.error_improvement_factor is not None
                and math.isinf(result.error_improvement_factor)
                else format_number(result.error_improvement_factor, 3)
            ),
        )

    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)


def render_base_calculations(result: RichardsonResult) -> None:
    """Render the base finite-difference calculations."""

    st.subheader("Base Finite-Difference Approximations")
    st.dataframe(
        round_numeric_dataframe(create_base_approximation_dataframe(result)),
        use_container_width=True,
        hide_index=True,
    )

    for item in result.base_approximations:
        with st.expander(
            f"Level {item.level}: h = {format_number(item.step_size, 10)}",
            expanded=(item.level == 0),
        ):
            st.markdown(f"**Formula:** {result.base_formula_text}")
            st.code(item.substitution_text, language=None)
            st.markdown(
                "**Base estimate:** "
                f"{format_number(item.approximation, 12)}"
            )
            if item.absolute_error is not None:
                st.markdown(
                    f"**Absolute error:** {scientific_number(item.absolute_error)}"
                )


def render_richardson_table(result: RichardsonResult) -> None:
    """Render the triangular Richardson table and all recursion steps."""

    st.subheader("Richardson Extrapolation Table")
    st.dataframe(
        round_numeric_dataframe(create_richardson_table_dataframe(result), 10),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Each new column removes the leading truncation-error term. For a base "
        f"method of order p = {result.base_order}, column j uses the factor "
        "2^(p·j) − 1 and has an expected order of "
        f"O(h^{result.base_order * 2}), O(h^{result.base_order * 3}), and so on."
    )

    st.markdown("#### Extrapolation Steps")
    for step in result.extrapolation_steps:
        with st.expander(
            f"R({step.row},{step.column}) — expected O(h^{step.expected_order})"
        ):
            st.markdown(f"**Formula:** `{step.formula_text}`")
            st.code(step.substitution_text, language=None)
            detail_df = pd.DataFrame(
                {
                    "Property": [
                        "Step size",
                        "Current lower-order estimate",
                        "Previous-row estimate",
                        "Exponent",
                        "Denominator factor",
                        "Correction",
                        "Extrapolated estimate",
                        "Absolute error",
                    ],
                    "Value": [
                        step.step_size,
                        step.lower_order_estimate,
                        step.previous_row_estimate,
                        step.exponent,
                        step.denominator_factor,
                        step.correction,
                        step.extrapolated_estimate,
                        step.absolute_error,
                    ],
                }
            )
            st.dataframe(
                round_numeric_dataframe(detail_df, 10),
                use_container_width=True,
                hide_index=True,
            )


def render_error_analysis(result: RichardsonResult) -> None:
    """Render diagonal error and convergence analysis."""

    st.subheader("Error and Convergence Analysis")
    analysis_df = create_diagonal_analysis_dataframe(result)
    st.dataframe(
        round_numeric_dataframe(analysis_df, 10),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Richardson extrapolation assumes an asymptotic truncation-error "
        "expansion in powers compatible with the selected base method. The "
        "central base formula uses even powers of h, while the forward base "
        "formula begins with first-order error. Very small h can eventually "
        "increase error because of floating-point cancellation."
    )

    if result.latest_observed_order is not None:
        st.markdown(
            "**Latest observed convergence order:** "
            f"{format_number(result.latest_observed_order, 4)}"
        )

    try:
        figure = create_convergence_plot(result)
    except ValueError as error:
        st.info(str(error))
    else:
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)


def render_function_graph(result: RichardsonResult) -> None:
    """Render the function and tangent approximation graph."""

    st.subheader("Function Graph")
    try:
        figure = create_function_plot(result)
    except ValueError as error:
        st.warning(f"The graph could not be displayed. {error}")
    else:
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)


def render_excel_download(result: RichardsonResult) -> None:
    """Create and render the Excel report download button."""

    st.subheader("Excel Report")
    report_signature = result.input_signature
    cached_signature = st.session_state.get("richardson_excel_signature")

    if cached_signature != report_signature:
        try:
            report_bytes = create_excel_report(result)
        except (ValueError, OSError, RuntimeError) as error:
            st.error(f"The Excel report could not be generated. {error}")
            return
        st.session_state.richardson_excel_report = report_bytes
        st.session_state.richardson_excel_signature = report_signature

    report_bytes = st.session_state.get("richardson_excel_report")
    if report_bytes is None:
        st.error("The Excel report is unavailable.")
        return

    timestamp = result.execution_datetime.strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="Download Excel Report",
        data=report_bytes,
        file_name=f"richardson_extrapolation_report_{timestamp}.xlsx",
        mime=EXCEL_MIME_TYPE,
        use_container_width=True,
        key="richardson_download_button",
    )


# =============================================================================
# Streamlit page
# =============================================================================
def render_page() -> None:
    """Render the complete Richardson Extrapolation Streamlit page."""

    st.set_page_config(
        page_title="Richardson Extrapolation Solver | Numerical Methods",
        page_icon="📈",
        layout="wide",
    )
    load_css()

    navbar(active_page="solver")

    st.html(
        """
        <section class="solver-hero">
            <div>
                <div class="page-label">NUMERICAL DIFFERENTIATION TOOL</div>
                <h1>Richardson Extrapolation Solver</h1>
                <p>
                    Improve finite-difference derivative estimates by combining
                    results from successively refined step sizes. Review the
                    complete triangular Richardson table, every extrapolation
                    operation, error analysis, convergence graph, and Excel report.
                </p>

                <div class="method-actions">
                    <a href="/Richardson_Extrapolation" target="_self"
                       class="btn-outline-ui">Review Lesson →</a>
                    <a href="/Richardson_Extrapolation_Quiz" target="_self"
                       class="btn-primary-ui">Take Quiz →</a>
                </div>
            </div>
        </section>
        """
    )

    # Match the centered Bisection solver content width.
    left_margin, main_area, right_margin = st.columns([0.035, 0.93, 0.035])
    with main_area:
        st.markdown(
            '<main class="solver-wrapper solver-streamlit-area">',
            unsafe_allow_html=True,
        )
    
        guide_column, conditions_column = st.columns(2)

        with guide_column:
            with st.container(border=True):
                st.subheader('How to Write the Function')
                st.markdown(
                    """
                Enter only the mathematical expression, without `f(x) =` or an equals sign.

                - Use only **x** as the variable.
                - Powers: write `x**2`, not **x^2**.
                - Multiplication: write `2*x`, not `2x`.
                - Use lowercase functions such as **sin(x)**, **cos(x)**, **exp(x)**, **sqrt(x)**, and **log(x)**.
                - Use parentheses whenever the order of operations could be unclear.
                    """
                )

        with conditions_column:
            with st.container(border=True):
                st.subheader('Before Solving')
                st.markdown(
                    """
                - Choose a positive initial step size **h** and at least two refinement levels.
                - The function must be finite at every point required by the selected base-difference formula.
                - Richardson extrapolation assumes the leading error behaves like a known power of **h**.
                - Central difference uses base order **2**; forward difference uses base order **1**.
                - Very small refined step sizes can amplify floating-point error.
                    """
                )

        input_column, result_column = st.columns([1.35, 1.0])
    
        with input_column:
            with st.container(border=True):
                st.markdown(
                    '<h3 class="solver-box-title">Input</h3>',
                    unsafe_allow_html=True,
                )
    
                st.markdown(
                    '<div class="input-label-ui">Function f(x)</div>',
                    unsafe_allow_html=True,
                )
                function_text = st.text_input(
                    "Function f(x)",
                    value=DEFAULT_FUNCTION,
                    placeholder="Example: sin(x), exp(x), or x**3 - 2*x + 1",
                    label_visibility="collapsed",
                    key="richardson_function",
                )
    
                value_columns = st.columns(2)
                with value_columns[0]:
                    st.markdown(
                        '<div class="input-label-ui">Evaluation point x₀</div>',
                        unsafe_allow_html=True,
                    )
                    x_value = st.number_input(
                        "Evaluation point x0",
                        value=DEFAULT_X_VALUE,
                        format="%.12g",
                        label_visibility="collapsed",
                        key="richardson_x_value",
                    )
    
                with value_columns[1]:
                    st.markdown(
                        '<div class="input-label-ui">Initial step size h</div>',
                        unsafe_allow_html=True,
                    )
                    step_size = st.number_input(
                        "Initial step size h",
                        value=DEFAULT_STEP_SIZE,
                        min_value=0.0,
                        format="%.12g",
                        label_visibility="collapsed",
                        key="richardson_step_size",
                    )
    
                st.markdown(
                    '<div class="input-label-ui">Base difference method</div>',
                    unsafe_allow_html=True,
                )
                base_method_name = st.selectbox(
                    "Base difference method",
                    options=list(BASE_METHOD_OPTIONS.keys()),
                    index=0,
                    label_visibility="collapsed",
                    key="richardson_base_method",
                )
    
                st.markdown(
                    '<div class="input-label-ui">Richardson levels</div>',
                    unsafe_allow_html=True,
                )
                levels = st.slider(
                    "Richardson levels",
                    min_value=MIN_LEVELS,
                    max_value=MAX_LEVELS,
                    value=DEFAULT_LEVELS,
                    step=1,
                    label_visibility="collapsed",
                    key="richardson_levels",
                )
    
                st.caption(
                    "The solver uses h, h/2, h/4, and so on. More levels can improve "
                    "truncation error, but excessively small steps may amplify "
                    "floating-point cancellation."
                )
    
                solve_button_clicked = st.button(
                    "Solve",
                    use_container_width=True,
                    key="richardson_solve_button",
                )
    
        current_input_signature = create_input_signature(
            function_text=function_text,
            x_value=x_value,
            step_size=step_size,
            base_method_name=base_method_name,
            levels=levels,
        )
    
        with result_column:
            with st.container(border=True):
                st.markdown(
                    '<h3 class="solver-box-title">Final Result</h3>',
                    unsafe_allow_html=True,
                )
    
                stored_result = st.session_state.get("richardson_result")
                if stored_result is None:
                    st.info("Enter the function and parameters, then click Solve.")
                elif stored_result.input_signature != current_input_signature:
                    st.info(
                        "The function or numerical parameters have changed. "
                        "Click Solve to calculate a new result."
                    )
                else:
                    render_final_result(stored_result)
    
        if solve_button_clicked:
            st.session_state.richardson_result = solve_richardson_extrapolation(
                function_text=function_text,
                raw_x_value=x_value,
                raw_step_size=step_size,
                base_method_name=base_method_name,
                raw_levels=levels,
                input_signature=current_input_signature,
            )
            st.session_state.pop("richardson_excel_report", None)
            st.session_state.pop("richardson_excel_signature", None)
            st.rerun()
    
        active_result = st.session_state.get("richardson_result")
        if active_result is not None and active_result.input_signature == current_input_signature:
            if active_result.success:
                st.divider()
                render_base_calculations(active_result)
    
                st.divider()
                render_richardson_table(active_result)
    
                st.divider()
                render_error_analysis(active_result)
    
                st.divider()
                render_function_graph(active_result)
    
                st.divider()
                render_excel_download(active_result)
    
                st.divider()
                navigation_left_column, navigation_right_column = st.columns(2)
    
                with navigation_left_column:
                    if st.button(
                        "Review Richardson Extrapolation Lesson",
                        use_container_width=True,
                        key="review_richardson_lesson",
                    ):
                        st.switch_page("pages/Richardson_Extrapolation.py")
    
                with navigation_right_column:
                    if st.button(
                        "Back to Solver Menu",
                        use_container_width=True,
                        key="back_to_solver_menu_richardson",
                    ):
                        st.switch_page("pages/Numerical_Solver.py")
    
        st.markdown("</main>", unsafe_allow_html=True)

    st.html(
        """
        <footer class="footer-ui">
            <div>NM • © 2026 Numerical Methods</div>
            <div>Numerical Differentiation • Richardson Extrapolation</div>
        </footer>
        """
    )


if __name__ == "__main__":
    render_page()
