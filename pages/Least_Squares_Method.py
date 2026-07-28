from __future__ import annotations

import html

import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css


DATA = {'title': 'Least Squares Method',
 'page_label': 'CURVE FITTING METHOD',
 'hero_text': 'Learn how least squares determines the polynomial that best represents measured data by '
              'minimizing the total squared residual error.',
 'solver_page': 'Least_Squares_Solver',
 'quiz_page': 'Least_Squares_Quiz',
 'footer_text': 'Least Squares Method • Curve Fitting',
 'summary_intro': 'Least squares fits a trend to data rather than forcing the model through every '
                  'observation.',
 'summary': [('METHOD TYPE', 'Data fitting and regression'),
             ('OBJECTIVE', 'Minimize squared residuals'),
             ('OUTPUT', 'Best-fit polynomial coefficients'),
             ('KEY SYSTEM', 'Normal equations')],
 'core_intro': 'The method converts a fitting problem into a linear system for the unknown model '
               'coefficients.',
 'overview': {'title': 'Overview',
              'text': 'Given N observations, polynomial least squares selects coefficients for a '
                      'degree-m polynomial whose predicted values are collectively as close as possible '
                      'to the data.',
              'formulas': ['p_m(x)=a_0+a_1x+\\cdots+a_mx^m', 'r_i=y_i-p_m(x_i)']},
 'foundation': {'title': 'Matrix Formulation',
                'text': 'The design matrix contains one column for each polynomial term. The '
                        'coefficient vector minimizes the Euclidean norm of the residual vector.',
                'formulas': ['Xa\\approx y',
                             '\\min_a\\|Xa-y\\|_2^2',
                             'X=\\begin{bmatrix}1&x_1&\\cdots&x_1^m\\\\\\vdots&\\vdots&&\\vdots\\\\1&x_N&\\cdots&x_N^m\\end{bmatrix}']},
 'requirements_intro': 'A valid degree and a full-rank design matrix are required for a unique '
                       'coefficient vector.',
 'conditions': ['The number of observations must be at least the number of coefficients: N ≥ m + 1.',
                'The design matrix must have full column rank.',
                'For a degree-m polynomial, at least m + 1 distinct x-values are required.',
                'The chosen degree should reflect the data trend without unnecessary overfitting.'],
 'formula': {'title': 'Normal Equations',
             'text': 'Setting the gradient of the squared-error objective equal to zero produces the '
                     'coefficient system.',
             'formulas': ['S(a)=\\sum_{i=1}^{N}\\left[y_i-p_m(x_i)\\right]^2',
                          'X^T(Xa-y)=0',
                          '(X^TX)a=X^Ty'],
             'note': 'Normal equations are educational and convenient, but QR factorization or SVD is '
                     'generally more stable for ill-conditioned data.'},
 'analysis_intro': 'Fit quality is evaluated with residual statistics, while numerical reliability '
                   'depends on the conditioning of the design matrix.',
 'analysis': [{'title': 'Residual and Fit Metrics',
               'text': 'Residuals measure vertical differences between observed and fitted values. SSE, '
                       'RMSE, and R² summarize different aspects of the fit.',
               'formulas': ['\\mathrm{SSE}=\\sum r_i^2',
                            '\\mathrm{RMSE}=\\sqrt{\\frac{1}{N}\\sum r_i^2}',
                            'R^2=1-\\frac{\\mathrm{SSE}}{\\sum(y_i-\\bar y)^2}'],
               'note': 'R² is undefined when every observed y-value is identical because the total sum '
                       'of squares is zero.'},
              {'title': 'Conditioning',
               'text': 'Large x-values, a high polynomial degree, or poorly distributed points can make '
                       'the coefficient calculation sensitive to rounding and small data changes.',
               'formulas': ['\\kappa_2(X^TX)=\\kappa_2(X)^2\\quad\\text{when }X\\text{ has full column '
                            'rank}'],
               'note': 'Centering and scaling x, or solving with QR/SVD, can improve numerical '
                       'behavior.'}],
 'checks_title': 'Model and Data Checks',
 'checks_intro': 'A small residual alone does not guarantee a sensible or numerically stable model.',
 'checks': [{'title': 'Rank Check',
             'formula': '\\operatorname{rank}(X)=m+1',
             'caption': 'Ensures that the coefficients are uniquely determined.'},
            {'title': 'Residual Check',
             'formula': 'r=y-Xa',
             'caption': 'Inspect both the residual magnitudes and their pattern.'},
            {'title': 'Degree Check',
             'formula': 'm<N',
             'caption': 'Use the lowest degree that adequately represents the relationship.'}],
 'algorithm_intro': 'The workflow separates model construction, coefficient solution, and fit '
                    'verification.',
 'algorithm': ['Collect and validate the x and y observations.',
               'Choose a polynomial degree m.',
               'Build the design matrix X with columns 1, x, …, xᵐ.',
               'Form the normal equations XᵀX a = Xᵀy.',
               'Solve the coefficient system with a stable linear-system procedure.',
               'Calculate fitted values, residuals, error metrics, and diagnostic graphs.'],
 'example': {'intro': 'Fit a straight line to three observations that do not lie exactly on one line.',
             'setup': [{'label': 'Data points', 'formulas': ['(0,1),\\quad(1,2),\\quad(2,2)']},
                       {'label': 'Model', 'formulas': ['p_1(x)=a_0+a_1x']}],
             'setup_note': 'There are three observations and two unknown coefficients, so the system is '
                           'overdetermined.',
             'steps': [{'title': 'Build the Normal Equations',
                        'formulas': ['X=\\begin{bmatrix}1&0\\\\1&1\\\\1&2\\end{bmatrix},\\qquad '
                                     'y=\\begin{bmatrix}1\\\\2\\\\2\\end{bmatrix}',
                                     'X^TX=\\begin{bmatrix}3&3\\\\3&5\\end{bmatrix},\\qquad '
                                     'X^Ty=\\begin{bmatrix}5\\\\6\\end{bmatrix}'],
                        'text': 'The coefficient problem is now a 2 × 2 linear system.'},
                       {'title': 'Solve for the Coefficients',
                        'formulas': ['a_0=\\frac76\\approx1.166667', 'a_1=\\frac12=0.5'],
                        'text': 'The fitted line balances the positive and negative residuals in the '
                                'least-squares sense.'},
                       {'title': 'Calculate the Residuals',
                        'formulas': ['\\hat y=\\begin{bmatrix}7/6\\\\5/3\\\\13/6\\end{bmatrix}',
                                     'r=y-\\hat y=\\begin{bmatrix}-1/6\\\\1/3\\\\-1/6\\end{bmatrix}'],
                        'text': 'The residuals sum to zero because the model includes an intercept.'},
                       {'title': 'Evaluate the Fit',
                        'formulas': ['\\mathrm{SSE}=\\frac16',
                                     '\\mathrm{RMSE}=\\sqrt{\\frac1{18}}\\approx0.235702',
                                     'R^2=0.75'],
                        'text': 'The line explains 75% of the variation in this small example.'}],
             'result_title': 'Best-Fit Line',
             'result_formulas': ['\\hat y=\\frac76+\\frac12x'],
             'result_text': 'The line does not pass through every point; it minimizes the total squared '
                            'vertical error across all observations.'},
 'complexity_intro': 'For polynomial degree m and N observations, building and solving the small '
                     'coefficient system dominates the work.',
 'complexity': [('Design Matrix', 'O(Nm)'),
                ('Normal Matrix', 'O(Nm²)'),
                ('Coefficient Solve', 'O(m³)'),
                ('Stored Data', 'O(Nm)')],
 'complexity_note': {'text': 'When m is small compared with N, the method scales approximately linearly '
                             'with the number of observations.',
                     'formulas': ['\\text{total work}=O(Nm^2+m^3)']},
 'applications_intro': 'Least squares is used whenever measurements contain noise or an exact '
                       'interpolating curve is not desirable.',
 'applications': [('Experimental Physics',
                   'Estimating relationships and parameters from measured data.'),
                  ('Calibration', 'Constructing conversion curves for instruments and sensors.'),
                  ('Engineering Design', 'Building empirical response and performance models.'),
                  ('Trend Analysis', 'Summarizing the direction and curvature of time-dependent data.'),
                  ('Data Science', 'Fitting baseline regression models and polynomial features.'),
                  ('Error Modeling',
                   'Quantifying residual variation after fitting a deterministic trend.')],
 'evaluation_intro': 'Least squares is flexible and interpretable, but degree selection and '
                     'conditioning require care.',
 'advantages': ['Uses all observations simultaneously.',
                'Handles noisy and overdetermined data.',
                'Provides clear residual and goodness-of-fit measures.',
                'Extends naturally from straight lines to polynomial models.'],
 'limitations': ['Sensitive to outliers because residuals are squared.',
                 'High-degree models may overfit and become ill-conditioned.',
                 'Normal equations can amplify conditioning problems.',
                 'A high R² does not by itself prove that the model is appropriate.']}
PYTHON_CODE = 'import numpy as np\n\n\ndef gaussian_elimination(matrix, rhs, tolerance=1e-12):\n    A = np.asarray(matrix, dtype=float).copy()\n    b = np.asarray(rhs, dtype=float).copy()\n    n = b.size\n\n    for k in range(n - 1):\n        pivot_row = k + np.argmax(np.abs(A[k:, k]))\n        if abs(A[pivot_row, k]) <= tolerance:\n            raise ValueError("The coefficient system is singular.")\n\n        if pivot_row != k:\n            A[[k, pivot_row]] = A[[pivot_row, k]]\n            b[[k, pivot_row]] = b[[pivot_row, k]]\n\n        for i in range(k + 1, n):\n            multiplier = A[i, k] / A[k, k]\n            A[i, k:] -= multiplier * A[k, k:]\n            b[i] -= multiplier * b[k]\n\n    coefficients = np.zeros(n)\n    for i in range(n - 1, -1, -1):\n        if abs(A[i, i]) <= tolerance:\n            raise ValueError("The coefficient system is singular.")\n        coefficients[i] = (\n            b[i] - A[i, i + 1:] @ coefficients[i + 1:]\n        ) / A[i, i]\n\n    return coefficients\n\n\ndef polynomial_least_squares(x, y, degree):\n    x = np.asarray(x, dtype=float)\n    y = np.asarray(y, dtype=float)\n\n    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:\n        raise ValueError("x and y must be one-dimensional arrays of equal length.")\n    if degree < 0 or x.size < degree + 1:\n        raise ValueError("Not enough observations for the requested degree.")\n    if np.unique(x).size < degree + 1:\n        raise ValueError("Not enough distinct x-values.")\n\n    X = np.column_stack([x**power for power in range(degree + 1)])\n    normal_matrix = X.T @ X\n    normal_rhs = X.T @ y\n    coefficients = gaussian_elimination(normal_matrix, normal_rhs)\n\n    fitted = X @ coefficients\n    residuals = y - fitted\n    sse = float(residuals @ residuals)\n    rmse = float(np.sqrt(sse / x.size))\n\n    return coefficients, fitted, residuals, rmse\n\n\nx = [0, 1, 2]\ny = [1, 2, 2]\ncoefficients, fitted, residuals, rmse = polynomial_least_squares(x, y, 1)\n\nprint("Coefficients [a0, a1]:", coefficients)\nprint("RMSE:", rmse)'
MATLAB_CODE = "function [a, fitted, residuals, rmse] = PolynomialLeastSquares(x, y, degree)\n\n    x = x(:);\n    y = y(:);\n\n    if length(x) ~= length(y)\n        error('x and y must have the same length.');\n    end\n\n    if degree < 0 || length(x) < degree + 1\n        error('Not enough observations for the requested degree.');\n    end\n\n    X = zeros(length(x), degree + 1);\n    for power = 0:degree\n        X(:, power + 1) = x.^power;\n    end\n\n    if rank(X) < degree + 1\n        error('The design matrix is rank-deficient.');\n    end\n\n    % Normal equations. QR or X\\y is generally preferable in production.\n    a = (X' * X) \\ (X' * y);\n    fitted = X * a;\n    residuals = y - fitted;\n    rmse = sqrt(mean(residuals.^2));\nend\n\n% Example:\nx = [0; 1; 2];\ny = [1; 2; 2];\n[a, fitted, residuals, rmse] = PolynomialLeastSquares(x, y, 1)"


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title=f"{DATA['title']} | Numerical Methods",
    page_icon="📘",
    layout="wide",
)

load_css()
navbar(active_page="learn")


# =========================================================
# Page-specific visual styling — aligned with Bisection
# =========================================================
st.markdown(
    """
    <style>
    .method-page {
        padding-top: 26px;
        padding-bottom: 12px;
    }

    .section-kicker {
        color: #0f766e;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .section-title {
        color: #0b1b3a;
        font-size: 25px;
        font-weight: 900;
        margin: 0 0 7px;
    }

    .section-intro {
        color: #475569;
        font-size: 15px;
        line-height: 1.65;
        margin: 0 0 18px;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 4px 0 22px;
    }

    .summary-card {
        min-height: 112px;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(15, 61, 62, 0.10);
        box-shadow: 0 8px 20px rgba(15, 61, 62, 0.06);
    }

    .summary-card:nth-child(1) {
        background: linear-gradient(135deg, #f0fdfa, #ecfeff);
    }

    .summary-card:nth-child(2) {
        background: linear-gradient(135deg, #eff6ff, #e0f2fe);
    }

    .summary-card:nth-child(3) {
        background: linear-gradient(135deg, #f5f3ff, #faf5ff);
    }

    .summary-card:nth-child(4) {
        background: linear-gradient(135deg, #fff7ed, #fffbeb);
    }

    .summary-card span {
        display: block;
        color: #64748b;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .summary-card strong {
        display: block;
        color: #0b1b3a;
        font-size: 18px;
        font-weight: 900;
        line-height: 1.25;
    }

    .condition-list {
        display: grid;
        gap: 12px;
        margin-top: 10px;
    }

    .condition-item {
        display: flex;
        gap: 12px;
        align-items: flex-start;
        padding: 14px 16px;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }

    .condition-number {
        width: 28px;
        height: 28px;
        min-width: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #14b8a6, #0f766e);
        color: white;
        display: grid;
        place-items: center;
        font-size: 13px;
        font-weight: 900;
    }

    .condition-item div:last-child {
        color: #334155;
        font-size: 14px;
        line-height: 1.6;
    }

    .algorithm-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-top: 8px;
    }

    .algorithm-step {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 15px 16px;
    }

    .algorithm-step span {
        display: inline-block;
        color: #0f766e;
        font-size: 12px;
        font-weight: 900;
        margin-bottom: 6px;
    }

    .algorithm-step p {
        color: #334155;
        font-size: 14px;
        line-height: 1.55;
        margin: 0;
    }

    .application-grid-advanced {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
    }

    .application-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 18px;
        color: #334155;
        font-size: 14px;
        line-height: 1.55;
        min-height: 86px;
    }

    .application-box strong {
        display: block;
        color: #0b1b3a;
        margin-bottom: 5px;
        font-size: 14px;
    }

    .advantage-box,
    .limitation-box {
        border-radius: 16px;
        padding: 20px;
        min-height: 220px;
    }

    .advantage-box {
        background: #f0fdfa;
        border: 1px solid #99f6e4;
    }

    .limitation-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
    }

    .advantage-box h3,
    .limitation-box h3 {
        color: #0b1b3a;
        font-size: 18px;
        font-weight: 900;
        margin: 0 0 10px;
    }

    .advantage-box li,
    .limitation-box li {
        color: #475569;
        margin-bottom: 8px;
        line-height: 1.5;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;
        border: 1px solid rgba(15, 61, 62, 0.10) !important;
        box-shadow: 0 10px 24px rgba(15, 61, 62, 0.06) !important;
    }

    div[data-testid="stExpander"] {
        border-radius: 14px !important;
        border-color: rgba(15, 61, 62, 0.12) !important;
        overflow: hidden !important;
    }

    @media (max-width: 1000px) {
        .summary-grid,
        .application-grid-advanced,
        .algorithm-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Hero
# =========================================================
quiz_button = ""
if DATA.get("quiz_page"):
    quiz_button = f"""
        <a href="/{DATA['quiz_page']}" target="_self" class="btn-outline-ui">
            Take Quiz →
        </a>
    """

st.html(
    f"""
    <section class="method-hero">
        <div>
            <div class="page-label">{html.escape(DATA['page_label'])}</div>
            <h1>{html.escape(DATA['title'])}</h1>
            <p>{html.escape(DATA['hero_text'])}</p>
            <div class="method-actions">
                <a href="/{DATA['solver_page']}" target="_self" class="btn-primary-ui">
                    Open Solver →
                </a>
                {quiz_button}
            </div>
        </div>
    </section>
    """
)


# =========================================================
# Main page
# =========================================================
left_margin, content, right_margin = st.columns([0.035, 0.93, 0.035])

with content:
    st.markdown('<div class="method-page">', unsafe_allow_html=True)

    # Quick summary
    summary_cards = "".join(
        f"""
        <div class="summary-card">
            <span>{html.escape(label)}</span>
            <strong>{html.escape(value)}</strong>
        </div>
        """
        for label, value in DATA["summary"]
    )

    st.html(
        f"""
        <div class="section-kicker">Quick reference</div>
        <h2 class="section-title">Method at a Glance</h2>
        <p class="section-intro">{html.escape(DATA['summary_intro'])}</p>
        <div class="summary-grid">{summary_cards}</div>
        """
    )

    # Overview and foundation
    st.html(
        f"""
        <div class="section-kicker">Core concept</div>
        <h2 class="section-title">Overview and Mathematical Foundation</h2>
        <p class="section-intro">{html.escape(DATA['core_intro'])}</p>
        """
    )

    overview_col, foundation_col = st.columns(2)

    for column, section in zip(
        (overview_col, foundation_col),
        (DATA["overview"], DATA["foundation"]),
    ):
        with column:
            with st.container(border=True):
                st.subheader(section["title"])
                st.write(section["text"])
                for formula in section.get("formulas", []):
                    st.latex(formula)
                if section.get("note"):
                    st.info(section["note"])

    # Conditions and core formula
    st.html(
        f"""
        <div class="section-kicker">Requirements</div>
        <h2 class="section-title">Conditions and Core Formula</h2>
        <p class="section-intro">{html.escape(DATA['requirements_intro'])}</p>
        """
    )

    conditions_col, formula_col = st.columns([1.05, 0.95])

    with conditions_col:
        with st.container(border=True):
            st.subheader("Required Conditions")
            conditions_html = "".join(
                f"""
                <div class="condition-item">
                    <div class="condition-number">{index}</div>
                    <div>{html.escape(condition)}</div>
                </div>
                """
                for index, condition in enumerate(DATA["conditions"], start=1)
            )
            st.html(f'<div class="condition-list">{conditions_html}</div>')

    with formula_col:
        with st.container(border=True):
            st.subheader(DATA["formula"]["title"])
            st.write(DATA["formula"]["text"])
            for formula in DATA["formula"]["formulas"]:
                st.latex(formula)
            if DATA["formula"].get("note"):
                st.info(DATA["formula"]["note"])

    # Accuracy and numerical behavior
    st.html(
        f"""
        <div class="section-kicker">Accuracy</div>
        <h2 class="section-title">Error and Numerical Behavior</h2>
        <p class="section-intro">{html.escape(DATA['analysis_intro'])}</p>
        """
    )

    analysis_columns = st.columns(2)
    for column, section in zip(analysis_columns, DATA["analysis"]):
        with column:
            with st.container(border=True):
                st.subheader(section["title"])
                st.write(section["text"])
                for formula in section.get("formulas", []):
                    st.latex(formula)
                if section.get("note"):
                    st.caption(section["note"])

    # Validation checks
    with st.container(border=True):
        st.subheader(DATA["checks_title"])
        st.write(DATA["checks_intro"])
        check_columns = st.columns(len(DATA["checks"]))
        for column, check in zip(check_columns, DATA["checks"]):
            with column:
                st.markdown(f"**{check['title']}**")
                if check.get("formula"):
                    st.latex(check["formula"])
                else:
                    st.markdown(f"### {check['value']}")
                st.caption(check["caption"])

    # Algorithm
    algorithm_html = "".join(
        f"""
        <div class="algorithm-step">
            <span>STEP {index}</span>
            <p>{html.escape(step)}</p>
        </div>
        """
        for index, step in enumerate(DATA["algorithm"], start=1)
    )

    st.html(
        f"""
        <div class="section-kicker">Procedure</div>
        <h2 class="section-title">Algorithm</h2>
        <p class="section-intro">{html.escape(DATA['algorithm_intro'])}</p>
        <div class="algorithm-grid">{algorithm_html}</div>
        """
    )

    # Worked example
    example = DATA["example"]
    st.html(
        f"""
        <div class="section-kicker">Application</div>
        <h2 class="section-title">Worked Example</h2>
        <p class="section-intro">{html.escape(example['intro'])}</p>
        """
    )

    with st.container(border=True):
        setup_columns = st.columns(len(example["setup"]))
        for column, setup_item in zip(setup_columns, example["setup"]):
            with column:
                st.markdown(f"**{setup_item['label']}**")
                for formula in setup_item["formulas"]:
                    st.latex(formula)
        if example.get("setup_note"):
            st.success(example["setup_note"])

    for start in range(0, len(example["steps"]), 2):
        step_columns = st.columns(2)
        for offset, column in enumerate(step_columns):
            step_index = start + offset
            if step_index >= len(example["steps"]):
                continue
            step = example["steps"][step_index]
            with column:
                with st.container(border=True):
                    st.subheader(step["title"])
                    for formula in step.get("formulas", []):
                        st.latex(formula)
                    if step.get("text"):
                        st.write(step["text"])

    with st.container(border=True):
        st.subheader(example["result_title"])
        result_col, note_col = st.columns([0.46, 0.54])
        with result_col:
            for formula in example["result_formulas"]:
                st.latex(formula)
        with note_col:
            st.write(example["result_text"])

    # Implementations
    st.html(
        """
        <div class="section-kicker">Programming</div>
        <h2 class="section-title">Implementation</h2>
        <p class="section-intro">
            Expand either language to examine a complete, validated implementation.
        </p>
        """
    )

    with st.container(border=True):
        python_column, matlab_column = st.columns(2)
        with python_column:
            with st.expander("🐍 Python Implementation", expanded=False):
                st.code(PYTHON_CODE, language="python")
        with matlab_column:
            with st.expander("🟠 MATLAB Implementation", expanded=False):
                st.code(MATLAB_CODE, language="matlab")

    # Complexity
    st.html(
        f"""
        <div class="section-kicker">Performance</div>
        <h2 class="section-title">Computational Complexity</h2>
        <p class="section-intro">{html.escape(DATA['complexity_intro'])}</p>
        """
    )

    complexity_columns = st.columns(4)
    for column, metric in zip(complexity_columns, DATA["complexity"]):
        with column:
            st.metric(metric[0], metric[1])

    with st.container(border=True):
        st.write(DATA["complexity_note"]["text"])
        for formula in DATA["complexity_note"].get("formulas", []):
            st.latex(formula)

    # Applications
    application_html = "".join(
        f"""
        <div class="application-box">
            <strong>{html.escape(title)}</strong>
            {html.escape(description)}
        </div>
        """
        for title, description in DATA["applications"]
    )

    st.html(
        f"""
        <div class="section-kicker">Engineering context</div>
        <h2 class="section-title">Applications</h2>
        <p class="section-intro">{html.escape(DATA['applications_intro'])}</p>
        <div class="application-grid-advanced">{application_html}</div>
        """
    )

    # Advantages and limitations
    advantages_html = "".join(
        f"<li>{html.escape(item)}</li>" for item in DATA["advantages"]
    )
    limitations_html = "".join(
        f"<li>{html.escape(item)}</li>" for item in DATA["limitations"]
    )

    st.html(
        f"""
        <div class="section-kicker">Evaluation</div>
        <h2 class="section-title">Advantages and Limitations</h2>
        <p class="section-intro">{html.escape(DATA['evaluation_intro'])}</p>
        """
    )

    advantages_col, limitations_col = st.columns(2)
    with advantages_col:
        st.html(
            f"""
            <div class="advantage-box">
                <h3>Advantages</h3>
                <ul>{advantages_html}</ul>
            </div>
            """
        )
    with limitations_col:
        st.html(
            f"""
            <div class="limitation-box">
                <h3>Limitations</h3>
                <ul>{limitations_html}</ul>
            </div>
            """
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Footer
# =========================================================
st.html(
    f"""
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>{html.escape(DATA['footer_text'])}</div>
    </footer>
    """
)
