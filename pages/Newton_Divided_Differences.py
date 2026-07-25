from __future__ import annotations

import html

import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css


DATA = {'title': 'Newton Divided Differences',
 'page_label': 'POLYNOMIAL INTERPOLATION METHOD',
 'hero_text': 'Learn how a triangular divided-difference table produces an incremental Newton '
              'interpolation polynomial that is efficient to extend and evaluate.',
 'solver_page': 'Newton_Divided_Differences_Solver',
 'quiz_page': 'Polynomial_Interpolation_Quiz',
 'footer_text': 'Newton Divided Differences • Polynomial Interpolation',
 'summary_intro': 'Newton form stores interpolation information as divided-difference coefficients.',
 'summary': [('METHOD TYPE', 'Incremental interpolation'),
             ('COEFFICIENTS', 'Top row of difference table'),
             ('TABLE COST', 'O(n²)'),
             ('EVALUATION COST', 'O(n)')],
 'core_intro': 'Recursive differences encode the slope, curvature, and higher-order behavior of the '
               'interpolation polynomial.',
 'overview': {'title': 'Overview',
              'text': 'Newton divided differences constructs the same unique interpolation polynomial '
                      'as Lagrange interpolation, but in a nested form based on ordered nodes.',
              'formulas': ['P_n(x)=a_0+a_1(x-x_0)+\\cdots+a_n\\prod_{j=0}^{n-1}(x-x_j)']},
 'foundation': {'title': 'Divided Differences',
                'text': 'First-order differences are secant slopes. Higher-order divided differences '
                        'are calculated recursively from lower-order entries.',
                'formulas': ['f[x_i,x_{i+1}]=\\frac{f[x_{i+1}]-f[x_i]}{x_{i+1}-x_i}',
                             'f[x_i,\\ldots,x_{i+k}]=\\frac{f[x_{i+1},\\ldots,x_{i+k}]-f[x_i,\\ldots,x_{i+k-1}]}{x_{i+k}-x_i}']},
 'requirements_intro': 'The recursion divides by differences between node coordinates, so ordinary '
                       'Newton interpolation requires distinct x-values.',
 'conditions': ['All x-values must be distinct.',
                'The entered node order must remain consistent throughout the table and polynomial.',
                'The x and y arrays must have equal nonzero length and finite values.',
                'Repeated nodes require Hermite interpolation, not the ordinary divided-difference '
                'formula.'],
 'formula': {'title': 'Newton Polynomial',
             'text': 'The first entry of each divided-difference column becomes one coefficient in the '
                     'Newton form.',
             'formulas': ['a_k=f[x_0,x_1,\\ldots,x_k]',
                          'P_n(x)=f[x_0]+\\sum_{k=1}^{n}f[x_0,\\ldots,x_k]\\prod_{j=0}^{k-1}(x-x_j)'],
             'note': 'The nested form can be evaluated with a Horner-like backward calculation using '
                     'O(n) operations.'},
 'analysis_intro': 'The polynomial is exact at its nodes, and its remainder is the same as for any '
                   'degree-n interpolation formula.',
 'analysis': [{'title': 'Interpolation Error',
               'text': 'For a sufficiently smooth function, the remainder depends on the next divided '
                       'difference or derivative.',
               'formulas': ['f(x)-P_n(x)=f[x_0,\\ldots,x_n,x]\\prod_{i=0}^{n}(x-x_i)',
                            'f(x)-P_n(x)=\\frac{f^{(n+1)}(\\xi)}{(n+1)!}\\prod_{i=0}^{n}(x-x_i)']},
              {'title': 'Incremental Extension',
               'text': 'When a new node is appended, the existing coefficients remain valid. Only the '
                       'new divided differences and one additional Newton term must be calculated.',
               'formulas': ['P_{n+1}(x)=P_n(x)+f[x_0,\\ldots,x_{n+1}]\\prod_{j=0}^{n}(x-x_j)'],
               'note': 'This is a major practical advantage over rebuilding all direct Lagrange basis '
                       'functions.'}],
 'checks_title': 'Table and Polynomial Checks',
 'checks_intro': 'The triangular table, Newton coefficients, and node residuals provide several '
                 'independent checks.',
 'checks': [{'title': 'Denominator Check',
             'formula': 'x_{i+k}-x_i\\neq0',
             'caption': 'Every divided difference must have a valid node span.'},
            {'title': 'Coefficient Check',
             'formula': 'a_k=f[x_0,\\ldots,x_k]',
             'caption': 'Read one coefficient from the first row of each order.'},
            {'title': 'Node Reproduction',
             'formula': 'P_n(x_i)=y_i',
             'caption': 'The final polynomial should match all supplied data.'}],
 'algorithm_intro': 'Build the table one order at a time, extract the leading coefficients, then '
                    'evaluate the nested polynomial.',
 'algorithm': ['Validate equal-length data arrays and distinct x-values.',
               'Place the y-values in the first divided-difference column.',
               'Calculate first-order differences from adjacent entries.',
               'Continue recursively until the highest-order difference is obtained.',
               'Read the first entry of each column as a Newton coefficient.',
               'Evaluate the nested Newton polynomial and verify it at the nodes.'],
 'example': {'intro': 'Build a second-degree Newton polynomial from three nodes and evaluate it at x = '
                      '1.5.',
             'setup': [{'label': 'Interpolation nodes', 'formulas': ['(0,1),\\quad(1,2),\\quad(2,5)']},
                       {'label': 'Evaluation point', 'formulas': ['x=1.5']}],
             'setup_note': 'The nodes are distinct and remain in the entered order 0, 1, 2.',
             'steps': [{'title': 'First-Order Differences',
                        'formulas': ['f[x_0,x_1]=\\frac{2-1}{1-0}=1', 'f[x_1,x_2]=\\frac{5-2}{2-1}=3'],
                        'text': 'These entries are the secant slopes across adjacent intervals.'},
                       {'title': 'Second-Order Difference',
                        'formulas': ['f[x_0,x_1,x_2]=\\frac{3-1}{2-0}=1'],
                        'text': 'This coefficient captures the quadratic contribution.'},
                       {'title': 'Construct the Polynomial',
                        'formulas': ['P_2(x)=1+1(x-0)+1(x-0)(x-1)'],
                        'text': 'The coefficients are 1, 1, and 1.'},
                       {'title': 'Evaluate',
                        'formulas': ['P_2(1.5)=1+1.5+1.5(0.5)=3.25'],
                        'text': 'Expanding gives P₂(x) = x² + 1.'}],
             'result_title': 'Interpolated Value',
             'result_formulas': ['P_2(1.5)=3.25'],
             'result_text': 'The result is identical to the Lagrange result because both forms '
                            'represent the same unique polynomial.'},
 'complexity_intro': 'The table is quadratic to construct, but each later evaluation is linear in the '
                     'number of nodes.',
 'complexity': [('Table Construction', 'O(n²)'),
                ('Table Storage', 'O(n²)'),
                ('Coefficient Storage', 'O(n)'),
                ('Nested Evaluation', 'O(n)')],
 'complexity_note': {'text': 'An in-place coefficient algorithm can reduce storage to O(n), while '
                             'retaining O(n²) construction time.',
                     'formulas': ['\\text{build}=O(n^2),\\qquad \\text{evaluate}=O(n)']},
 'applications_intro': 'Newton form is valuable when interpolation data grows over time or the same '
                       'polynomial must be evaluated repeatedly.',
 'applications': [('Incremental Data',
                   'Appending new measurements without rebuilding every previous term.'),
                  ('Lookup Tables',
                   'Evaluating a fixed interpolation polynomial at many target values.'),
                  ('Experimental Science', 'Reconstructing a trend from exact sampled values.'),
                  ('Numerical Analysis',
                   'Developing interpolation-based integration and differentiation formulas.'),
                  ('Engineering Tables', 'Estimating intermediate material or system properties.'),
                  ('Educational Computing',
                   'Displaying every interpolation coefficient in a triangular table.')],
 'evaluation_intro': 'Newton form is efficient to extend and evaluate, but it still inherits the '
                     'limitations of high-degree global interpolation.',
 'advantages': ['Easy to extend when a new node is added.',
                'Nested evaluation requires only O(n) work.',
                'The table clearly exposes every interpolation coefficient.',
                'Produces the same unique polynomial as Lagrange interpolation.'],
 'limitations': ['Ordinary divided differences do not allow repeated nodes.',
                 'The displayed coefficients depend on the chosen node order.',
                 'High-degree interpolation may oscillate or become ill-conditioned.',
                 'Building the full table requires O(n²) work and storage.']}
PYTHON_CODE = 'import numpy as np\n\n\ndef divided_difference_coefficients(x_nodes, y_nodes):\n    x_nodes = np.asarray(x_nodes, dtype=float)\n    coefficients = np.asarray(y_nodes, dtype=float).copy()\n    n = x_nodes.size\n\n    if n == 0 or coefficients.size != n:\n        raise ValueError("x and y must have the same nonzero length.")\n    if np.unique(x_nodes).size != n:\n        raise ValueError("All x-values must be distinct.")\n\n    # In-place divided differences. The first k + 1 entries after each\n    # stage contain the Newton coefficients calculated so far.\n    for order in range(1, n):\n        for i in range(n - 1, order - 1, -1):\n            denominator = x_nodes[i] - x_nodes[i - order]\n            coefficients[i] = (\n                coefficients[i] - coefficients[i - 1]\n            ) / denominator\n\n    return coefficients\n\n\ndef evaluate_newton(x_nodes, coefficients, x_target):\n    x_nodes = np.asarray(x_nodes, dtype=float)\n    coefficients = np.asarray(coefficients, dtype=float)\n\n    value = coefficients[-1]\n    for i in range(coefficients.size - 2, -1, -1):\n        value = coefficients[i] + (x_target - x_nodes[i]) * value\n\n    return float(value)\n\n\nx = [0, 1, 2]\ny = [1, 2, 5]\ncoefficients = divided_difference_coefficients(x, y)\nresult = evaluate_newton(x, coefficients, 1.5)\n\nprint("Newton coefficients:", coefficients)\nprint("P(1.5) =", result)'
MATLAB_CODE = "function [coefficients, value] = NewtonInterpolation(x, y, target)\n\n    x = x(:).';\n    coefficients = y(:).';\n    n = length(x);\n\n    if n == 0 || length(coefficients) ~= n\n        error('x and y must have the same nonzero length.');\n    end\n\n    if length(unique(x)) ~= n\n        error('All x-values must be distinct.');\n    end\n\n    for order = 2:n\n        for i = n:-1:order\n            coefficients(i) = (coefficients(i) - coefficients(i - 1)) ...\n                / (x(i) - x(i - order + 1));\n        end\n    end\n\n    value = coefficients(n);\n    for i = n-1:-1:1\n        value = coefficients(i) + (target - x(i)) * value;\n    end\nend\n\n% Example:\nx = [0 1 2];\ny = [1 2 5];\n[coefficients, result] = NewtonInterpolation(x, y, 1.5)"


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
