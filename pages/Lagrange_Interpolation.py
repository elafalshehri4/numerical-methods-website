from __future__ import annotations

import html

import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css


DATA = {'title': 'Lagrange Interpolation',
 'page_label': 'POLYNOMIAL INTERPOLATION METHOD',
 'hero_text': 'Learn how Lagrange basis polynomials combine distinct data points directly into the '
              'unique polynomial that passes through every supplied node.',
 'solver_page': 'Lagrange_Interpolation_Solver',
 'quiz_page': 'Polynomial_Interpolation_Quiz',
 'footer_text': 'Lagrange Interpolation • Polynomial Interpolation',
 'summary_intro': 'Lagrange interpolation constructs the polynomial directly from cardinal basis '
                  'functions.',
 'summary': [('METHOD TYPE', 'Direct interpolation formula'),
             ('REQUIREMENT', 'Distinct x-values'),
             ('NODE PROPERTY', 'Lᵢ(xⱼ) = δᵢⱼ'),
             ('STANDARD COST', 'O(n²) per evaluation')],
 'core_intro': 'Each basis polynomial selects one data value while vanishing at every other node.',
 'overview': {'title': 'Overview',
              'text': 'For n + 1 distinct nodes, Lagrange interpolation expresses the unique '
                      'degree-at-most-n polynomial as a weighted sum of basis polynomials.',
              'formulas': ['P_n(x)=\\sum_{i=0}^{n}y_iL_i(x)']},
 'foundation': {'title': 'Cardinal Basis Property',
                'text': 'The i-th basis is one at xᵢ and zero at every other node. Multiplying Lᵢ by yᵢ '
                        'therefore contributes exactly the correct value at its own node.',
                'formulas': ['L_i(x_j)=\\delta_{ij}=\\begin{cases}1,&i=j\\\\0,&i\\neq j\\end{cases}']},
 'requirements_intro': 'Every denominator contains differences between nodes, so repeated or nearly '
                       'repeated x-values require special care.',
 'conditions': ['All x-values must be distinct.',
                'The x and y arrays must have the same nonzero length.',
                'All data and evaluation values must be finite real numbers.',
                'For repeated evaluation or high degree, a barycentric form is preferred over '
                'repeatedly forming raw products.'],
 'formula': {'title': 'Lagrange Basis Formula',
             'text': 'Each basis multiplies one factor for every node except its own.',
             'formulas': ['L_i(x)=\\prod_{\\substack{j=0\\\\j\\neq i}}^{n}\\frac{x-x_j}{x_i-x_j}',
                          'P_n(x)=\\sum_{i=0}^{n}y_i\\prod_{\\substack{j=0\\\\j\\neq '
                          'i}}^{n}\\frac{x-x_j}{x_i-x_j}'],
             'note': 'Ordinary Lagrange interpolation and barycentric Lagrange interpolation represent '
                     'the same polynomial; the barycentric form is usually better for repeated '
                     'numerical evaluation.'},
 'analysis_intro': 'The method is exact at the nodes, while error between nodes depends on smoothness '
                   'and node placement.',
 'analysis': [{'title': 'Interpolation Error',
               'text': 'For a smooth underlying function, the error has the standard interpolation '
                       'remainder form.',
               'formulas': ['f(x)-P_n(x)=\\frac{f^{(n+1)}(\\xi)}{(n+1)!}\\prod_{i=0}^{n}(x-x_i)'],
               'note': 'The error is exactly zero whenever x equals one of the interpolation nodes.'},
              {'title': 'Numerical Form',
               'text': 'The direct product formula is clear for learning and small problems. For many '
                       'evaluations, precomputed barycentric weights reduce the evaluation cost and '
                       'usually improve numerical behavior.',
               'formulas': ['w_i=\\frac{1}{\\prod_{j\\neq i}(x_i-x_j)}'],
               'note': 'A high-degree global interpolant may still oscillate even when evaluated '
                       'stably.'}],
 'checks_title': 'Interpolation Checks',
 'checks_intro': 'Validate the basis denominators and confirm that the constructed polynomial '
                 'reproduces every input point.',
 'checks': [{'title': 'Denominator Check',
             'formula': 'x_i-x_j\\neq0',
             'caption': 'Prevents division by zero in every basis factor.'},
            {'title': 'Partition of Unity',
             'formula': '\\sum_{i=0}^{n}L_i(x)=1',
             'caption': 'A useful identity for testing the basis evaluation.'},
            {'title': 'Node Reproduction',
             'formula': 'P_n(x_i)=y_i',
             'caption': 'Residuals at all supplied nodes should be near machine roundoff.'}],
 'algorithm_intro': 'Construct and weight one basis polynomial for each node.',
 'algorithm': ['Validate equal-length arrays and distinct x-values.',
               'Initialize the interpolated value to zero.',
               'For each node i, initialize Lᵢ(x) to one.',
               'Multiply all factors (x − xⱼ)/(xᵢ − xⱼ) for j ≠ i.',
               'Add yᵢLᵢ(x) to the accumulated value.',
               'Verify the result at the nodes and report extrapolation when applicable.'],
 'example': {'intro': 'Use three nodes to estimate the value at x = 1.5.',
             'setup': [{'label': 'Interpolation nodes', 'formulas': ['(0,1),\\quad(1,2),\\quad(2,5)']},
                       {'label': 'Evaluation point', 'formulas': ['x=1.5']}],
             'setup_note': 'The three distinct nodes determine a unique polynomial of degree at most '
                           'two.',
             'steps': [{'title': 'Calculate L₀ and L₁',
                        'formulas': ['L_0(1.5)=\\frac{(1.5-1)(1.5-2)}{(0-1)(0-2)}=-0.125',
                                     'L_1(1.5)=\\frac{(1.5-0)(1.5-2)}{(1-0)(1-2)}=0.75'],
                        'text': 'The signs arise from the relative location of the target and the '
                                'nodes.'},
                       {'title': 'Calculate L₂',
                        'formulas': ['L_2(1.5)=\\frac{(1.5-0)(1.5-1)}{(2-0)(2-1)}=0.375'],
                        'text': 'The three basis values sum to one: −0.125 + 0.75 + 0.375 = 1.'},
                       {'title': 'Combine the Contributions',
                        'formulas': ['P_2(1.5)=1(-0.125)+2(0.75)+5(0.375)'],
                        'text': 'Each observed y-value is weighted by its basis value at the target.'},
                       {'title': 'Simplify',
                        'formulas': ['P_2(1.5)=-0.125+1.5+1.875=3.25'],
                        'text': 'The expanded polynomial is x² + 1.'}],
             'result_title': 'Interpolated Value',
             'result_formulas': ['P_2(1.5)=3.25'],
             'result_text': 'The target is inside the node interval, and the polynomial exactly '
                            'reproduces all three supplied values.'},
 'complexity_intro': 'The textbook formula repeats many products; barycentric preprocessing improves '
                     'repeated evaluation.',
 'complexity': [('Direct Evaluation', 'O(n²)'),
                ('Direct Storage', 'O(n)'),
                ('Barycentric Setup', 'O(n²)'),
                ('Barycentric Evaluation', 'O(n)')],
 'complexity_note': {'text': 'For one small problem, the direct formula is sufficient. For many target '
                             'values, compute barycentric weights once and reuse them.',
                     'formulas': ['\\text{direct: }O(n^2),\\qquad \\text{barycentric after setup: '
                                  '}O(n)']},
 'applications_intro': 'Lagrange interpolation is especially useful when a direct formula is needed for '
                       'a small set of exact nodes.',
 'applications': [('Tabulated Data', 'Estimating values between accurate table entries.'),
                  ('Quadrature Derivation', 'Constructing Newton–Cotes integration weights.'),
                  ('Finite Differences', 'Deriving differentiation formulas from local nodes.'),
                  ('Computer Graphics', 'Building curves that pass through control data.'),
                  ('Calibration', 'Converting intermediate instrument readings.'),
                  ('Education', 'Demonstrating basis functions and polynomial uniqueness directly.')],
 'evaluation_intro': 'The formula is direct and elegant, but the basic product form is not the best '
                     'implementation for every numerical setting.',
 'advantages': ['Direct formula with no separate coefficient system.',
                'Exactly matches all distinct data nodes.',
                'Symmetric with respect to the ordering of the nodes.',
                'Easy to evaluate for small interpolation problems.'],
 'limitations': ['The direct formula costs O(n²) for each target value.',
                 'Adding or removing a node changes all basis functions.',
                 'High-degree interpolation may oscillate strongly.',
                 'Raw products can be numerically less stable than barycentric evaluation.']}
PYTHON_CODE = 'import numpy as np\n\n\ndef lagrange_interpolation(x_nodes, y_nodes, x_target):\n    x_nodes = np.asarray(x_nodes, dtype=float)\n    y_nodes = np.asarray(y_nodes, dtype=float)\n\n    if x_nodes.ndim != 1 or y_nodes.ndim != 1:\n        raise ValueError("x and y must be one-dimensional.")\n    if x_nodes.size == 0 or x_nodes.size != y_nodes.size:\n        raise ValueError("x and y must have the same nonzero length.")\n    if not np.all(np.isfinite(x_nodes)) or not np.all(np.isfinite(y_nodes)):\n        raise ValueError("All data values must be finite.")\n    if np.unique(x_nodes).size != x_nodes.size:\n        raise ValueError("All x-values must be distinct.")\n\n    value = 0.0\n\n    for i in range(x_nodes.size):\n        basis = 1.0\n        for j in range(x_nodes.size):\n            if i != j:\n                basis *= (x_target - x_nodes[j]) / (x_nodes[i] - x_nodes[j])\n        value += y_nodes[i] * basis\n\n    return float(value)\n\n\nx = [0, 1, 2]\ny = [1, 2, 5]\nprint("P(1.5) =", lagrange_interpolation(x, y, 1.5))'
MATLAB_CODE = "function value = LagrangeInterpolation(x, y, target)\n\n    x = x(:).';\n    y = y(:).';\n    n = length(x);\n\n    if n == 0 || length(y) ~= n\n        error('x and y must have the same nonzero length.');\n    end\n\n    if length(unique(x)) ~= n\n        error('All x-values must be distinct.');\n    end\n\n    value = 0;\n\n    for i = 1:n\n        basis = 1;\n        for j = 1:n\n            if i ~= j\n                basis = basis * (target - x(j)) / (x(i) - x(j));\n            end\n        end\n        value = value + y(i) * basis;\n    end\nend\n\n% Example:\nx = [0 1 2];\ny = [1 2 5];\nresult = LagrangeInterpolation(x, y, 1.5)"


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
