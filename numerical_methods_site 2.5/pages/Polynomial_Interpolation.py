from __future__ import annotations

import html

import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css


DATA = {'title': 'Polynomial Interpolation',
 'page_label': 'INTERPOLATION OVERVIEW',
 'hero_text': 'Learn the common mathematical foundation of Lagrange interpolation and Newton divided '
              'differences, including uniqueness, error behavior, and reliable node selection.',
 'solver_page': 'Polynomial_Interpolation_Solver',
 'quiz_page': 'Polynomial_Interpolation_Quiz',
 'footer_text': 'Polynomial Interpolation • Lagrange and Newton Forms',
 'summary_intro': 'Polynomial interpolation constructs one polynomial that reproduces every supplied '
                  'data value exactly.',
 'summary': [('INPUT', 'n + 1 distinct data points'),
             ('OUTPUT', 'Unique polynomial of degree ≤ n'),
             ('NODE PROPERTY', 'P(xᵢ) = yᵢ exactly'),
             ('COMMON FORMS', 'Lagrange and Newton')],
 'core_intro': 'Different interpolation formulas represent the same unique polynomial when they use the '
               'same distinct nodes.',
 'overview': {'title': 'Overview',
              'text': 'For n + 1 points with distinct x-coordinates, polynomial interpolation '
                      'constructs a polynomial of degree at most n that passes through all points.',
              'formulas': ['P_n(x_i)=y_i,\\qquad i=0,1,\\ldots,n']},
 'foundation': {'title': 'Uniqueness',
                'text': 'If two degree-n polynomials matched the same n + 1 distinct nodes, their '
                        'difference would have n + 1 roots while having degree at most n. The '
                        'difference must therefore be the zero polynomial.',
                'formulas': ['P_n(x)\\ \\text{is unique when}\\ x_i\\neq x_j\\ \\text{for}\\ i\\neq j']},
 'requirements_intro': 'Distinct nodes are essential, and the location and spacing of the nodes affect '
                       'numerical quality.',
 'conditions': ['All x-values must be distinct.',
                'The x and y lists must contain the same number of finite values.',
                'Interpolation is usually more reliable inside the node interval than outside it.',
                'High-degree global interpolation should use carefully selected nodes or a piecewise '
                'alternative.'],
 'formula': {'title': 'Two Equivalent Forms',
             'text': 'Lagrange uses cardinal basis polynomials, while Newton uses recursively '
                     'calculated divided differences.',
             'formulas': ['P_n(x)=\\sum_{i=0}^{n}y_iL_i(x)',
                          'P_n(x)=f[x_0]+f[x_0,x_1](x-x_0)+\\cdots+f[x_0,\\ldots,x_n]\\prod_{j=0}^{n-1}(x-x_j)'],
             'note': 'For the same nodes and values, both formulas produce the same polynomial, apart '
                     'from floating-point roundoff.'},
 'analysis_intro': 'Exact matching at the nodes does not guarantee small error between nodes or stable '
                   'high-degree behavior.',
 'analysis': [{'title': 'Interpolation Error',
               'text': 'If the underlying function has n + 1 continuous derivatives, the error at x '
                       'depends on the next derivative and the product of distances from the nodes.',
               'formulas': ['f(x)-P_n(x)=\\frac{f^{(n+1)}(\\xi)}{(n+1)!}\\prod_{i=0}^{n}(x-x_i)'],
               'note': 'The point ξ lies somewhere in the interval containing x and the interpolation '
                       'nodes.'},
              {'title': 'Runge Phenomenon and Node Choice',
               'text': 'Increasing the degree with equally spaced nodes can create large oscillations '
                       'near the endpoints. Chebyshev-like nodes or piecewise polynomials often behave '
                       'better.',
               'formulas': ['x_k=\\cos\\left(\\frac{(2k+1)\\pi}{2(n+1)}\\right)\\quad\\text{(scaled '
                            'Chebyshev nodes)}'],
               'note': 'More points do not automatically mean a better global interpolant.'}],
 'checks_title': 'Interpolation Checks',
 'checks_intro': 'A correct implementation should verify the nodes, the reconstruction, and whether the '
                 'requested point is interpolation or extrapolation.',
 'checks': [{'title': 'Distinct Nodes',
             'formula': 'x_i\\neq x_j',
             'caption': 'Repeated nodes make ordinary interpolation undefined.'},
            {'title': 'Node Reproduction',
             'formula': 'P_n(x_i)=y_i',
             'caption': 'Residuals at the supplied nodes should be near roundoff.'},
            {'title': 'Range Check',
             'formula': 'x\\in[\\min x_i,\\max x_i]',
             'caption': 'Outside this interval, the calculation is extrapolation.'}],
 'algorithm_intro': 'Choose a representation, construct its coefficients, evaluate the polynomial, and '
                    'verify the result.',
 'algorithm': ['Collect n + 1 points and verify that all x-values are distinct.',
               'Choose the Lagrange form or the Newton divided-difference form.',
               'Construct the basis polynomials or the divided-difference table.',
               'Assemble the interpolation polynomial.',
               'Evaluate the polynomial at the requested x-value.',
               'Check node residuals, conditioning, and whether the request is extrapolation.'],
 'example': {'intro': 'Interpolate three values generated by the quadratic function x² + 1.',
             'setup': [{'label': 'Interpolation nodes', 'formulas': ['(0,1),\\quad(1,2),\\quad(2,5)']},
                       {'label': 'Evaluation point', 'formulas': ['x=1.5']}],
             'setup_note': 'Three distinct nodes determine one polynomial of degree at most two.',
             'steps': [{'title': 'Lagrange Representation',
                        'formulas': ['P_2(x)=1L_0(x)+2L_1(x)+5L_2(x)',
                                     'L_0(1.5)=-0.125,\\quad L_1(1.5)=0.75,\\quad L_2(1.5)=0.375'],
                        'text': 'The basis values add to one and weight the three data values.'},
                       {'title': 'Newton Representation',
                        'formulas': ['f[x_0]=1,\\quad f[x_0,x_1]=1,\\quad f[x_0,x_1,x_2]=1',
                                     'P_2(x)=1+x+x(x-1)'],
                        'text': 'The divided-difference coefficients produce the same polynomial.'},
                       {'title': 'Expanded Polynomial',
                        'formulas': ['P_2(x)=x^2+1'],
                        'text': 'The unique degree-two polynomial reproduces all three nodes exactly.'},
                       {'title': 'Evaluate',
                        'formulas': ['P_2(1.5)=1.5^2+1=3.25'],
                        'text': 'The target lies inside [0, 2], so this is interpolation.'}],
             'result_title': 'Interpolated Value',
             'result_formulas': ['P_2(1.5)=3.25'],
             'result_text': 'Lagrange and Newton divided differences return the same value because they '
                            'represent the same unique polynomial.'},
 'complexity_intro': 'The cost depends on the chosen representation and on whether one or many '
                     'evaluations are required.',
 'complexity': [('Lagrange Evaluation', 'O(n²)'),
                ('Newton Table', 'O(n²)'),
                ('Newton Evaluation', 'O(n)'),
                ('Polynomial Degree', '≤ n')],
 'complexity_note': {'text': 'Newton form is convenient when evaluating the same polynomial many times '
                             'or adding nodes incrementally; barycentric Lagrange is also efficient for '
                             'repeated evaluation.',
                     'formulas': ['\\text{construction}=O(n^2),\\qquad \\text{Newton evaluation}=O(n)']},
 'applications_intro': 'Interpolation is appropriate when the supplied values are treated as exact and '
                       'intermediate values are required.',
 'applications': [('Property Tables',
                   'Estimating thermodynamic or material values between tabulated entries.'),
                  ('Calibration Tables', 'Converting between measured and reference values.'),
                  ('Computer Graphics', 'Constructing curves through specified control data.'),
                  ('Numerical Integration', 'Deriving quadrature rules from interpolation polynomials.'),
                  ('Numerical Differentiation',
                   'Deriving finite-difference formulas from local interpolants.'),
                  ('Scientific Data', 'Estimating values between accurate sampled observations.')],
 'evaluation_intro': 'Interpolation exactly respects the data, but global high-degree polynomials can '
                     'be fragile.',
 'advantages': ['Produces an exact match at every supplied node.',
                'Has a unique solution for distinct x-values.',
                'Offers several mathematically equivalent forms.',
                'Provides a theoretical error formula for smooth functions.'],
 'limitations': ['Sensitive to noisy data because every point is matched exactly.',
                 'High-degree global polynomials may oscillate strongly.',
                 'Extrapolation can be unreliable.',
                 'Poorly spaced nodes can make coefficient calculations ill-conditioned.']}
PYTHON_CODE = 'import numpy as np\n\n\ndef lagrange_value(x_nodes, y_nodes, x_target):\n    x_nodes = np.asarray(x_nodes, dtype=float)\n    y_nodes = np.asarray(y_nodes, dtype=float)\n\n    if x_nodes.size != y_nodes.size or x_nodes.size == 0:\n        raise ValueError("x and y must have the same nonzero length.")\n    if np.unique(x_nodes).size != x_nodes.size:\n        raise ValueError("All x-values must be distinct.")\n\n    value = 0.0\n    for i in range(x_nodes.size):\n        basis = 1.0\n        for j in range(x_nodes.size):\n            if i != j:\n                basis *= (x_target - x_nodes[j]) / (x_nodes[i] - x_nodes[j])\n        value += y_nodes[i] * basis\n    return value\n\n\ndef newton_coefficients(x_nodes, y_nodes):\n    x_nodes = np.asarray(x_nodes, dtype=float)\n    coefficients = np.asarray(y_nodes, dtype=float).copy()\n    n = x_nodes.size\n\n    if coefficients.size != n or np.unique(x_nodes).size != n:\n        raise ValueError("The nodes must have equal lengths and distinct x-values.")\n\n    for order in range(1, n):\n        coefficients[order:n] = (\n            coefficients[order:n] - coefficients[order - 1:n - 1]\n        ) / (x_nodes[order:n] - x_nodes[:n - order])\n\n    return coefficients\n\n\ndef evaluate_newton(x_nodes, coefficients, x_target):\n    value = coefficients[-1]\n    for i in range(len(coefficients) - 2, -1, -1):\n        value = coefficients[i] + (x_target - x_nodes[i]) * value\n    return value\n\n\nx = np.array([0.0, 1.0, 2.0])\ny = np.array([1.0, 2.0, 5.0])\ntarget = 1.5\n\nprint("Lagrange:", lagrange_value(x, y, target))\ncoefficients = newton_coefficients(x, y)\nprint("Newton:", evaluate_newton(x, coefficients, target))'
MATLAB_CODE = "function PolynomialInterpolationExample()\n\n    x = [0 1 2];\n    y = [1 2 5];\n    target = 1.5;\n\n    lagrangeResult = LagrangeValue(x, y, target);\n    coefficients = NewtonCoefficients(x, y);\n    newtonResult = EvaluateNewton(x, coefficients, target);\n\n    fprintf('Lagrange value: %.8f\\n', lagrangeResult);\n    fprintf('Newton value:   %.8f\\n', newtonResult);\nend\n\nfunction value = LagrangeValue(x, y, target)\n    n = length(x);\n    if length(unique(x)) ~= n\n        error('All x-values must be distinct.');\n    end\n\n    value = 0;\n    for i = 1:n\n        basis = 1;\n        for j = 1:n\n            if i ~= j\n                basis = basis * (target - x(j)) / (x(i) - x(j));\n            end\n        end\n        value = value + y(i) * basis;\n    end\nend\n\nfunction coefficients = NewtonCoefficients(x, y)\n    n = length(x);\n    coefficients = y(:).';\n\n    if length(unique(x)) ~= n\n        error('All x-values must be distinct.');\n    end\n\n    for order = 2:n\n        for i = n:-1:order\n            coefficients(i) = (coefficients(i) - coefficients(i - 1)) ...\n                / (x(i) - x(i - order + 1));\n        end\n    end\nend\n\nfunction value = EvaluateNewton(x, coefficients, target)\n    value = coefficients(end);\n    for i = length(coefficients)-1:-1:1\n        value = coefficients(i) + (target - x(i)) * value;\n    end\nend"


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
