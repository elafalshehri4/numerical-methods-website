
import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Systems of Nonlinear Equations',
 'label': 'MULTIVARIABLE ROOT-FINDING METHOD',
 'solver': 'Systems_of_Nonlinear_Equations_Solver',
 'quiz': 'Systems_of_Nonlinear_Equations_Quiz',
 'hero': "Solve several coupled nonlinear equations simultaneously with Newton's method and the "
         'Jacobian matrix.',
 'summary': [('PROBLEM FORM', 'F(x)=0'),
             ('CORE MATRIX', 'Jacobian J(x)'),
             ('UPDATE', 'Solve JΔ=-F'),
             ('CONVERGENCE', 'Usually quadratic near a simple root')],
 'overview': 'A nonlinear system contains two or more equations whose variables are coupled and '
             'appear nonlinearly. The goal is to find a vector x* for which every component of the '
             'residual vector F(x*) is zero.',
 'foundation': "Multivariable Newton's method replaces the nonlinear system near the current guess "
               'by its first-order Taylor model. Solving the resulting linear system produces a '
               'correction vector.',
 'foundation_formula': 'F(x_k+\\Delta x)\\approx F(x_k)+J(x_k)\\Delta x,\\qquad J(x_k)\\Delta '
                       'x=-F(x_k)',
 'conditions': ['The number of equations should match the number of unknowns for a square Newton '
                'system.',
                'All required first partial derivatives must exist near the solution.',
                'The Jacobian must be nonsingular or sufficiently well-conditioned at each '
                'successful iteration.',
                'The initial guess should lie in the basin of attraction of the desired root.'],
 'formula_title': 'Newton Vector Update',
 'formula': 'x_{k+1}=x_k+\\Delta x_k,\\qquad J(x_k)\\Delta x_k=-F(x_k)',
 'formula_note': 'The correction is found by solving a linear system; explicitly forming J⁻¹ is '
                 'unnecessary and less stable.',
 'accuracy_title': 'Local Convergence',
 'accuracy_formula': '\\|e_{k+1}\\|\\le C\\|e_k\\|^2\\quad\\text{near a simple root}',
 'accuracy_text': 'With a sufficiently close initial guess and a nonsingular Jacobian at the root, '
                  "Newton's method is locally quadratically convergent.",
 'comparison_title': 'Practical Behavior',
 'comparison_text': 'Far from a root, the method may diverge, converge to a different root, or '
                    'encounter a singular Jacobian. Damping or line search can improve robustness.',
 'steps': ['Write the residual vector F(x).',
           'Choose an initial vector x₀.',
           'Evaluate F(xₖ) and the Jacobian J(xₖ).',
           'Solve J(xₖ)Δxₖ=-F(xₖ).',
           'Update xₖ₊₁=xₖ+Δxₖ.',
           'Stop when the residual and/or correction norm is below tolerance.'],
 'example_title': 'Two-Equation Newton Step',
 'example_setup': 'F(x,y)=\\begin{bmatrix}x^2+y^2-4\\\\x-y\\end{bmatrix},\\qquad '
                  '(x_0,y_0)=(1.5,1.0)',
 'example_lines': ['J(x,y)=\\begin{bmatrix}2x&2y\\\\1&-1\\end{bmatrix}',
                   'F(1.5,1)=\\begin{bmatrix}-0.75\\\\0.5\\end{bmatrix}',
                   '\\begin{bmatrix}3&2\\\\1&-1\\end{bmatrix}\\begin{bmatrix}\\Delta x\\\\\\Delta '
                   'y\\end{bmatrix}=\\begin{bmatrix}0.75\\\\-0.5\\end{bmatrix}',
                   '(\\Delta x,\\Delta y)=(-0.05,0.45),\\qquad (x_1,y_1)=(1.45,1.45)'],
 'question': 'Why should the Jacobian inverse not be formed explicitly in a numerical '
             'implementation?',
 'options': ['Solving the linear system is usually more stable and efficient.',
             'The Jacobian contains no derivatives.',
             'An inverse never exists for nonlinear systems.'],
 'answer': 'Solving the linear system is usually more stable and efficient.',
 'answer_explanation': 'Newton only needs the correction Δ from JΔ=-F; a linear solver obtains it '
                       'without explicitly computing J⁻¹.',
 'python': 'import numpy as np\n'
           '\n'
           'def newton_system(F, J, x0, tolerance=1e-8, max_iterations=50):\n'
           '    x = np.asarray(x0, dtype=float)\n'
           '\n'
           '    for iteration in range(max_iterations):\n'
           '        residual = np.asarray(F(x), dtype=float)\n'
           '        if np.linalg.norm(residual, ord=np.inf) <= tolerance:\n'
           '            return x\n'
           '\n'
           '        jacobian = np.asarray(J(x), dtype=float)\n'
           '        correction = np.linalg.solve(jacobian, -residual)\n'
           '        x = x + correction\n'
           '\n'
           '        if np.linalg.norm(correction, ord=np.inf) <= tolerance:\n'
           '            return x\n'
           '\n'
           '    raise RuntimeError("Maximum iterations reached.")',
 'matlab': 'function x = NewtonSystem(F,J,x0,tol,maxIter)\n'
           'x = x0(:);\n'
           '\n'
           'for k = 1:maxIter\n'
           '    residual = F(x);\n'
           '    if norm(residual,inf) <= tol\n'
           '        return\n'
           '    end\n'
           '\n'
           '    correction = J(x)\\(-residual);\n'
           '    x = x + correction;\n'
           '\n'
           '    if norm(correction,inf) <= tol\n'
           '        return\n'
           '    end\n'
           'end\n'
           '\n'
           "error('Maximum iterations reached.')\n"
           'end',
 'applications': [('Chemical Equilibria', 'Solve coupled nonlinear concentration constraints.'),
                  ('Circuit Analysis', 'Find operating points in nonlinear electronic networks.'),
                  ('Engineering Design',
                   'Satisfy several nonlinear geometry or balance equations simultaneously.')],
 'advantages': ['Quadratic local convergence near a simple root.',
                'Handles strongly coupled variables.',
                'Uses standard linear algebra at each iteration.',
                'Extends the familiar scalar Newton method.'],
 'limitations': ['Requires a Jacobian or a reliable approximation.',
                 'Sensitive to the initial guess.',
                 'A singular or ill-conditioned Jacobian can stop the method.',
                 'May converge to an unintended root without safeguards.'],
 'footer': 'Systems of Nonlinear Equations • Root Finding'}

st.set_page_config(
    page_title=f"{DATA['title']} | Numerical Methods",
    page_icon="📘",
    layout="wide",
)

load_css()
navbar(active_page="learn")

st.markdown('\n<style>\n.method-page { padding-top: 26px; padding-bottom: 12px; }\n.section-kicker { color:#0f766e; font-size:12px; font-weight:900; letter-spacing:1.2px; text-transform:uppercase; margin:26px 0 5px; }\n.section-title { color:#0b1b3a; font-size:25px; font-weight:900; margin:0 0 7px; }\n.section-intro { color:#475569; font-size:15px; line-height:1.65; margin:0 0 18px; }\n.summary-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:4px 0 22px; }\n.summary-card { min-height:112px; padding:18px; border-radius:16px; border:1px solid rgba(15,61,62,.10); box-shadow:0 8px 20px rgba(15,61,62,.06); }\n.summary-card:nth-child(1){background:linear-gradient(135deg,#f0fdfa,#ecfeff);}\n.summary-card:nth-child(2){background:linear-gradient(135deg,#eff6ff,#e0f2fe);}\n.summary-card:nth-child(3){background:linear-gradient(135deg,#f5f3ff,#faf5ff);}\n.summary-card:nth-child(4){background:linear-gradient(135deg,#fff7ed,#fffbeb);}\n.summary-card span{display:block;color:#64748b;font-size:12px;font-weight:800;margin-bottom:8px;}\n.summary-card strong{display:block;color:#0b1b3a;font-size:18px;font-weight:900;line-height:1.25;}\n.condition-list{display:grid;gap:12px;margin-top:10px;}\n.condition-item{display:flex;gap:12px;align-items:flex-start;padding:14px 16px;border-radius:14px;background:#f8fafc;border:1px solid #e2e8f0;}\n.condition-number{width:28px;height:28px;min-width:28px;border-radius:50%;background:linear-gradient(135deg,#14b8a6,#0f766e);color:white;display:grid;place-items:center;font-size:13px;font-weight:900;}\n.condition-item div:last-child{color:#334155;font-size:14px;line-height:1.6;}\n.algorithm-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:8px;}\n.algorithm-step{background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:15px 16px;}\n.algorithm-step span{display:inline-block;color:#0f766e;font-size:12px;font-weight:900;margin-bottom:6px;}\n.algorithm-step p{color:#334155;font-size:14px;line-height:1.55;margin:0;}\n.application-grid-advanced{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}\n.application-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:15px;padding:18px;color:#334155;font-size:14px;line-height:1.55;min-height:94px;}\n.application-box strong{display:block;color:#0b1b3a;margin-bottom:5px;font-size:14px;}\n.advantage-box,.limitation-box{border-radius:16px;padding:20px;min-height:215px;}\n.advantage-box{background:#f0fdfa;border:1px solid #99f6e4;}\n.limitation-box{background:#fff7ed;border:1px solid #fed7aa;}\n.advantage-box h3,.limitation-box h3{color:#0b1b3a;font-size:18px;font-weight:900;margin:0 0 10px;}\n.advantage-box li,.limitation-box li{color:#475569;margin-bottom:8px;line-height:1.5;}\ndiv[data-testid="stVerticalBlockBorderWrapper"]{border-radius:18px!important;border:1px solid rgba(15,61,62,.10)!important;box-shadow:0 10px 24px rgba(15,61,62,.06)!important;}\ndiv[data-testid="stExpander"]{border-radius:14px!important;border-color:rgba(15,61,62,.12)!important;overflow:hidden!important;}\n@media(max-width:1000px){.summary-grid,.application-grid-advanced,.algorithm-grid{grid-template-columns:1fr;}}\n</style>\n', unsafe_allow_html=True)

st.html(
    f"""
    <section class="method-hero">
        <div>
            <div class="page-label">{html.escape(DATA['label'])}</div>
            <h1>{html.escape(DATA['title'])}</h1>
            <p>{html.escape(DATA['hero'])}</p>
            <div class="method-actions">
                <a href="/{DATA['solver']}" target="_self" class="btn-primary-ui">Open Solver →</a>
                <a href="/{DATA['quiz']}" target="_self" class="btn-outline-ui">Take Quiz →</a>
            </div>
        </div>
    </section>
    """
)

left_margin, content, right_margin = st.columns([0.035, 0.93, 0.035])

with content:
    st.markdown('<div class="method-page">', unsafe_allow_html=True)

    summary_cards = "".join(
        f'<div class="summary-card"><span>{html.escape(label)}</span>'
        f'<strong>{html.escape(value)}</strong></div>'
        for label, value in DATA["summary"]
    )
    st.html(
        f"""
        <div class="section-kicker">Quick reference</div>
        <h2 class="section-title">Method at a Glance</h2>
        <p class="section-intro">Review the essential properties before studying the derivation and algorithm.</p>
        <div class="summary-grid">{summary_cards}</div>
        """
    )

    st.html(
        """
        <div class="section-kicker">Core concept</div>
        <h2 class="section-title">Overview and Mathematical Foundation</h2>
        <p class="section-intro">Connect the intuitive idea to the mathematical approximation.</p>
        """
    )
    overview_col, foundation_col = st.columns(2)
    with overview_col:
        with st.container(border=True):
            st.subheader("Overview")
            st.write(DATA["overview"])
    with foundation_col:
        with st.container(border=True):
            st.subheader("Mathematical Foundation")
            st.write(DATA["foundation"])
            st.latex(DATA["foundation_formula"])

    st.html(
        """
        <div class="section-kicker">Requirements</div>
        <h2 class="section-title">Conditions and Core Formula</h2>
        <p class="section-intro">Use the method only when its mathematical and data requirements are satisfied.</p>
        """
    )
    conditions_col, formula_col = st.columns([1.15, 0.85])
    with conditions_col:
        with st.container(border=True):
            st.subheader("Required Conditions")
            condition_html = "".join(
                f'<div class="condition-item"><div class="condition-number">{index}</div>'
                f'<div>{html.escape(condition)}</div></div>'
                for index, condition in enumerate(DATA["conditions"], start=1)
            )
            st.html(f'<div class="condition-list">{condition_html}</div>')
    with formula_col:
        with st.container(border=True):
            st.subheader(DATA["formula_title"])
            st.latex(DATA["formula"])
            st.info(DATA["formula_note"])

    st.html(
        """
        <div class="section-kicker">Accuracy</div>
        <h2 class="section-title">Error and Method Selection</h2>
        <p class="section-intro">The order of accuracy explains how the approximation improves as the step or spacing is refined.</p>
        """
    )
    accuracy_col, comparison_col = st.columns(2)
    with accuracy_col:
        with st.container(border=True):
            st.subheader(DATA["accuracy_title"])
            st.latex(DATA["accuracy_formula"])
            st.write(DATA["accuracy_text"])
    with comparison_col:
        with st.container(border=True):
            st.subheader(DATA["comparison_title"])
            st.write(DATA["comparison_text"])

    st.html(
        """
        <div class="section-kicker">Procedure</div>
        <h2 class="section-title">Algorithm</h2>
        <p class="section-intro">Follow the steps in order and validate the input before performing the calculation.</p>
        """
    )
    algorithm_html = "".join(
        f'<div class="algorithm-step"><span>STEP {index}</span><p>{html.escape(step)}</p></div>'
        for index, step in enumerate(DATA["steps"], start=1)
    )
    st.html(f'<div class="algorithm-grid">{algorithm_html}</div>')

    st.html(
        """
        <div class="section-kicker">Worked example</div>
        <h2 class="section-title">Numerical Example</h2>
        <p class="section-intro">The example applies the formula carefully and checks the meaning of the result.</p>
        """
    )
    with st.container(border=True):
        st.markdown(f"**{DATA['example_title']}**")
        st.latex(DATA["example_setup"])
        for line in DATA["example_lines"]:
            st.latex(line)

        st.divider()
        prediction = st.radio(
            DATA["question"],
            DATA["options"],
            index=None,
            key=f"{DATA['solver']}_lesson_prediction",
        )
        reveal = st.checkbox(
            "Reveal answer",
            key=f"{DATA['solver']}_lesson_reveal",
        )
        if reveal:
            if prediction is None:
                st.info("Select an answer first, then compare it with the explanation.")
            elif prediction == DATA["answer"]:
                st.success("Correct.")
            else:
                st.warning("Review the explanation below.")
            st.markdown(f"**Correct answer:** {DATA['answer']}")
            st.write(DATA["answer_explanation"])

    st.html(
        """
        <div class="section-kicker">Programming</div>
        <h2 class="section-title">Implementation</h2>
        <p class="section-intro">Expand either language to examine a complete, validated implementation.</p>
        """
    )
    with st.container(border=True):
        python_col, matlab_col = st.columns(2)
        with python_col:
            with st.expander("🐍 Python Implementation", expanded=False):
                st.code(DATA["python"], language="python")
        with matlab_col:
            with st.expander("🟠 MATLAB Implementation", expanded=False):
                st.code(DATA["matlab"], language="matlab")

    st.html(
        """
        <div class="section-kicker">Use cases</div>
        <h2 class="section-title">Applications</h2>
        <p class="section-intro">These examples show where the method is commonly useful.</p>
        """
    )
    application_html = "".join(
        f'<div class="application-box"><strong>{html.escape(title)}</strong>{html.escape(text)}</div>'
        for title, text in DATA["applications"]
    )
    st.html(f'<div class="application-grid-advanced">{application_html}</div>')

    st.html(
        """
        <div class="section-kicker">Evaluation</div>
        <h2 class="section-title">Advantages and Limitations</h2>
        <p class="section-intro">Choose a numerical method by balancing accuracy, cost, assumptions, and stability.</p>
        """
    )
    advantages_col, limitations_col = st.columns(2)
    with advantages_col:
        advantage_items = "".join(f"<li>{html.escape(item)}</li>" for item in DATA["advantages"])
        st.html(f'<div class="advantage-box"><h3>Advantages</h3><ul>{advantage_items}</ul></div>')
    with limitations_col:
        limitation_items = "".join(f"<li>{html.escape(item)}</li>" for item in DATA["limitations"])
        st.html(f'<div class="limitation-box"><h3>Limitations</h3><ul>{limitation_items}</ul></div>')

    with st.container(border=True):
        st.subheader("Continue Learning")
        navigation_left, navigation_right = st.columns(2)
        with navigation_left:
            if st.button("Open Interactive Solver", use_container_width=True):
                st.switch_page(f"pages/{DATA['solver']}.py")
        with navigation_right:
            if st.button("Take the Quiz", use_container_width=True):
                st.switch_page(f"pages/{DATA['quiz']}.py")

    st.markdown("</div>", unsafe_allow_html=True)

st.html(
    f"""
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>{html.escape(DATA['footer'])}</div>
    </footer>
    """
)
