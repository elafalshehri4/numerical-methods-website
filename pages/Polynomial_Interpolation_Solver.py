from __future__ import annotations

import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css


st.set_page_config(
    page_title="Polynomial Interpolation Solvers",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
load_css()
navbar(active_page="solver")

st.markdown(
    """
    <main class="page-shell-ui">
        <section class="hero-ui">
            <div class="eyebrow-ui">INTERPOLATION SOLVER GUIDE</div>
            <h1>Polynomial Interpolation</h1>
            <p>
                Lagrange interpolation and Newton divided differences construct
                the same unique polynomial from distinct nodes, but they use
                different representations and calculation steps.
            </p>
        </section>
    </main>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Choose the specialized solver that matches the form you want to study. "
    "Both should return the same interpolated value for the same valid data, "
    "apart from floating-point roundoff."
)

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Lagrange Interpolation")
        st.write(
            "Build and inspect the Lagrange basis polynomials, barycentric "
            "weights, expanded coefficients, node residuals, graphs, and Excel report."
        )
        if st.button(
            "Open Lagrange Solver",
            use_container_width=True,
            type="primary",
            key="open_lagrange_from_polynomial_guide",
        ):
            st.switch_page("pages/Lagrange_Interpolation_Solver.py")

with right:
    with st.container(border=True):
        st.subheader("Newton Divided Differences")
        st.write(
            "Construct the divided-difference table and Newton coefficients, "
            "then evaluate the nested polynomial with its graphs and Excel report."
        )
        if st.button(
            "Open Newton Solver",
            use_container_width=True,
            type="primary",
            key="open_newton_from_polynomial_guide",
        ):
            st.switch_page("pages/Newton_Divided_Differences_Solver.py")

st.divider()
nav_left, nav_right = st.columns(2)
with nav_left:
    if st.button(
        "Review Polynomial Interpolation Lesson",
        use_container_width=True,
        key="review_polynomial_interpolation_overview",
    ):
        st.switch_page("pages/Polynomial_Interpolation.py")
with nav_right:
    if st.button(
        "Back to Solver Menu",
        use_container_width=True,
        key="back_from_polynomial_solver_guide",
    ):
        st.switch_page("pages/Numerical_Solver.py")

st.html(
    """
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>Polynomial Interpolation • Solver Guide</div>
    </footer>
    """
)
