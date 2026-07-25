import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Polynomial Interpolation Quiz',
 'label': 'POLYNOMIAL INTERPOLATION QUIZ',
 'lesson': 'Polynomial_Interpolation',
 'solver': 'Polynomial_Interpolation_Solver',
 'category': 'Interpolation',
 'key': 'polynomial_interpolation',
 'description': 'Test interpolation conditions, Lagrange bases, Newton divided differences, polynomial degree, and '
                'Runge’s phenomenon.',
 'questions': [('The primary objective of interpolation is to:',
                ['A) Estimate values outside the given data range.',
                 'B) Find a function that passes exactly through the given data points.',
                 'C) Minimize the sum of squared errors.',
                 'D) Solve systems of linear equations.'],
                'B'),
               ('Interpolation differs from curve fitting because interpolation:',
                ['A) Does not use polynomials.',
                 'B) Passes exactly through all given data points.',
                 'C) Minimizes residual errors.',
                 'D) Ignores some data points.'],
                'B'),
               ('Which interpolation method constructs the polynomial using basis polynomials?',
                ['A) Newton-Raphson Method',
                 'B) Lagrange Interpolation',
                 'C) Least Squares Method',
                 'D) Gaussian Elimination'],
                'B'),
               ("Newton's Divided Difference method is especially useful because:",
                ['A) It cannot add new data points.',
                 'B) It allows efficient updating when new points are added.',
                 'C) It only works for equally spaced data.',
                 'D) It does not use polynomial functions.'],
                'B'),
               ('Which statement about interpolation is TRUE?',
                ['A) It is mainly used for extrapolation.',
                 'B) It estimates unknown values within the range of known data.',
                 'C) It minimizes squared residuals.',
                 'D) It always produces a linear function.'],
                'B'),
               ('The degree of the interpolation polynomial for n data points is at most:',
                ['A) n', 'B) n + 1', 'C) n - 1', 'D) 2n'],
                'C'),
               ('In Lagrange interpolation, each basis polynomial is:',
                ['A) Equal to 1 at its own data point and 0 at all other data points.',
                 'B) Constant over the interval.',
                 'C) Always linear.',
                 'D) Independent of the given data.'],
                'A'),
               ("Newton's Divided Difference method is based on:",
                ['A) Taylor Series.',
                 'B) Divided difference coefficients.',
                 'C) Fourier Series.',
                 'D) Numerical integration.'],
                'B'),
               ('Which interpolation method is easier to modify when an additional data point is introduced?',
                ['A) Lagrange Interpolation',
                 "B) Newton's Divided Difference",
                 'C) Linear Regression',
                 'D) Gaussian Quadrature'],
                'B'),
               ('Which phenomenon may occur when using a very high-degree interpolation polynomial?',
                ["A) Runge's Phenomenon", 'B) Gibbs Phenomenon', 'C) Aliasing', 'D) Round-off Stability'],
                'A'),
               ('Interpolation assumes that:',
                ['A) The given data points are accurate.',
                 'B) The data always contains random noise.',
                 'C) The function is periodic.',
                 'D) The matrix is singular.'],
                'A'),
               ('Which of the following is NOT an interpolation method?',
                ['A) Lagrange Interpolation',
                 "B) Newton's Divided Difference",
                 'C) Least Squares Method',
                 'D) Newton Interpolation Polynomial'],
                'C'),
               ('The divided difference table is primarily associated with:',
                ['A) Gaussian Elimination',
                 "B) Newton's Divided Difference",
                 "C) Simpson's Rule",
                 'D) Secant Method'],
                'B'),
               ('Which statement correctly compares Lagrange and Newton interpolation?',
                ['A) Both always require solving systems of equations.',
                 "B) Newton's method is easier to extend when new data becomes available.",
                 'C) Lagrange interpolation cannot interpolate more than three points.',
                 'D) Newton interpolation minimizes squared errors.'],
                'B'),
               ('Interpolation is commonly used to:',
                ['A) Estimate unknown values between measured data points.',
                 'B) Solve nonlinear equations.',
                 'C) Compute definite integrals.',
                 'D) Find matrix inverses.'],
                'A')]}

st.set_page_config(
    page_title=f"{DATA['title']} | Numerical Methods",
    page_icon="📝",
    layout="wide",
)

load_css()
navbar(active_page="quizzes")

st.markdown(
    """
    <style>
    .quiz-page-note { color:#64748b; font-size:14px; line-height:1.6; margin:-6px 0 18px; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius:18px!important;
        border:1px solid rgba(15,61,62,.10)!important;
        box-shadow:0 10px 24px rgba(15,61,62,.06)!important;
    }
    div[role="radiogroup"] { gap:.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.html(
    f"""
    <section class="quiz-hero">
        <div>
            <div class="page-label">{html.escape(DATA['label'])}</div>
            <h1>{html.escape(DATA['title'])}</h1>
            <p>{html.escape(DATA['description'])}</p>
            <div class="method-actions">
                <a href="/{DATA['lesson']}" target="_self" class="btn-outline-ui">Review Lesson →</a>
                <a href="/{DATA['solver']}" target="_self" class="btn-primary-ui">Open Solver →</a>
            </div>
        </div>
    </section>
    """
)


def student_level(score: int, total_questions: int) -> str:
    percentage = score / total_questions if total_questions else 0
    if percentage == 1:
        return "Excellent"
    if percentage >= 0.8:
        return "Very Good"
    if percentage >= 0.6:
        return "Good"
    if percentage >= 0.4:
        return "Acceptable"
    return "Needs Review"


left_margin, quiz_area, right_margin = st.columns([0.035, 0.93, 0.035])

with quiz_area:
    st.html(
        """
        <div class="section-header quiz-header-row">
            <div>
                <h2>Answer the Questions</h2>
                <p>Select one answer for every question, then submit the quiz.</p>
            </div>
        </div>
        """
    )
    st.markdown(
        '<p class="quiz-page-note">No option is selected automatically. '
        'All 15 questions must be answered before grading.</p>',
        unsafe_allow_html=True,
    )

    user_answers = []
    for question_index, (question, options, answer_key) in enumerate(DATA["questions"]):
        with st.container(border=True):
            st.html(
                f"""
                <div class="quiz-question-title">
                    <span>Question {question_index + 1} of {len(DATA['questions'])}</span>
                    <h3>{html.escape(question)}</h3>
                </div>
                """
            )
            selected = st.radio(
                "Choose your answer:",
                options,
                index=None,
                key=f"{DATA['key']}_question_{question_index}",
                label_visibility="collapsed",
            )
            user_answers.append(selected[0] if selected else None)

    submit_quiz = st.button(
        "Submit Quiz",
        type="primary",
        use_container_width=True,
        key=f"{DATA['key']}_submit",
    )

    if submit_quiz:
        unanswered = sum(answer is None for answer in user_answers)
        if unanswered:
            st.warning(f"Please answer all questions. {unanswered} question(s) remain.")
        else:
            score = sum(
                user_answers[index] == question_data[2]
                for index, question_data in enumerate(DATA["questions"])
            )
            level = student_level(score, len(DATA["questions"]))
            percentage = round(100 * score / len(DATA["questions"]))

            st.html(
                f"""
                <div class="quiz-result-card">
                    <span>Your Score</span>
                    <strong>{score} / {len(DATA['questions'])}</strong>
                    <p>{level} • {percentage}%</p>
                </div>
                """
            )

            if score == len(DATA["questions"]):
                st.balloons()

            with st.expander("Review Answers", expanded=False):
                for question_index, (question, options, answer_key) in enumerate(DATA["questions"]):
                    correct_option = next(option for option in options if option.startswith(answer_key))
                    user_option = next(option for option in options if option.startswith(user_answers[question_index]))
                    st.markdown(f"### Question {question_index + 1}")
                    st.write(question)
                    st.markdown(f"**Your answer:** {user_option}")
                    st.markdown(f"**Correct answer:** {correct_option}")
                    if user_answers[question_index] == answer_key:
                        st.success("Correct")
                    else:
                        st.error("Incorrect")
                    st.divider()

    with st.container(border=True):
        st.subheader("Continue Learning")
        left_navigation, right_navigation = st.columns(2)
        with left_navigation:
            if st.button("Review Lesson", use_container_width=True, key=f"{DATA['key']}_review_lesson"):
                st.switch_page(f"pages/{DATA['lesson']}.py")
        with right_navigation:
            if st.button("Back to Quizzes", use_container_width=True, key=f"{DATA['key']}_back_quizzes"):
                st.switch_page("pages/Quizzes.py")

st.html(
    f"""
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>{html.escape(DATA['title'])} • {html.escape(DATA['category'])}</div>
    </footer>
    """
)
