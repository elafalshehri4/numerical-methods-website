import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Least Squares Method Quiz',
 'label': 'LEAST SQUARES METHOD QUIZ',
 'lesson': 'Least_Squares_Method',
 'solver': 'Least_Squares_Solver',
 'category': 'Curve Fitting',
 'key': 'least_squares',
 'description': 'Test residuals, normal equations, fitted coefficients, error measures, R², and overfitting.',
 'questions': [('What is the primary objective of curve fitting?',
                ['A) To compute matrix determinants.',
                 'B) To find a mathematical function that best represents a set of data points.',
                 'C) To solve differential equations.',
                 'D) To perform numerical integration.'],
                'B'),
               ('Which method is most commonly used for linear curve fitting?',
                ['A) Newton-Raphson Method',
                 'B) Least Squares Method',
                 'C) Bisection Method',
                 'D) Gaussian Quadrature'],
                'B'),
               ('The Least Squares Method minimizes:',
                ['A) The sum of the residuals.',
                 'B) The sum of the squared residuals.',
                 'C) The largest residual only.',
                 'D) The determinant of the coefficient matrix.'],
                'B'),
               ('A residual is defined as:',
                ['A) The predicted value.',
                 'B) The difference between the observed value and the predicted value.',
                 'C) The slope of the fitted curve.',
                 'D) The intercept of the fitted line.'],
                'B'),
               ('In the equation y = a + bx, the parameter b represents:',
                ['A) The intercept.', 'B) The slope.', 'C) The residual.', 'D) The standard deviation.'],
                'B'),
               ('Which quantity measures the goodness of fit of a model?',
                ['A) Jacobian Matrix', 'B) Coefficient of Determination (R²)', 'C) Determinant', 'D) Eigenvalue'],
                'B'),
               ('A value of R² close to 1 indicates:',
                ['A) A poor fit.',
                 'B) A strong fit between the model and the data.',
                 'C) A singular matrix.',
                 'D) Large numerical errors.'],
                'B'),
               ('Polynomial curve fitting is generally used when:',
                ['A) The relationship between variables is nonlinear.',
                 'B) The data is always linear.',
                 'C) The determinant is zero.',
                 'D) Matrix inversion is required.'],
                'A'),
               ('Increasing the polynomial degree excessively may result in:',
                ['A) Underfitting.',
                 'B) Overfitting.',
                 'C) Improved numerical stability.',
                 'D) Reduced computation.'],
                'B'),
               ('For polynomial least squares with design matrix X, the normal equations are:',
                ['A) (XᵀX)a=Xᵀy', 'B) Xa=0', 'C) X²a=y²', 'D) X+a=y'],
                'A'),
               ('Which assumption is generally made about random errors in Least Squares fitting?',
                ['A) Errors are systematic only.',
                 'B) Errors have zero mean.',
                 'C) Errors are always positive.',
                 'D) Errors are ignored.'],
                'B'),
               ('Which of the following is NOT a type of curve fitting model?',
                ['A) Linear', 'B) Polynomial', 'C) Exponential', 'D) Gaussian Elimination'],
                'D'),
               ('The intercept in a linear regression equation represents:',
                ['A) The value of y when x = 0.',
                 'B) The maximum residual.',
                 'C) The slope of the curve.',
                 'D) The variance of the data.'],
                'A'),
               ('Which statement about overfitting is TRUE?',
                ['A) It always improves prediction accuracy.',
                 'B) It fits the training data well but may perform poorly on new data.',
                 'C) It occurs only in linear regression.',
                 'D) It minimizes computational cost.'],
                'B'),
               ('Curve fitting is primarily used to:',
                ['A) Estimate unknown values and model relationships between variables.',
                 'B) Compute eigenvectors.',
                 'C) Solve systems of nonlinear equations.',
                 'D) Evaluate definite integrals.'],
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
