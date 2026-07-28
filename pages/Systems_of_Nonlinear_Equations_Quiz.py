import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Systems of Nonlinear Equations Quiz',
 'label': 'NONLINEAR SYSTEMS QUIZ',
 'lesson': 'Systems_of_Nonlinear_Equations',
 'solver': 'Systems_of_Nonlinear_Equations_Solver',
 'category': 'Nonlinear Systems',
 'key': 'nonlinear_systems',
 'description': 'Test residual vectors, Jacobians, Newton corrections, convergence, damping, and initial-guess '
                'sensitivity.',
 'questions': [('A system of nonlinear equations is characterized by:',
                ['A) All equations are linear.',
                 'B) At least one equation is nonlinear.',
                 'C) All variables appear only to the first power.',
                 'D) The coefficient matrix is singular.'],
                'B'),
               ('Which numerical method is commonly extended to solve systems of nonlinear equations?',
                ['A) Euler Method', 'B) Newton-Raphson Method', 'C) Gaussian Elimination', "D) Simpson's Rule"],
                'B'),
               ('In the multidimensional Newton-Raphson Method, the derivative is replaced by:',
                ['A) Hessian Matrix', 'B) Jacobian Matrix', 'C) Identity Matrix', 'D) Diagonal Matrix'],
                'B'),
               ('The Jacobian Matrix consists of:',
                ['A) Second-order derivatives',
                 'B) Partial derivatives of the functions',
                 'C) Function values only',
                 'D) Constants only'],
                'B'),
               ('The Newton-Raphson Method for nonlinear systems usually requires:',
                ['A) One initial value',
                 'B) Initial approximations for all variables',
                 'C) Only the exact solution',
                 'D) A linear coefficient matrix'],
                'B'),
               ('If the Jacobian Matrix is singular during an iteration:',
                ['A) The method may fail.',
                 'B) Convergence is guaranteed.',
                 'C) The iteration becomes faster.',
                 'D) The Jacobian is ignored.'],
                'A'),
               ('Which factor has the greatest effect on convergence?',
                ['A) Variable names', 'B) Initial guess', 'C) Number of equations', 'D) Matrix size only'],
                'B'),
               ('The size of the Jacobian Matrix for n variables is:',
                ['A) n × 1', 'B) 1 × n', 'C) n × n', 'D) (n+1) × n'],
                'C'),
               ('Which statement about nonlinear systems is TRUE?',
                ['A) They always have one unique solution.',
                 'B) They may have multiple or no solutions.',
                 'C) They can always be solved analytically.',
                 'D) They never require iteration.'],
                'B'),
               ('What is solved during each Newton iteration?',
                ['A) A differential equation',
                 'B) A linear system involving the Jacobian',
                 'C) A quadratic equation',
                 'D) An eigenvalue problem'],
                'B'),
               ('The Newton correction Δx is commonly found from:',
                ['A) J(xₖ)Δx=-F(xₖ)', 'B) J(xₖ)Δx=F′′(xₖ)', 'C) Δx=det(J)', 'D) F(xₖ)=0 without solving anything'],
                'A'),
               ('After solving for Δx, the usual update is:',
                ['A) xₖ₊₁=xₖ+Δx', 'B) xₖ₊₁=xₖΔx', 'C) xₖ₊₁=F(xₖ)', 'D) xₖ₊₁=J(xₖ)'],
                'A'),
               ('A common convergence test monitors:',
                ['A) ‖F(xₖ)‖ and/or ‖Δx‖',
                 'B) Only the variable names',
                 'C) The determinant of an unrelated matrix',
                 'D) The number of equations only'],
                'A'),
               ('Damping or line search is used mainly to:',
                ['A) Improve robustness when a full Newton step is too aggressive',
                 'B) Remove the Jacobian permanently',
                 'C) Make every system linear',
                 'D) Guarantee a unique solution'],
                'A'),
               ('Near a nonsingular simple solution and with a good initial guess, Newton’s method for systems is '
                'typically:',
                ['A) Quadratically convergent',
                 'B) Linearly divergent',
                 'C) Exact in one step for every nonlinear system',
                 'D) Independent of the Jacobian'],
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
