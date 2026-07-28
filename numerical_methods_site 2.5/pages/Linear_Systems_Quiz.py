import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Linear Systems Quiz',
 'label': 'LINEAR SYSTEMS OVERVIEW QUIZ',
 'lesson': 'Naive_Gaussian_Elimination',
 'solver': 'Naive_Gaussian_Elimination_Solver',
 'category': 'Linear Systems',
 'key': 'linear_systems',
 'description': 'Review matrix form, solution classification, elimination, pivoting, Gauss–Jordan reduction, and '
                'computational cost.',
 'questions': [('A system of linear equations can be represented in matrix form as:',
                ['A) Ax = b', 'B) A + x = b', 'C) A/x = b', 'D) A²x = b'],
                'A'),
               ('A unique solution exists when:',
                ['A) det(A) = 0', 'B) det(A) ≠ 0', 'C) All rows are identical', 'D) The matrix is singular'],
                'B'),
               ('The primary goal of Gaussian Elimination is to transform a matrix into:',
                ['A) Identity Matrix',
                 'B) Upper Triangular Matrix',
                 'C) Lower Triangular Matrix',
                 'D) Diagonal Matrix'],
                'B'),
               ('Back substitution is performed after:',
                ['A) Matrix inversion', 'B) LU decomposition', 'C) Forward elimination', 'D) Scaling'],
                'C'),
               ('A singular matrix is characterized by:',
                ['A) det(A)=1', 'B) det(A)>0', 'C) det(A)<0', 'D) det(A)=0'],
                'D'),
               ('Partial Pivoting improves:',
                ['A) Graphical representation', 'B) Numerical stability', 'C) Memory allocation', 'D) Matrix size'],
                'B'),
               ('In Partial Pivoting, the pivot row is selected using:',
                ['A) Smallest element', 'B) Random element', 'C) Largest absolute value in the column', 'D) Row sum'],
                'C'),
               ('Scaled Partial Pivoting uses:',
                ['A) Determinants', 'B) Eigenvalues', 'C) Scaling factors', 'D) Matrix inverses'],
                'C'),
               ('The scaling factor of a row is:',
                ['A) The smallest value in the row',
                 'B) The largest absolute value in the row',
                 'C) The determinant',
                 'D) The diagonal entry'],
                'B'),
               ('Gauss-Jordan elimination transforms the matrix into:',
                ['A) Upper Triangular Form',
                 'B) Lower Triangular Form',
                 'C) Reduced Row Echelon Form',
                 'D) Hessenberg Form'],
                'C'),
               ('An advantage of Gauss-Jordan over Gaussian Elimination is:',
                ['A) Requires fewer operations',
                 'B) Eliminates the need for back substitution',
                 'C) Uses less memory',
                 'D) Avoids pivoting'],
                'B'),
               ('The computational complexity of Gaussian Elimination is approximately:',
                ['A) O(n)', 'B) O(log n)', 'C) O(n²)', 'D) O(n³)'],
                'D'),
               ('Which method is considered a direct method?',
                ['A) Jacobi Method', 'B) Gauss-Seidel Method', 'C) Gaussian Elimination', 'D) Fixed Point Iteration'],
                'C'),
               ('If two equations in a system are linearly dependent, the system:',
                ['A) Must have a unique solution',
                 'B) Cannot have any solution',
                 'C) May have infinitely many solutions',
                 'D) Must be diagonal'],
                'C'),
               ('Which statement about Linear Systems is TRUE?',
                ['A) Every system has a unique solution',
                 'B) Singular systems always have no solution',
                 'C) A system may have one, infinitely many, or no solutions',
                 'D) Determinants are only used for nonlinear systems'],
                'C')]}

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
