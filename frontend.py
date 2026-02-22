# frontend.py — Streamlit UI
import os
import time
import traceback
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Question Assignment", page_icon="🧠", layout="wide")

def get_system_class():
    try:
        from importlib import import_module, reload
        m = import_module("backend")
        m = reload(m)
        return m.AIQuestionAssignmentSystem
    except Exception as e:
        st.error(f"Couldn't import AIQuestionAssignmentSystem from backend.py:\n{e}")
        st.stop()

st.sidebar.title("⚙️ Settings")
use_uploaded = st.sidebar.radio(
    "Choose data source",
    options=["Use uploads", "Use local files (marksheet.xlsx, questions.xlsx)"],
    index=0
)
questions_per_student = st.sidebar.slider("Questions per student", 1, 10, 3)
st.sidebar.markdown("---")
st.sidebar.caption("Expected columns → marksheet: Name, Marks · questions: Question")

st.title("🧠 AI-based Question Assignment System")
st.caption("AI classifies question difficulty, saves it to question_difficulty.xlsx, and assigns questions per student level.")

uploaded_marks = uploaded_questions = None
if use_uploaded == "Use uploads":
    c1, c2 = st.columns(2)
    with c1: uploaded_marks = st.file_uploader("Upload marksheet.xlsx", type=["xlsx"])
    with c2: uploaded_questions = st.file_uploader("Upload questions.xlsx", type=["xlsx"])

run_clicked = st.button("🚀 Run Assignment")

def load_inputs():
    if use_uploaded == "Use uploads":
        if not uploaded_marks or not uploaded_questions:
            st.warning("Please upload **both** files to continue."); st.stop()
        return pd.read_excel(uploaded_marks), pd.read_excel(uploaded_questions), None, None
    else:
        marks_path, ques_path = "marksheet.xlsx", "questions.xlsx"
        if not (os.path.exists(marks_path) and os.path.exists(ques_path)):
            st.error("Local files not found. Place them next to this app or switch to 'Use uploads'."); st.stop()
        return None, None, marks_path, ques_path

if run_clicked:
    try:
        AIQuestionAssignmentSystem = get_system_class()

        if use_uploaded == "Use uploads":
            marks_df, questions_df, _, _ = load_inputs()
            tmp_marks = f"marksheet_{int(time.time())}.xlsx"
            tmp_ques  = f"questions_{int(time.time())}.xlsx"
            marks_df.to_excel(tmp_marks, index=False); questions_df.to_excel(tmp_ques, index=False)
            marks_path, ques_path = tmp_marks, tmp_ques
        else:
            _, _, marks_path, ques_path = load_inputs()

        system = AIQuestionAssignmentSystem(
            marksheet_path=marks_path,
            questions_path=ques_path,
            output_path="assigned_questions.xlsx",
            question_difficulty_path="question_difficulty.xlsx",
        )

        with st.spinner("Running the pipeline..."):
            marksheet, questions_df = system.load_and_validate_data()
            st.success(f"Loaded {len(marksheet)} students and {len(questions_df)} questions.")

            st.subheader("📄 Preview: Inputs")
            c1, c2 = st.columns(2)
            with c1: st.markdown("**Marksheet**");  st.dataframe(marksheet.head(15), use_container_width=True)
            with c2: st.markdown("**Questions**");  st.dataframe(questions_df.head(30), use_container_width=True)

            marksheet = system.cluster_student_difficulty(marksheet)
            questions_df = system.classify_question_difficulty(questions_df)

            # 👉 NEW: save & show AI difficulty table
            qdiff_path = system.save_question_difficulty(questions_df)
            st.subheader("🧪 Question Difficulty (AI)")
            show_cols = ["Question", "AI_Difficulty", "Confidence_Score", "Question_Difficulty"]
            st.dataframe(questions_df[show_cols].head(50), use_container_width=True)
            if os.path.exists(qdiff_path):
                with open(qdiff_path, "rb") as f:
                    st.download_button("⬇️ Download question_difficulty.xlsx", f, file_name=os.path.basename(qdiff_path))

            assigned_df = system.assign_questions_to_students(marksheet, questions_df, questions_per_student)
            summary = system.generate_summary_report(marksheet, questions_df, assigned_df)
            system.save_results(assigned_df, summary)

        st.success("✅ Assignment complete!")

        st.subheader("📝 Assigned Questions (sample)")
        st.dataframe(assigned_df.head(25), use_container_width=True)

        st.subheader("📊 Distributions")
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown("**Students per Level**");   st.bar_chart(pd.DataFrame.from_dict(summary["difficulty_distribution"], orient="index", columns=["count"]))
        with c2: st.markdown("**Questions per Level**");  st.bar_chart(pd.DataFrame.from_dict(summary["questions_per_difficulty"], orient="index", columns=["count"]))
        with c3:
            st.markdown("**Avg Marks per Level**")
            if summary["average_marks_per_level"]:
                st.bar_chart(pd.DataFrame.from_dict(summary["average_marks_per_level"], orient="index", columns=["avg_marks"]))

        # Results downloads
        if os.path.exists(system.output_path):
            with open(system.output_path, "rb") as f:
                st.download_button("⬇️ Download assigned_questions.xlsx", f, file_name=os.path.basename(system.output_path))
        s_path = system.output_path.replace(".xlsx", "_summary.xlsx")
        if os.path.exists(s_path):
            with open(s_path, "rb") as f:
                st.download_button("⬇️ Download summary.xlsx", f, file_name=os.path.basename(s_path))

        with st.expander("🔧 Debug Info"): st.json(summary)

    except Exception as e:
        st.error("The pipeline failed."); st.exception(e); st.text(traceback.format_exc())

st.markdown("---")
st.caption("AI difficulty is saved to question_difficulty.xlsx and displayed above.")
