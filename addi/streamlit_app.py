import os
import io
import time
import traceback
import pandas as pd
import streamlit as st

# Try to import the user's assignment system
try:
    from main3 import AIQuestionAssignmentSystem
except Exception as e:
    st.error("Couldn't import AIQuestionAssignmentSystem from main3.py. Please ensure main3.py is in the same folder.")
    st.stop()

st.set_page_config(page_title="AI Question Assignment", page_icon="🧠", layout="wide")

# ---- Sidebar ----
st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("**Input Files**")
use_uploaded = st.sidebar.radio(
    "Choose data source",
    options=["Use uploads", "Use local files (marksheet.xlsx, questions.xlsx)"],
    index=0
)

questions_per_student = st.sidebar.slider("Questions per student", min_value=1, max_value=10, value=3, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: For best results, include columns:\n- marksheet.xlsx → **Name, Marks**\n- questions.xlsx → **Question**")

# ---- Main Title ----
st.title("🧠 AI-based Question Assignment System")
st.caption("Upload your marksheet and question bank, classify difficulties, and assign tailored questions to each student.")

# ---- File Uploaders ----
uploaded_marks = None
uploaded_questions = None

if use_uploaded == "Use uploads":
    col1, col2 = st.columns(2)
    with col1:
        uploaded_marks = st.file_uploader("Upload marksheet.xlsx (columns: Name, Marks)", type=["xlsx"])
    with col2:
        uploaded_questions = st.file_uploader("Upload questions.xlsx (column: Question)", type=["xlsx"])

# ---- Run Button ----
run_clicked = st.button("🚀 Run Assignment")

# ---- Helper: load dataframes from uploads or disk ----
def load_inputs():
    if use_uploaded == "Use uploads":
        if not uploaded_marks or not uploaded_questions:
            st.warning("Please upload **both** files to continue.")
            st.stop()
        marks_df = pd.read_excel(uploaded_marks)
        questions_df = pd.read_excel(uploaded_questions)
        return marks_df, questions_df, None, None
    else:
        # Use local files
        marks_path = "marksheet.xlsx"
        ques_path = "questions.xlsx"
        if not os.path.exists(marks_path) or not os.path.exists(ques_path):
            st.error("Local files not found. Place **marksheet.xlsx** and **questions.xlsx** next to this app, or switch to 'Use uploads'.")
            st.stop()
        return None, None, marks_path, ques_path

# ---- Runner ----
if run_clicked:
    try:
        # Prepare system
        if use_uploaded == "Use uploads":
            marks_df, questions_df, _, _ = load_inputs()
            # Save temp copies so the downstream class can read paths
            tmp_marks_path = f"marksheet_{int(time.time())}.xlsx"
            tmp_ques_path = f"questions_{int(time.time())}.xlsx"
            marks_df.to_excel(tmp_marks_path, index=False)
            questions_df.to_excel(tmp_ques_path, index=False)
            marks_path, ques_path = tmp_marks_path, tmp_ques_path
        else:
            _, _, marks_path, ques_path = load_inputs()

        output_path = "assigned_questions.xlsx"
        system = AIQuestionAssignmentSystem(
            marksheet_path=marks_path,
            questions_path=ques_path,
            output_path=output_path
        )

        with st.spinner("Loading data and running the pipeline..."):
            # Steps mirror main3.py run()
            marksheet, questions_df = system.load_and_validate_data()

            st.success(f"Loaded {len(marksheet)} students and {len(questions_df)} questions.")

            st.subheader("📄 Preview: Inputs")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Marksheet (first 15 rows)**")
                st.dataframe(marksheet.head(15), use_container_width=True)
            with c2:
                st.markdown("**Questions (first 15 rows)**")
                st.dataframe(questions_df.head(15), use_container_width=True)

            # Cluster students
            st.info("Clustering students by difficulty...")
            marksheet = system.cluster_student_difficulty(marksheet)

            # Classify questions
            st.info("Classifying questions by difficulty (heuristics + zero-shot model)...")
            questions_df = system.classify_question_difficulty(questions_df)

            # Assign
            st.info("Assigning questions to students...")
            assigned_df = system.assign_questions_to_students(
                marksheet, questions_df, questions_per_student=questions_per_student
            )

            # Summary
            summary = system.generate_summary_report(marksheet, questions_df, assigned_df)

            # Save
            system.save_results(assigned_df, summary)
        
        st.success("✅ Assignment complete!")
        
        # ---- Show results ----
        st.subheader("📝 Assigned Questions (sample)")
        st.dataframe(assigned_df.head(25), use_container_width=True)

        # Charts
        st.subheader("📊 Distributions")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown("**Students per Level**")
            st.bar_chart(pd.DataFrame.from_dict(summary["difficulty_distribution"], orient="index", columns=["count"]))
        with cc2:
            st.markdown("**Questions per Level**")
            st.bar_chart(pd.DataFrame.from_dict(summary["questions_per_difficulty"], orient="index", columns=["count"]))
        with cc3:
            st.markdown("**Avg Marks per Level**")
            if summary["average_marks_per_level"]:
                st.bar_chart(pd.DataFrame.from_dict(summary["average_marks_per_level"], orient="index", columns=["avg_marks"]))
            else:
                st.info("Not enough data to compute averages.")

        # ---- Downloads ----
        # Assigned file
        if os.path.exists(system.output_path):
            with open(system.output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download assigned_questions.xlsx",
                    data=f.read(),
                    file_name=os.path.basename(system.output_path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        # Summary file
        summary_path = system.output_path.replace(".xlsx", "_summary.xlsx")
        if os.path.exists(summary_path):
            with open(summary_path, "rb") as f:
                st.download_button(
                    "⬇️ Download summary.xlsx",
                    data=f.read(),
                    file_name=os.path.basename(summary_path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        # ---- Debug / Details ----
        with st.expander("🔧 Details / Debug Info"):
            st.json(summary)

    except Exception as e:
        st.error("The pipeline failed. See error details below.")
        st.exception(e)
        st.text(traceback.format_exc())

# ---- Footer ----
st.markdown("---")
st.caption("Built on your existing pipeline from main3.py. Tip: If the zero-shot model download is slow, consider pre-downloading models or using the sklearn path in model_trainer.py.")