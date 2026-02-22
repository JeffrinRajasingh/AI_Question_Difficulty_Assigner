# main3.py — Corrected end-to-end pipeline with consistent Level labels
import pandas as pd
import random
import numpy as np
import warnings
import os
from typing import Dict, Tuple
import logging

# Optional: local sklearn classifier (if trained by model_trainer.py)
import joblib

from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AIQuestionAssignmentSystem:
    """
    End-to-end system:
      1) Load & validate Excel inputs
      2) Cluster students into Level1/Level2/Level3 (by marks)
      3) Classify questions (internally as Easy/Medium/Hard), then map -> Level1/2/3
      4) Assign questions by matched level
      5) Save results + summary
    """

    def __init__(
        self,
        marksheet_path: str = "marksheet.xlsx",
        questions_path: str = "questions.xlsx",
        output_path: str = "assigned_questions.xlsx",
    ):
        self.marksheet_path = marksheet_path
        self.questions_path = questions_path
        self.output_path = output_path

        # Canonical label sets
        self.ai_labels = ["Easy", "Medium", "Hard"]            # For AI/heuristics
        self.level_labels = ["Level1", "Level2", "Level3"]     # For students/questions final

        # Mapping between AI labels and Level labels (customize names here if you like)
        self.ai_to_level = {"Easy": "Level1", "Medium": "Level2", "Hard": "Level3"}
        self.level_to_ai = {v: k for k, v in self.ai_to_level.items()}

        # Classifiers
        self.classifier = None  # zero-shot
        self.qclf = None        # sklearn question classifier (optional)
        self.qvec = None

        # Try to load local sklearn models if present
        try:
            if os.path.exists("models/question_sklearn_model.pkl") and os.path.exists("models/question_vectorizer.pkl"):
                self.qclf = joblib.load("models/question_sklearn_model.pkl")
                self.qvec = joblib.load("models/question_vectorizer.pkl")
                logger.info("Loaded sklearn question classifier and vectorizer from models/")
        except Exception as e:
            logger.warning(f"Could not load local sklearn classifier: {e}")
            self.qclf, self.qvec = None, None

    # ----------------------------
    # IO + Validation
    # ----------------------------
    def load_and_validate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate input data files"""
        # Marksheet
        if not os.path.exists(self.marksheet_path):
            raise FileNotFoundError(f"Marksheet file not found: {self.marksheet_path}")
        marksheet = pd.read_excel(self.marksheet_path)
        logger.info(f"Loaded marksheet with {len(marksheet)} rows")

        req_cols = ["Name", "Marks"]
        missing = [c for c in req_cols if c not in marksheet.columns]
        if missing:
            raise ValueError(f"Marksheet missing required columns: {missing}")

        marksheet = marksheet.dropna(subset=["Name", "Marks"])
        marksheet["Marks"] = pd.to_numeric(marksheet["Marks"], errors="coerce")
        marksheet = marksheet.dropna(subset=["Marks"])
        if marksheet["Marks"].min() < 0 or marksheet["Marks"].max() > 100:
            logger.warning("Marks outside 0–100 detected.")

        # Questions
        if not os.path.exists(self.questions_path):
            raise FileNotFoundError(f"Questions file not found: {self.questions_path}")
        questions_df = pd.read_excel(self.questions_path)
        logger.info(f"Loaded questions with {len(questions_df)} rows")

        if "Question" not in questions_df.columns:
            raise ValueError("Questions file must have a 'Question' column")

        questions_df = questions_df.dropna(subset=["Question"])
        questions_df["Question"] = questions_df["Question"].astype(str).str.strip()
        questions_df = questions_df[questions_df["Question"] != ""]

        return marksheet, questions_df

    # ----------------------------
    # Student clustering -> Levels
    # ----------------------------
    def cluster_student_difficulty(self, marksheet: pd.DataFrame) -> pd.DataFrame:
        """Cluster students into Level1/Level2/Level3 using KMeans on marks"""
        marks = marksheet["Marks"].values.reshape(-1, 1)
        scaler = StandardScaler()
        marks_scaled = scaler.fit_transform(marks)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(marks_scaled)
        marksheet["Difficulty_Cluster"] = clusters

        # Order clusters by original-scale centers (low->high)
        centers_original = scaler.inverse_transform(kmeans.cluster_centers_).flatten()
        order = np.argsort(centers_original)  # index 0 -> lowest marks cluster
        # Map to Level1 (lowest) ... Level3 (highest)
        cluster_to_level = {}
        for rank, cidx in enumerate(order):
            cluster_to_level[cidx] = self.level_labels[rank]

        marksheet["Student_Difficulty"] = marksheet["Difficulty_Cluster"].map(cluster_to_level)

        # Stats
        for lvl in self.level_labels:
            subset = marksheet[marksheet["Student_Difficulty"] == lvl]["Marks"]
            if len(subset) > 0:
                logger.info(f"{lvl}: {len(subset)} students, marks {subset.min():.1f}–{subset.max():.1f}")
            else:
                logger.info(f"{lvl}: 0 students")

        return marksheet

    # ----------------------------
    # Question classification (Easy/Medium/Hard) -> mapped to Level1/2/3
    # ----------------------------
    def classify_question_difficulty(self, questions_df: pd.DataFrame) -> pd.DataFrame:
        """Classify questions; internally E/M/H, then map to Level1/2/3"""
        # Lazy-load zero-shot classifier
        if self.classifier is None:
            logger.info("Loading zero-shot classifier (facebook/bart-large-mnli)...")
            warnings.filterwarnings("ignore")
            # device=-1 forces CPU to avoid GPU issues
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

        use_sklearn = self.qclf is not None and self.qvec is not None
        if use_sklearn:
            logger.info("Using sklearn classifier when heuristics do not decide.")

        preds_ai = []
        confs = []

        for i, q in enumerate(questions_df["Question"]):
            if i % 10 == 0:
                logger.info(f"Classifying question {i}/{len(questions_df)}")

            # 1) Heuristics first
            h = self._classify_by_heuristics(q)
            if h is not None:
                preds_ai.append(h)  # E/M/H
                confs.append(0.85)
                continue

            # 2) sklearn model if available
            if use_sklearn:
                try:
                    X = self.qvec.transform([q])
                    proba = self.qclf.predict_proba(X)[0]
                    idx = int(np.argmax(proba))
                    pred = self.qclf.classes_[idx]  # should be "Easy/Medium/Hard"
                    preds_ai.append(pred)
                    confs.append(float(proba[idx]))
                    continue
                except Exception as e:
                    logger.warning(f"sklearn classify failed; falling back to zero-shot. Err: {e}")

            # 3) zero-shot fallback (labels must be E/M/H)
            result = self.classifier(q, self.ai_labels)
            preds_ai.append(result["labels"][0])
            confs.append(result["scores"][0])

        # Map AI labels -> Level labels
        questions_df = questions_df.copy()
        questions_df["AI_Difficulty"] = preds_ai  # keep for debug/inspection
        questions_df["Confidence_Score"] = confs
        questions_df["Question_Difficulty"] = questions_df["AI_Difficulty"].map(self.ai_to_level)

        # Balance distribution across Levels
        questions_df = self._balance_question_distribution(questions_df)

        # Log stats
        for lvl in self.level_labels:
            cnt = (questions_df["Question_Difficulty"] == lvl).sum()
            if cnt > 0:
                avgc = questions_df.loc[
                    questions_df["Question_Difficulty"] == lvl, "Confidence_Score"
                ].mean()
                logger.info(f"{lvl} questions: {cnt}, avg confidence: {avgc:.3f}")
            else:
                logger.warning(f"No {lvl} questions found after mapping!")

        return questions_df

    def _classify_by_heuristics(self, question: str):
        """Return 'Easy' | 'Medium' | 'Hard' or None if undecided."""
        q = question.lower().strip()

        easy = [
            "define ", "what is ", "what are ", "who is ", "who are ",
            "name the ", "list the ", "state the ", "mention ",
            "give the definition", "meaning of ", "expand ",
            "full form", "stands for", "acronym",
        ]
        hard = [
            "analyze ", "analyse ", "evaluate ", "justify ",
            "critically examine", "compare and contrast", "assess ",
            "critique ", "synthesize", "derive ", "prove ",
            "design ", "develop ", "create a strategy", "formulate ",
            "why do you think", "in your opinion",
        ]
        medium = [
            "explain ", "describe ", "how does ", "why does ",
            "what happens when", "what would happen if",
            "compare ", "differentiate", "distinguish between",
            "illustrate ", "demonstrate ", "show that ",
            "calculate ", "solve ", "find the ", "determine ",
        ]

        if any(p in q for p in easy):
            return "Easy"
        if any(p in q for p in hard):
            return "Hard"
        if any(p in q for p in medium):
            return "Medium"

        # Short WH questions → Easy
        if len(question.split()) <= 6 and any(w in q for w in ["what", "define", "name", "who", "when", "where"]):
            return "Easy"
        # Complex punctuation / clauses → Hard
        if q.count(",") >= 2 or q.count(";") >= 1 or len(question.split()) > 22:
            return "Hard"
        return None

    def _balance_question_distribution(self, questions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance counts across Level1/Level2/Level3 using confidence-aware reassignment.
        Works on 'Question_Difficulty' which is already Level labels.
        """
        counts = questions_df["Question_Difficulty"].value_counts()
        total = len(questions_df)
        target = max(1, total // 3)

        logger.info(f"Current distribution: {dict(counts)}; target ≈ {target} per level")

        working = questions_df.copy()

        # Pass 1: Fill missing levels (if any)
        for lvl in self.level_labels:
            cur = counts.get(lvl, 0)
            if cur == 0 and len(counts) > 0:
                donor = counts.idxmax()
                pool = working[working["Question_Difficulty"] == donor].sort_values("Confidence_Score", ascending=True)
                to_move = min(target, len(pool), counts[donor] - 1)
                if to_move > 0:
                    idxs = pool.head(to_move).index
                    working.loc[idxs, "Question_Difficulty"] = lvl
                    counts[lvl] = to_move
                    counts[donor] -= to_move
                    logger.info(f"Reassigned {to_move} from {donor} -> {lvl}")

        # Pass 2: Fine tune
        updated = working["Question_Difficulty"].value_counts()
        for lvl in self.level_labels:
            cur = updated.get(lvl, 0)
            if cur < target:
                need = target - cur
                for donor in self.level_labels:
                    if updated.get(donor, 0) > target and need > 0:
                        excess = updated[donor] - target
                        move = min(need, excess)
                        cand = working[
                            (working["Question_Difficulty"] == donor)
                            & (working["Confidence_Score"] < 0.7)
                        ].sort_values("Confidence_Score", ascending=True)
                        if len(cand) >= move and move > 0:
                            idxs = cand.head(move).index
                            working.loc[idxs, "Question_Difficulty"] = lvl
                            updated[lvl] = updated.get(lvl, 0) + move
                            updated[donor] -= move
                            need -= move
                            logger.info(f"Moved {move} low-confidence {donor} -> {lvl}")

        questions_df["Question_Difficulty"] = working["Question_Difficulty"]
        logger.info(f"Final distribution: {dict(questions_df['Question_Difficulty'].value_counts())}")
        return questions_df

    # ----------------------------
    # Assignment
    # ----------------------------
    def assign_questions_to_students(
        self,
        marksheet: pd.DataFrame,
        questions_df: pd.DataFrame,
        questions_per_student: int = 3,
    ) -> pd.DataFrame:
        """Assign Level-matched questions."""
        assigned_rows = []

        for _, student in marksheet.iterrows():
            level = student["Student_Difficulty"]  # Level1/Level2/Level3
            matching = questions_df[questions_df["Question_Difficulty"] == level]

            if len(matching) == 0:
                # Prefer Level2 as neutral fallback; else any questions
                logger.warning(f"No {level} questions for {student['Name']}; falling back.")
                matching = questions_df[questions_df["Question_Difficulty"] == "Level2"]
                if len(matching) == 0:
                    matching = questions_df

            # Sort by confidence desc (if available)
            if "Confidence_Score" in matching.columns:
                matching = matching.sort_values("Confidence_Score", ascending=False)

            qlist = matching["Question"].tolist()
            num = min(questions_per_student, len(qlist))
            assigned = random.sample(qlist, num) if num > 0 else []

            assigned_rows.append(
                {
                    "Name": student["Name"],
                    "Marks": student["Marks"],
                    "Difficulty_Level": level,
                    "Questions_Assigned": len(assigned),
                    "Assigned_Questions": " | ".join(assigned) if assigned else "",
                }
            )

        return pd.DataFrame(assigned_rows)

    # ----------------------------
    # Summary + Save
    # ----------------------------
    def generate_summary_report(self, marksheet: pd.DataFrame, questions_df: pd.DataFrame, assigned_df: pd.DataFrame) -> Dict:
        summary = {
            "total_students": len(marksheet),
            "total_questions": len(questions_df),
            "difficulty_distribution": {},
            "questions_per_difficulty": {},
            "average_marks_per_level": {},
        }

        for lvl in self.level_labels:
            students = marksheet[marksheet["Student_Difficulty"] == lvl]
            summary["difficulty_distribution"][lvl] = len(students)
            if len(students) > 0:
                summary["average_marks_per_level"][lvl] = round(students["Marks"].mean(), 2)

        for lvl in self.level_labels:
            summary["questions_per_difficulty"][lvl] = int((questions_df["Question_Difficulty"] == lvl).sum())

        return summary

    def save_results(self, assigned_df: pd.DataFrame, summary: Dict):
        """Save results and a human-readable summary workbook."""
        import time
        from datetime import datetime

        def safe_save_excel(df, filepath, **kwargs):
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    df.to_excel(filepath, **kwargs)
                    return filepath
                except PermissionError:
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"File {filepath} locked; retrying in 2s... (attempt {attempt+1}/{max_attempts})"
                        )
                        time.sleep(2)
                    else:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        base = filepath.replace(".xlsx", "")
                        fallback = f"{base}_{ts}.xlsx"
                        logger.warning(f"Using fallback filename: {fallback}")
                        df.to_excel(fallback, **kwargs)
                        return fallback
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt+1} failed: {e}")
                    time.sleep(1)

        # Main results
        actual_output = safe_save_excel(assigned_df, self.output_path, index=False)
        logger.info(f"Results saved to {actual_output}")

        # Summary workbook
        summary_path = self.output_path.replace(".xlsx", "_summary.xlsx")
        rows = [["Metric", "Value"]]
        rows += [["Total Students", summary["total_students"]], ["Total Questions", summary["total_questions"]], ["", ""]]
        rows += [["Student Distribution", ""]]
        for lvl in self.level_labels:
            rows.append([f"{lvl} Students", summary["difficulty_distribution"].get(lvl, 0)])
            if lvl in summary["average_marks_per_level"]:
                rows.append([f"{lvl} Avg Marks", summary["average_marks_per_level"][lvl]])
        rows += [["", ""], ["Question Distribution", ""]]
        for lvl in self.level_labels:
            rows.append([f"{lvl} Questions", summary["questions_per_difficulty"].get(lvl, 0)])

        summary_df = pd.DataFrame(rows)
        actual_summary = safe_save_excel(summary_df, summary_path, index=False, header=False)
        logger.info(f"Summary saved to {actual_summary}")

        self.output_path = actual_output  # in case fallback filename was used

    # ----------------------------
    # Orchestrator
    # ----------------------------
    def run(self, questions_per_student: int = 3) -> None:
        logger.info("🚀 Starting AI-based Question Assignment System")
        marksheet, questions_df = self.load_and_validate_data()

        logger.info("📊 Clustering students into levels...")
        marksheet = self.cluster_student_difficulty(marksheet)

        logger.info("🤖 Classifying questions (heuristics ➜ sklearn ➜ zero-shot) ...")
        questions_df = self.classify_question_difficulty(questions_df)

        logger.info("📝 Assigning questions to students...")
        assigned_df = self.assign_questions_to_students(marksheet, questions_df, questions_per_student)

        summary = self.generate_summary_report(marksheet, questions_df, assigned_df)
        self.save_results(assigned_df, summary)

        print("\n✅ AI-based difficulty assignment completed!")
        print(f"📈 Summary:")
        print(f"  • Total Students: {summary['total_students']}")
        print(f"  • Total Questions: {summary['total_questions']}")
        print(f"  • Student Distribution: {summary['difficulty_distribution']}")
        print(f"  • Question Distribution: {summary['questions_per_difficulty']}")
        print(f"📁 Results saved to: {self.output_path}")


# ----------------------------
# CLI entry
# ----------------------------
if __name__ == "__main__":
    system = AIQuestionAssignmentSystem(
        marksheet_path="marksheet.xlsx",
        questions_path="questions.xlsx",
        output_path="assigned_questions.xlsx",
    )
    system.run(questions_per_student=3)