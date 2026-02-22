import pandas as pd
import random
import numpy as np
from transformers import pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import os
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIQuestionAssignmentSystem:
    def __init__(self, marksheet_path: str = "marksheet.xlsx", 
                 questions_path: str = "questions.xlsx",
                 output_path: str = "assigned_questions.xlsx"):
        """
        Initialize the AI Question Assignment System
        
        Args:
            marksheet_path: Path to student marksheet Excel file
            questions_path: Path to questions bank Excel file
            output_path: Path for output Excel file
        """
        self.marksheet_path = marksheet_path
        self.questions_path = questions_path
        self.output_path = output_path
        self.difficulty_labels = ["Level1", "Level2", "Level3"]
        self.classifier = None
        
    def load_and_validate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate input data files"""
        try:
            # Load marksheet
            if not os.path.exists(self.marksheet_path):
                raise FileNotFoundError(f"Marksheet file not found: {self.marksheet_path}")
            
            marksheet = pd.read_excel(self.marksheet_path)
            logger.info(f"Loaded marksheet with {len(marksheet)} students")
            
            # Validate marksheet columns
            required_cols = ['Name', 'Marks']
            missing_cols = [col for col in required_cols if col not in marksheet.columns]
            if missing_cols:
                raise ValueError(f"Marksheet missing required columns: {missing_cols}")
            
            # Clean and validate marks data
            marksheet = marksheet.dropna(subset=['Name', 'Marks'])
            marksheet['Marks'] = pd.to_numeric(marksheet['Marks'], errors='coerce')
            marksheet = marksheet.dropna(subset=['Marks'])
            
            if marksheet['Marks'].min() < 0 or marksheet['Marks'].max() > 100:
                logger.warning("Marks should typically be between 0-100")
            
            # Load questions
            if not os.path.exists(self.questions_path):
                raise FileNotFoundError(f"Questions file not found: {self.questions_path}")
                
            questions_df = pd.read_excel(self.questions_path)
            logger.info(f"Loaded {len(questions_df)} questions")
            
            if "Question" not in questions_df.columns:
                raise ValueError("Questions file must have 'Question' column")
            
            # Clean questions data
            questions_df = questions_df.dropna(subset=['Question'])
            questions_df['Question'] = questions_df['Question'].astype(str).str.strip()
            questions_df = questions_df[questions_df['Question'] != '']
            
            return marksheet, questions_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def cluster_student_difficulty(self, marksheet: pd.DataFrame) -> pd.DataFrame:
        """Cluster students into difficulty levels based on marks"""
        try:
            # Prepare marks data
            marks = marksheet["Marks"].values.reshape(-1, 1)
            
            # Use StandardScaler for better clustering (optional but recommended)
            scaler = StandardScaler()
            marks_scaled = scaler.fit_transform(marks)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            marksheet["Difficulty_Cluster"] = kmeans.fit_predict(marks_scaled)
            
            # Map clusters to difficulty labels by ordering cluster centers
            # Transform centers back to original scale for interpretation
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_).flatten()
            sorted_centers = np.argsort(cluster_centers)
            
            # Create mapping: lowest marks cluster -> "Easy", highest -> "Hard"
            difficulty_map = {}
            for rank, cluster_idx in enumerate(sorted_centers):
                difficulty_map[cluster_idx] = self.difficulty_labels[rank]
            
            marksheet["Student_Difficulty"] = marksheet["Difficulty_Cluster"].map(difficulty_map)
            
            # Log cluster statistics
            for difficulty in self.difficulty_labels:
                subset = marksheet[marksheet["Student_Difficulty"] == difficulty]["Marks"]
                logger.info(f"{difficulty} level: {len(subset)} students, "
                           f"marks range: {subset.min():.1f}-{subset.max():.1f}")
            
            return marksheet
            
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
            raise
    
    def classify_question_difficulty(self, questions_df: pd.DataFrame) -> pd.DataFrame:
        """Classify questions into difficulty levels using AI with fallback heuristics"""
        try:
            # Initialize classifier with error handling
            if self.classifier is None:
                logger.info("Loading AI classifier...")
                warnings.filterwarnings("ignore")
                self.classifier = pipeline(
                    "zero-shot-classification", 
                    model="facebook/bart-large-mnli",
                    device=-1  # Use CPU to avoid CUDA issues
                )
            
            predicted_difficulties = []
            confidence_scores = []
            
            logger.info("Classifying question difficulties...")
            for i, question in enumerate(questions_df["Question"]):
                if i % 5 == 0:  # More frequent progress updates for small datasets
                    logger.info(f"Processed {i}/{len(questions_df)} questions")
                
                try:
                    # First, try heuristic-based classification for obvious cases
                    heuristic_difficulty = self._classify_by_heuristics(question)
                    
                    if heuristic_difficulty:
                        predicted_difficulties.append(heuristic_difficulty)
                        confidence_scores.append(0.8)  # High confidence for heuristic matches
                        logger.debug(f"Heuristic classification: '{question[:50]}...' -> {heuristic_difficulty}")
                        continue
                    
                    # Use AI classifier for complex cases
                    prompt = f"""
                    Educational Question Difficulty Analysis:
                    "{question}"
                    
                    Difficulty Guidelines:
                    - Easy: Simple definitions, basic recall, "what is", "define", single concepts
                    - Medium: Applications, comparisons, "how does", "explain why", multi-step
                    - Hard: Complex analysis, synthesis, "evaluate", "analyze", advanced reasoning
                    """
                    
                    result = self.classifier(prompt.strip(), self.difficulty_labels)
                    
                    predicted_difficulties.append(result["labels"][0])
                    confidence_scores.append(result["scores"][0])
                    
                except Exception as e:
                    logger.warning(f"Error classifying question {i}: {str(e)}")
                    # Use heuristic fallback
                    fallback = self._classify_by_heuristics(question) or self.difficulty_labels[i % 3]
                    predicted_difficulties.append(fallback)
                    confidence_scores.append(0.3)
            
            questions_df["Question_Difficulty"] = predicted_difficulties
            questions_df["Confidence_Score"] = confidence_scores
            
            # Log classification statistics
            for difficulty in self.difficulty_labels:
                count = sum(1 for d in predicted_difficulties if d == difficulty)
                if count > 0:
                    avg_conf = np.mean([conf for d, conf in zip(predicted_difficulties, confidence_scores) if d == difficulty])
                    logger.info(f"{difficulty} questions: {count}, avg confidence: {avg_conf:.3f}")
                else:
                    logger.warning(f"No {difficulty} questions found in classification!")
            
            # Balance question distribution if needed
            questions_df = self._balance_question_distribution(questions_df)
            
            return questions_df
            
        except Exception as e:
            logger.error(f"Error in question classification: {str(e)}")
            raise
    
    def _classify_by_heuristics(self, question: str) -> str:
        """Classify questions using rule-based heuristics for obvious cases"""
        question_lower = question.lower().strip()
        
        # Easy question patterns
        easy_patterns = [
            'define ', 'what is ', 'what are ', 'who is ', 'who are ',
            'name the ', 'list the ', 'state the ', 'mention ',
            'give the definition', 'meaning of ', 'expand ',
            'full form', 'stands for', 'acronym'
        ]
        
        # Hard question patterns  
        hard_patterns = [
            'analyze ', 'analyse ', 'evaluate ', 'justify ',
            'critically examine', 'compare and contrast', 'assess ',
            'critique ', 'synthesize', 'derive ', 'prove ',
            'design ', 'develop ', 'create a strategy', 'formulate ',
            'why do you think', 'in your opinion'
        ]
        
        # Medium question patterns
        medium_patterns = [
            'explain ', 'describe ', 'how does ', 'why does ',
            'what happens when', 'what would happen if',
            'compare ', 'differentiate', 'distinguish between',
            'illustrate ', 'demonstrate ', 'show that ',
            'calculate ', 'solve ', 'find the ', 'determine '
        ]
        
        # Check for easy patterns first
        if any(pattern in question_lower for pattern in easy_patterns):
            return "Easy"
            
        # Check for hard patterns
        if any(pattern in question_lower for pattern in hard_patterns):
            return "Hard"
            
        # Check for medium patterns
        if any(pattern in question_lower for pattern in medium_patterns):
            return "Medium"
        
        # Additional checks
        # Short questions with simple words tend to be easy
        if len(question.split()) <= 5 and any(word in question_lower for word in ['what', 'define', 'name']):
            return "Easy"
            
        # Questions with complex punctuation or multiple clauses tend to be harder
        if question.count(',') >= 2 or question.count(';') >= 1 or len(question.split()) > 20:
            return "Hard"
        
        return None  # Let AI classifier decide
    
    def _balance_question_distribution(self, questions_df: pd.DataFrame) -> pd.DataFrame:
        """Balance question distribution across difficulty levels"""
        # Count questions per difficulty
        difficulty_counts = questions_df["Question_Difficulty"].value_counts()
        total_questions = len(questions_df)
        
        # Target distribution: aim for roughly equal distribution
        target_per_level = max(1, total_questions // 3)
        
        logger.info(f"Current distribution: {dict(difficulty_counts)}")
        logger.info(f"Target per level: ~{target_per_level} questions")
        
        # Create a copy to work with
        working_df = questions_df.copy()
        
        # First pass: Handle levels with 0 questions
        for difficulty in self.difficulty_labels:
            current_count = difficulty_counts.get(difficulty, 0)
            
            if current_count == 0:
                logger.info(f"Rebalancing: {difficulty} level has 0 questions, reassigning...")
                
                # Find questions from the most over-represented level
                max_level = difficulty_counts.idxmax()
                candidates = working_df[working_df["Question_Difficulty"] == max_level].copy()
                
                # Sort by confidence (lowest first for reassignment)
                candidates = candidates.sort_values("Confidence_Score", ascending=True)
                
                # Reassign questions to this level
                num_to_reassign = min(target_per_level, len(candidates), difficulty_counts[max_level] - 1)
                if num_to_reassign > 0:
                    reassign_indices = candidates.head(num_to_reassign).index
                    working_df.loc[reassign_indices, "Question_Difficulty"] = difficulty
                    
                    # Update our tracking
                    difficulty_counts[difficulty] = num_to_reassign
                    difficulty_counts[max_level] -= num_to_reassign
                    
                    logger.info(f"Reassigned {num_to_reassign} questions from {max_level} to {difficulty}")
        
        # Second pass: Fine-tune the distribution
        updated_counts = working_df["Question_Difficulty"].value_counts()
        
        for difficulty in self.difficulty_labels:
            current_count = updated_counts.get(difficulty, 0)
            
            if current_count < target_per_level:
                needed = target_per_level - current_count
                
                # Find over-represented levels to take from
                for donor_level in self.difficulty_labels:
                    if updated_counts.get(donor_level, 0) > target_per_level and needed > 0:
                        excess = updated_counts[donor_level] - target_per_level
                        to_move = min(needed, excess)
                        
                        # Find lowest confidence questions from donor level
                        candidates = working_df[
                            (working_df["Question_Difficulty"] == donor_level) & 
                            (working_df["Confidence_Score"] < 0.7)
                        ].sort_values("Confidence_Score", ascending=True)
                        
                        if len(candidates) >= to_move:
                            move_indices = candidates.head(to_move).index
                            working_df.loc[move_indices, "Question_Difficulty"] = difficulty
                            
                            updated_counts[difficulty] = updated_counts.get(difficulty, 0) + to_move
                            updated_counts[donor_level] -= to_move
                            needed -= to_move
                            
                            logger.info(f"Moved {to_move} low-confidence questions from {donor_level} to {difficulty}")
        
        # Update the original dataframe
        questions_df["Question_Difficulty"] = working_df["Question_Difficulty"]
        
        # Log final distribution
        final_counts = questions_df["Question_Difficulty"].value_counts()
        logger.info(f"Final distribution: {dict(final_counts)}")
        
        return questions_df
    
    def assign_questions_to_students(self, marksheet: pd.DataFrame, 
                                   questions_df: pd.DataFrame,
                                   questions_per_student: int = 3) -> pd.DataFrame:
        """Assign appropriate questions to each student"""
        try:
            assigned_data = []
            
            for _, student in marksheet.iterrows():
                level = student["Student_Difficulty"]
                
                # Get questions matching student's difficulty level
                matching_questions = questions_df[
                    questions_df["Question_Difficulty"] == level
                ]
                
                if len(matching_questions) == 0:
                    # Fallback: use Medium difficulty questions if exact match not available
                    logger.warning(f"No {level} questions found for {student['Name']}, using Medium")
                    matching_questions = questions_df[
                        questions_df["Question_Difficulty"] == "Medium"
                    ]
                
                if len(matching_questions) == 0:
                    # Final fallback: use any available questions
                    matching_questions = questions_df
                    assigned_questions = ["No suitable questions available"]
                else:
                    # Select questions (prioritize high confidence if available)
                    if "Confidence_Score" in matching_questions.columns:
                        # Sort by confidence score descending
                        matching_questions = matching_questions.sort_values(
                            "Confidence_Score", ascending=False
                        )
                    
                    question_list = matching_questions["Question"].tolist()
                    num_to_assign = min(questions_per_student, len(question_list))
                    assigned_questions = random.sample(question_list, num_to_assign)
                
                assigned_data.append({
                    "Name": student["Name"],
                    "Marks": student["Marks"],
                    "Difficulty_Level": level,
                    "Questions_Assigned": len(assigned_questions) if assigned_questions != ["No suitable questions available"] else 0,
                    "Assigned_Questions": " | ".join(assigned_questions)
                })
            
            return pd.DataFrame(assigned_data)
            
        except Exception as e:
            logger.error(f"Error in question assignment: {str(e)}")
            raise
    
    def generate_summary_report(self, marksheet: pd.DataFrame, 
                              questions_df: pd.DataFrame,
                              assigned_df: pd.DataFrame) -> Dict:
        """Generate a summary report of the assignment process"""
        summary = {
            "total_students": len(marksheet),
            "total_questions": len(questions_df),
            "difficulty_distribution": {},
            "questions_per_difficulty": {},
            "average_marks_per_level": {}
        }
        
        # Student difficulty distribution
        for level in self.difficulty_labels:
            student_count = len(marksheet[marksheet["Student_Difficulty"] == level])
            summary["difficulty_distribution"][level] = student_count
            
            if student_count > 0:
                avg_marks = marksheet[marksheet["Student_Difficulty"] == level]["Marks"].mean()
                summary["average_marks_per_level"][level] = round(avg_marks, 2)
        
        # Question difficulty distribution
        for level in self.difficulty_labels:
            question_count = len(questions_df[questions_df["Question_Difficulty"] == level])
            summary["questions_per_difficulty"][level] = question_count
        
        return summary
    
    def save_results(self, assigned_df: pd.DataFrame, summary: Dict):
        """Save results and summary to Excel files"""
        import time
        from datetime import datetime
        
        def safe_save_excel(df, filepath, **kwargs):
            """Safely save Excel file with retry logic and timestamp fallback"""
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    df.to_excel(filepath, **kwargs)
                    return filepath
                except PermissionError:
                    if attempt < max_attempts - 1:
                        logger.warning(f"File {filepath} is locked, retrying in 2 seconds... (attempt {attempt + 1}/{max_attempts})")
                        time.sleep(2)
                    else:
                        # Create timestamped filename as fallback
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        base_name = filepath.replace('.xlsx', '')
                        fallback_path = f"{base_name}_{timestamp}.xlsx"
                        logger.warning(f"Using fallback filename: {fallback_path}")
                        df.to_excel(fallback_path, **kwargs)
                        return fallback_path
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(1)
        
        try:
            # Save main results
            actual_output_path = safe_save_excel(assigned_df, self.output_path, index=False)
            logger.info(f"Results saved to {actual_output_path}")
            
            # Save summary report
            summary_path = self.output_path.replace('.xlsx', '_summary.xlsx')
            
            summary_data = []
            summary_data.append(["Metric", "Value"])
            summary_data.append(["Total Students", summary["total_students"]])
            summary_data.append(["Total Questions", summary["total_questions"]])
            summary_data.append(["", ""])
            summary_data.append(["Student Distribution", ""])
            
            for level in self.difficulty_labels:
                summary_data.append([f"{level} Students", summary["difficulty_distribution"][level]])
                if level in summary["average_marks_per_level"]:
                    summary_data.append([f"{level} Avg Marks", summary["average_marks_per_level"][level]])
            
            summary_data.append(["", ""])
            summary_data.append(["Question Distribution", ""])
            for level in self.difficulty_labels:
                summary_data.append([f"{level} Questions", summary["questions_per_difficulty"][level]])
            
            summary_df = pd.DataFrame(summary_data)
            actual_summary_path = safe_save_excel(summary_df, summary_path, index=False, header=False)
            logger.info(f"Summary saved to {actual_summary_path}")
            
            # Update the output path for final message
            self.output_path = actual_output_path
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def run(self, questions_per_student: int = 3) -> None:
        """Run the complete AI question assignment process"""
        try:
            logger.info("🚀 Starting AI-based Question Assignment System")
            
            # Load and validate data
            marksheet, questions_df = self.load_and_validate_data()
            
            # Cluster students by difficulty
            logger.info("📊 Clustering students by difficulty level...")
            marksheet = self.cluster_student_difficulty(marksheet)
            
            # Classify questions by difficulty
            logger.info("🤖 Classifying questions using AI...")
            questions_df = self.classify_question_difficulty(questions_df)
            
            # Assign questions to students
            logger.info("📝 Assigning questions to students...")
            assigned_df = self.assign_questions_to_students(
                marksheet, questions_df, questions_per_student
            )
            
            # Generate summary
            summary = self.generate_summary_report(marksheet, questions_df, assigned_df)
            
            # Save results
            self.save_results(assigned_df, summary)
            
            # Print summary
            print("\n✅ AI-based difficulty assignment completed!")
            print(f"📈 Summary:")
            print(f"  • Total Students: {summary['total_students']}")
            print(f"  • Total Questions: {summary['total_questions']}")
            print(f"  • Student Distribution: {summary['difficulty_distribution']}")
            print(f"  • Question Distribution: {summary['questions_per_difficulty']}")
            print(f"📁 Results saved to: {self.output_path}")
            
        except Exception as e:
            logger.error(f"System failed: {str(e)}")
            raise

# Usage example
if __name__ == "__main__":
    # Initialize the system
    system = AIQuestionAssignmentSystem(
        marksheet_path="marksheet.xlsx",
        questions_path="questions.xlsx", 
        output_path="assigned_questions.xlsx"
    )
    
    # Run the complete process
    system.run(questions_per_student=3)