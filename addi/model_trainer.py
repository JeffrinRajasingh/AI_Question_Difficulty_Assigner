# model_trainer.py - Train and save models for question assignment
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StudentDifficultyTrainer:
    """Train student difficulty prediction model"""
    
    def __init__(self):
        self.model = None
        self.encoder = None
    
    def create_synthetic_data(self, n_samples=5000):
        """Create synthetic student data for training"""
        np.random.seed(42)
        
        # Train student model
    accuracy = student_trainer.train_model(student_data, model_type='random_forest')
    student_trainer.save_model()
    
    # Train question difficulty model
    logger.info("=" * 50)
    logger.info("Training Question Difficulty Model")
    logger.info("=" * 50)
    
    question_trainer = QuestionDifficultyTrainer()
    
    # Try to load existing question data, otherwise create synthetic
    try:
        logger.info("Looking for question_data.xlsx...")
        question_data = question_trainer.load_from_excel("question_data.xlsx")
        logger.info(f"Loaded {len(question_data)} questions from Excel")
        
        # Train custom classifier if data is available
        logger.info("Training custom question classifier with loaded data...")
        question_trainer.train_sklearn_classifier(question_data)
        
    except Exception as e:
        logger.info(f"No Excel file found or error loading: {e}")
        logger.info("Creating synthetic question data...")
        
        # Option 1: Train transformer model (computationally expensive)
        train_transformer = input("Train transformer model? (requires GPU, takes time) [y/N]: ").lower().strip()
        if train_transformer == 'y':
            try:
                question_trainer.train_transformer_model()
                logger.info("✅ Transformer model trained successfully!")
            except Exception as e:
                logger.error(f"Transformer training failed: {e}")
                logger.info("Falling back to scikit-learn model...")
                question_data = question_trainer.create_synthetic_questions()
                question_trainer.train_sklearn_classifier(question_data)
        else:
            # Option 2: Train scikit-learn model (faster, good performance)
            logger.info("Training scikit-learn classifier with synthetic data...")
            question_data = question_trainer.create_synthetic_questions()
            question_trainer.train_sklearn_classifier(question_data)
    
    # Create rule-based classifier info as fallback
    logger.info("Creating rule-based question classifier info...")
    rule_info = {
        "model_type": "rule_based",
        "easy_keywords": ['what', 'who', 'when', 'where', 'list', 'name', 'identify', 'define', 'recall'],
        "medium_keywords": ['explain', 'describe', 'apply', 'solve', 'calculate', 'determine', 'find', 'compute'],
        "hard_keywords": ['analyze', 'evaluate', 'compare', 'synthesize', 'derive', 'prove', 'assess', 'critique'],
        "created_date": pd.Timestamp.now().isoformat(),
        "version": "1.0"
    }
    
    import json
    with open("models/question_classifier_info.json", "w") as f:
        json.dump(rule_info, f, indent=2)
    
    logger.info("=" * 50)
    logger.info("Model Training Completed Successfully!")
    logger.info("=" * 50)
    logger.info(f"Student model accuracy: {accuracy:.3f}")
    logger.info("Files created:")
    logger.info("- models/student_difficulty_model.pkl")
    logger.info("- models/student_label_encoder.pkl")
    logger.info("- models/question_classifier_info.json")
    
    # Check what question models were created
    if os.path.exists("models/question_difficulty_classifier"):
        logger.info("- models/question_difficulty_classifier/ (Transformer)")
    if os.path.exists("models/question_sklearn_model.pkl"):
        logger.info("- models/question_sklearn_model.pkl")
        logger.info("- models/question_vectorizer.pkl")
    
    logger.info("=" * 50)

# Additional methods for QuestionDifficultyTrainer class
def add_sklearn_methods():
    """Add sklearn-based training methods to QuestionDifficultyTrainer"""
    
    def load_from_excel(self, filepath):
        """Load question training data from Excel file"""
        df = pd.read_excel(filepath)
        required_cols = ['question', 'difficulty']
        
        if not all(col in df.columns for col in required_cols):
            # Try alternative column names
            if 'Question' in df.columns:
                df['question'] = df['Question']
            if 'Difficulty' in df.columns:
                df['difficulty'] = df['Difficulty']
            elif 'Level' in df.columns:
                df['difficulty'] = df['Level']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Excel file must contain columns: {required_cols} (or Question, Difficulty)")
        
        # Clean and validate data
        df = df.dropna(subset=['question', 'difficulty'])
        df['difficulty'] = df['difficulty'].str.title()  # Easy, Medium, Hard
        
        # Filter valid difficulties
        valid_difficulties = ['Easy', 'Medium', 'Hard']
        df = df[df['difficulty'].isin(valid_difficulties)]
        
        return df
    
    def extract_features(self, questions):
        """Extract features from questions for sklearn models"""
        features = []
        
        for question in questions:
            question_lower = question.lower()
            
            # Basic text features
            word_count = len(question.split())
            char_count = len(question)
            sentence_count = question.count('.') + question.count('!') + question.count('?')
            
            # Keyword features
            easy_keywords = ['what', 'who', 'when', 'where', 'list', 'name', 'identify', 'define', 'recall', 'state']
            medium_keywords = ['explain', 'describe', 'apply', 'solve', 'calculate', 'determine', 'find', 'compute', 'show']
            hard_keywords = ['analyze', 'evaluate', 'compare', 'synthesize', 'derive', 'prove', 'assess', 'critique', 'justify']
            
            easy_score = sum(1 for keyword in easy_keywords if keyword in question_lower)
            medium_score = sum(1 for keyword in medium_keywords if keyword in question_lower)
            hard_score = sum(1 for keyword in hard_keywords if keyword in question_lower)
            
            # Question type features
            is_wh_question = any(question_lower.startswith(wh) for wh in ['what', 'who', 'when', 'where', 'why', 'how'])
            has_calculation = any(symbol in question for symbol in ['+', '-', '*', '/', '=', '%'])
            has_math_symbols = any(symbol in question for symbol in ['∫', '∑', '∏', '√', '±', '≤', '≥', '≠', '∞', '∂'])
            
            # Complexity indicators
            has_multiple_parts = question.count('and') > 1 or question.count(',') > 2
            has_comparison = any(word in question_lower for word in ['compare', 'contrast', 'versus', 'vs', 'difference'])
            
            features.append([
                word_count, char_count, sentence_count,
                easy_score, medium_score, hard_score,
                int(is_wh_question), int(has_calculation), int(has_math_symbols),
                int(has_multiple_parts), int(has_comparison)
            ])
        
        return np.array(features)
    
    def train_sklearn_classifier(self, df):
        """Train sklearn-based question classifier"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        
        logger.info("Training sklearn-based question classifier...")
        
        # Prepare data
        X_text = df['question'].values
        X_features = self.extract_features(df['question'].values)
        y = df['difficulty'].values
        
        # Create TF-IDF vectorizer for text
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        # Combine text features with manual features
        X_text_features = vectorizer.fit_transform(X_text).toarray()
        X_combined = np.hstack([X_text_features, X_features])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Sklearn model accuracy: {accuracy:.3f}")
        logger.info("Classification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        # Save model and vectorizer
        joblib.dump(model, "models/question_sklearn_model.pkl")
        joblib.dump(vectorizer, "models/question_vectorizer.pkl")
        
        # Save feature names for later use
        feature_info = {
            "text_features": vectorizer.get_feature_names_out().tolist(),
            "manual_features": [
                "word_count", "char_count", "sentence_count",
                "easy_score", "medium_score", "hard_score",
                "is_wh_question", "has_calculation", "has_math_symbols",
                "has_multiple_parts", "has_comparison"
            ],
            "classes": model.classes_.tolist(),
            "accuracy": accuracy
        }
        
        with open("models/sklearn_model_info.json", "w") as f:
            json.dump(feature_info, f, indent=2)
        
        logger.info("✅ Sklearn model saved successfully!")
        return accuracy
    
    # Add methods to the class
    QuestionDifficultyTrainer.load_from_excel = load_from_excel
    QuestionDifficultyTrainer.extract_features = extract_features
    QuestionDifficultyTrainer.train_sklearn_classifier = train_sklearn_classifier

# Add the methods to the class
add_sklearn_methods()

def create_sample_training_data():
    """Create sample Excel files for training"""
    logger.info("Creating sample training data files...")
    
    # Enhanced sample data for better training
    students_data = []
    np.random.seed(42)
    
    # Generate realistic student data
    for i in range(200):
        if i < 80:  # Easy students (40%)
            marks = max(0, min(100, np.random.normal(40, 15)))
            difficulty = 'Easy'
        elif i < 160:  # Medium students (40%)
            marks = max(0, min(100, np.random.normal(70, 12)))
            difficulty = 'Medium'
        else:  # Hard students (20%)
            marks = max(0, min(100, np.random.normal(88, 8)))
            difficulty = 'Hard'
        
        students_data.append({
            'Name': f'Student_{i+1:03d}',
            'Marks': round(marks, 1),
            'Difficulty': difficulty,
            'Subject': np.random.choice(['Math', 'Science', 'English', 'History'])
        })
    
    # Enhanced question data with more examples
    easy_questions = [
        "What is the capital of France?", "Who wrote Romeo and Juliet?",
        "What is 5 + 7?", "Name the largest planet.", "Define photosynthesis.",
        "List three primary colors.", "What is H2O?", "Who painted Mona Lisa?",
        "What year did WWII end?", "Name the smallest country.",
        "What is the chemical symbol for gold?", "Define gravity.",
        "List the days of the week.", "What is 12 x 3?",
        "Name the longest river.", "What is a noun?"
    ]
    
    medium_questions = [
        "Explain the process of photosynthesis.", "Calculate 15% of 200.",
        "Describe the water cycle.", "Solve: 2x + 5 = 15",
        "Explain how vaccines work.", "Find the area of a circle with radius 5.",
        "Describe cellular respiration.", "Calculate compound interest.",
        "Explain the causes of WWI.", "Determine the slope of y = 2x + 3.",
        "Describe the structure of an atom.", "Solve: x² - 4x + 3 = 0",
        "Explain supply and demand.", "Calculate the volume of a cylinder."
    ]
    
    hard_questions = [
        "Analyze the impact of climate change on biodiversity.",
        "Evaluate the effectiveness of renewable energy policies.",
        "Compare Keynesian and Austrian economic theories.",
        "Derive the quadratic formula from ax² + bx + c = 0.",
        "Analyze the causes of the 2008 financial crisis.",
        "Evaluate the ethical implications of genetic engineering.",
        "Compare the foreign policies of Roosevelt and Wilson.",
        "Analyze the relationship between quantum mechanics and relativity.",
        "Evaluate the impact of social media on democracy.",
        "Synthesize arguments for and against globalization."
    ]
    
    questions_data = []
    for q in easy_questions:
        questions_data.append({'Question': q, 'Difficulty': 'Easy', 'Subject': 'General'})
    for q in medium_questions:
        questions_data.append({'Question': q, 'Difficulty': 'Medium', 'Subject': 'General'})
    for q in hard_questions:
        questions_data.append({'Question': q, 'Difficulty': 'Hard', 'Subject': 'General'})
    
    # Save to Excel
    students_df = pd.DataFrame(students_data)
    students_df.to_excel("student_data.xlsx", index=False)
    
    questions_df = pd.DataFrame(questions_data)
    questions_df.to_excel("question_data.xlsx", index=False)
    
    logger.info("✅ Sample training data created:")
    logger.info("- student_data.xlsx (200 students)")
    logger.info("- question_data.xlsx (40+ questions)")
    
    return students_df, questions_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train models for question assignment system')
    parser.add_argument('--create-sample-data', action='store_true', 
                       help='Create sample training data files')
    parser.add_argument('--train-transformer', action='store_true',
                       help='Train transformer model (requires GPU)')
    parser.add_argument('--quick-train', action='store_true',
                       help='Quick training with sklearn models only')
    
    args = parser.parse_args()
    
    if args.create_sample_data:
        create_sample_training_data()
        print("\n✅ Sample data created! You can now run training.")
        print("Next steps:")
        print("1. python model_trainer.py --quick-train")
        print("2. python model_trainer.py --train-transformer (if you have GPU)")
        print("3. python model_trainer.py (interactive mode)")
    else:
        # Run main training
        main() Generate realistic mark distributions
        # Easy students: mostly low marks with some outliers
        easy_marks = np.concatenate([
            np.random.normal(35, 15, int(n_samples * 0.4)),
            np.random.uniform(0, 60, int(n_samples * 0.1))
        ])
        easy_marks = np.clip(easy_marks, 0, 100)
        
        # Medium students: normal distribution around 70
        medium_marks = np.random.normal(70, 12, int(n_samples * 0.4))
        medium_marks = np.clip(medium_marks, 0, 100)
        
        # Hard students: high marks with some variation
        hard_marks = np.concatenate([
            np.random.normal(85, 8, int(n_samples * 0.15)),
            np.random.uniform(80, 100, int(n_samples * 0.05))
        ])
        hard_marks = np.clip(hard_marks, 0, 100)
        
        # Combine data
        marks = np.concatenate([easy_marks, medium_marks, hard_marks])
        difficulties = (['Easy'] * len(easy_marks) + 
                       ['Medium'] * len(medium_marks) + 
                       ['Hard'] * len(hard_marks))
        
        # Add some noise and edge cases
        for i in range(len(marks)):
            if 55 <= marks[i] <= 65:  # Boundary region
                if np.random.random() < 0.2:
                    difficulties[i] = np.random.choice(['Easy', 'Medium'])
            elif 75 <= marks[i] <= 85:  # Another boundary
                if np.random.random() < 0.2:
                    difficulties[i] = np.random.choice(['Medium', 'Hard'])
        
        return pd.DataFrame({
            'marks': marks,
            'difficulty': difficulties
        })
    
    def train_model(self, df=None, model_type='random_forest'):
        """Train student difficulty model"""
        if df is None:
            logger.info("Creating synthetic training data...")
            df = self.create_synthetic_data()
        
        # Prepare data
        X = df[['marks']]
        y = df['difficulty']
        
        # Encode labels
        self.encoder = LabelEncoder()
        y_encoded = self.encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            self.model = DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            )
        
        logger.info(f"Training {model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.3f}")
        logger.info("Classification Report:")
        logger.info(classification_report(
            y_test, y_pred, 
            target_names=self.encoder.classes_
        ))
        
        return accuracy
    
    def save_model(self, model_path="models"):
        """Save trained model and encoder"""
        os.makedirs(model_path, exist_ok=True)
        
        joblib.dump(self.model, f"{model_path}/student_difficulty_model.pkl")
        joblib.dump(self.encoder, f"{model_path}/student_label_encoder.pkl")
        
        logger.info(f"Model saved to {model_path}/")
    
    def load_from_excel(self, filepath):
        """Load training data from Excel file"""
        df = pd.read_excel(filepath)
        required_cols = ['marks', 'difficulty']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Excel file must contain columns: {required_cols}")
        
        return df

class QuestionDifficultyTrainer:
    """Train question difficulty classification model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    def create_synthetic_questions(self):
        """Create synthetic question data for training"""
        easy_questions = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What year did World War II end?",
            "Name the largest planet in our solar system.",
            "What is 5 + 7?",
            "Define photosynthesis.",
            "List the primary colors.",
            "What is the chemical symbol for water?",
            "Who painted the Mona Lisa?",
            "What is the boiling point of water?",
            "Name three types of rocks.",
            "What does CPU stand for?",
            "Who was the first president of the United States?",
            "What is the smallest unit of matter?",
            "Identify the verb in this sentence: 'The dog runs fast.'",
        ]
        
        medium_questions = [
            "Explain the process of photosynthesis in plants.",
            "Calculate the area of a circle with radius 5cm.",
            "Describe the causes of the American Civil War.",
            "Solve for x: 2x + 5 = 15",
            "Explain how vaccines work in the human body.",
            "Calculate the compound interest on $1000 at 5% for 3 years.",
            "Describe the water cycle and its importance.",
            "Find the derivative of f(x) = x² + 3x - 2",
            "Explain the difference between mitosis and meiosis.",
            "Calculate the momentum of a 10kg object moving at 5m/s.",
            "Describe the structure and function of DNA.",
            "Solve the quadratic equation: x² - 4x + 3 = 0",
            "Explain the concept of supply and demand in economics.",
        ]
        
        hard_questions = [
            "Analyze the impact of globalization on developing economies.",
            "Evaluate the effectiveness of different renewable energy sources.",
            "Compare and contrast the philosophies of Kant and Nietzsche.",
            "Derive the quadratic formula from the general quadratic equation.",
            "Analyze the causes and long-term effects of climate change.",
            "Evaluate the role of artificial intelligence in modern healthcare.",
            "Compare the molecular mechanisms of different types of cancer.",
            "Synthesize a comprehensive argument for or against genetic engineering.",
            "Analyze the political and social factors that led to World War I.",
            "Evaluate the impact of social media on democratic processes.",
            "Derive and prove the Pythagorean theorem using geometric methods.",
            "Analyze the relationship between quantum mechanics and classical physics.",
            "Evaluate competing theories about the origin of consciousness.",
        ]
        
        # Create dataset
        questions = []
        labels = []
        
        for q in easy_questions:
            questions.extend([q] * 5)  # Duplicate for more training data
            labels.extend(['Easy'] * 5)
        
        for q in medium_questions:
            questions.extend([q] * 5)
            labels.extend(['Medium'] * 5)
        
        for q in hard_questions:
            questions.extend([q] * 5)
            labels.extend(['Hard'] * 5)
        
        return pd.DataFrame({
            'question': questions,
            'difficulty': labels
        })
    
    def prepare_dataset_for_training(self, df):
        """Prepare dataset for Hugging Face training"""
        label_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}
        df['labels'] = df['difficulty'].map(label_map)
        
        dataset = Dataset.from_pandas(df[['question', 'labels']])
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['question'],
                truncation=True,
                padding='max_length',
                max_length=256
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def train_transformer_model(self, model_name="distilbert-base-uncased"):
        """Train a transformer model for question classification"""
        from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
        
        logger.info(f"Training transformer model: {model_name}")
        
        # Create synthetic data
        df = self.create_synthetic_questions()
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label={0: 'Easy', 1: 'Medium', 2: 'Hard'},
            label2id={'Easy': 0, 'Medium': 1, 'Hard': 2}
        )
        
        # Prepare dataset
        dataset = self.prepare_dataset_for_training(df)
        train_dataset, eval_dataset = dataset.train_test_split(test_size=0.2).values()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./models/question_difficulty_classifier',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model('./models/question_difficulty_classifier')
        self.tokenizer.save_pretrained('./models/question_difficulty_classifier')
        
        logger.info("Transformer model training completed!")

def main():
    """Main training function"""
    logger.info("Starting model training process...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train student difficulty model
    logger.info("=" * 50)
    logger.info("Training Student Difficulty Model")
    logger.info("=" * 50)
    
    student_trainer = StudentDifficultyTrainer()
    
    # Try to load existing data, otherwise create synthetic
    try:
        logger.info("Looking for student_data.xlsx...")
        student_data = student_trainer.load_from_excel("student_data.xlsx")
        logger.info(f"Loaded {len(student_data)} student records from Excel")
    except Exception as e:
        logger.info(f"No Excel file found or error loading: {e}")
        logger.info("Creating synthetic data...")
        student_data = None
    
    # Train student model
    accuracy = student_trainer.train_model(student_data, model_type='random_forest')
    student_trainer.save_model()
    
    # Train question difficulty model
    logger.info("=" * 50)
    logger.info("Training Question Difficulty Model")
    logger.info("=" * 50)
    
    question_trainer = QuestionDifficultyTrainer()
    
    # Try to load existing question data, otherwise create synthetic
    try:
        logger.info("Looking for question_data.xlsx...")
        question_data = question_trainer.load_from_excel("question_data.xlsx")
        logger.info(f"Loaded {len(question_data)} questions from Excel")
        
        # Train custom classifier if data is available
        logger.info("Training custom question classifier with loaded data...")
        question_trainer.train_sklearn_classifier(question_data)
        
    except Exception as e:
        logger.info(f"No Excel file found or error loading: {e}")
        logger.info("Creating synthetic question data...")
        
        # Option 1: Train transformer model (computationally expensive)
        train_transformer = input("Train transformer model? (requires GPU, takes time) [y/N]: ").lower().strip()
        if train_transformer == 'y':
            try:
                question_trainer.train_transformer_model()
                logger.info("✅ Transformer model trained successfully!")
            except Exception as e:
                logger.error(f"Transformer training failed: {e}")
                logger.info("Falling back to scikit-learn model...")
                question_data = question_trainer.create_synthetic_questions()
                question_trainer.train_sklearn_classifier(question_data)
        else:
            # Option 2: Train scikit-learn model (faster, good performance)
            logger.info("Training scikit-learn classifier with synthetic data...")
            question_data = question_trainer.create_synthetic_questions()
            question_trainer.train_sklearn_classifier(question_data)
    
    # Create rule-based classifier info as fallback
    logger.info("Creating rule-based question classifier info...")
    rule_info = {
        "model_type": "rule_based",
        "easy_keywords": ['what', 'who', 'when', 'where', 'list', 'name', 'identify', 'define', 'recall'],
        "medium_keywords": ['explain', 'describe', 'apply', 'solve', 'calculate', 'determine', 'find', 'compute'],
        "hard_keywords": ['analyze', 'evaluate', 'compare', 'synthesize', 'derive', 'prove', 'assess', 'critique'],
        "created_date": pd.Timestamp.now().isoformat(),
        "version": "1.0"
    }
    
    import json
    with open("models/question_classifier_info.json", "w") as f:
        json.dump(rule_info, f, indent=2)
    
    logger.info("=" * 50)
    logger.info("Model Training Completed Successfully!")
    logger.info("=" * 50)
    logger.info(f"Student model accuracy: {accuracy:.3f}")
    logger.info("Files created:")
    logger.info("- models/student_difficulty_model.pkl")
    logger.info("- models/student_label_encoder.pkl")
    logger.info("- models/question_classifier_info.json")
    
    # Check what question models were created
    if os.path.exists("models/question_difficulty_classifier"):
        logger.info("- models/question_difficulty_classifier/ (Transformer)")
    if os.path.exists("models/question_sklearn_model.pkl"):
        logger.info("- models/question_sklearn_model.pkl")
        logger.info("- models/question_vectorizer.pkl")
    
    logger.info("=" * 50)
    logger.info("🎉 Training completed! You can now start the server:")
    logger.info("   python run_server.py")
    logger.info("=" * 50)