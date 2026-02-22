import pandas as pd
import random
from transformers import pipeline
from sklearn.cluster import KMeans
import numpy as np

# Load student marksheet
marksheet = pd.read_excel("marksheet.xlsx")  # Must have 'Name' and 'Marks'
if "Name" not in marksheet.columns or "Marks" not in marksheet.columns:
    raise ValueError("Marksheet must have 'Name' and 'Marks' columns.")

# Cluster student marks into 3 clusters (difficulty levels)
kmeans = KMeans(n_clusters=3, random_state=42)
marks = marksheet["Marks"].values.reshape(-1,1)
marksheet["Difficulty_Cluster"] = kmeans.fit_predict(marks)

# Map clusters to difficulty labels by ordering cluster centers
cluster_centers = kmeans.cluster_centers_.flatten()
sorted_centers = np.argsort(cluster_centers)  # low to high marks cluster indices

# Create a mapping from cluster index to difficulty label
difficulty_map = {}
difficulty_labels = ["Easy", "Medium", "Hard"]
for rank, cluster_idx in enumerate(sorted_centers):
    difficulty_map[cluster_idx] = difficulty_labels[rank]

# Apply mapping to assign difficulty labels
marksheet["Student_Difficulty"] = marksheet["Difficulty_Cluster"].map(difficulty_map)

# Load question bank
questions_df = pd.read_excel("questions.xlsx")  # Must have 'Question' column
if "Question" not in questions_df.columns:
    raise ValueError("Questions file must have 'Question' column.")

# Zero-shot classify question difficulty
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["Easy", "Medium", "Hard"]

predicted_difficulties = []
for question in questions_df["Question"]:
    result = classifier(question, candidate_labels)
    predicted_difficulties.append(result["labels"][0])

questions_df["Question_Difficulty"] = predicted_difficulties

# Assign questions matching student difficulty
assigned_data = []
for _, student in marksheet.iterrows():
    level = student["Student_Difficulty"]
    matching_questions = questions_df[questions_df["Question_Difficulty"] == level]["Question"].tolist()
    
    if not matching_questions:
        assigned_questions = ["No questions available"]
    else:
        assigned_questions = random.sample(matching_questions, min(2, len(matching_questions)))
    
    assigned_data.append({
        "Name": student["Name"],
        "Marks": student["Marks"],
        "Difficulty_Level": level,
        "Assigned_Questions": " | ".join(assigned_questions)
    })

assigned_df = pd.DataFrame(assigned_data)
assigned_df.to_excel("assigned_questions.xlsx", index=False)

print("✅ AI-based difficulty assignment done! Check assigned_questions.xlsx")
