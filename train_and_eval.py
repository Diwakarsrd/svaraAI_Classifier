import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = str(text).lower()
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    # Strip whitespace
    text = text.strip()
    return text

# -------------------------
# Load and preprocess dataset
# -------------------------
print("Loading and preprocessing dataset...")
file_path = "data/reply_classification_dataset.csv"
df = pd.read_csv(file_path)  # Use comma separator as it's a standard CSV

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Clean dataset
if "label" not in df.columns or "text" not in df.columns:
    raise ValueError("Dataset must contain 'text' and 'label' columns")

# Handle missing values
initial_size = len(df)
df = df.dropna(subset=["text", "label"])
print(f"Removed {initial_size - len(df)} rows with missing values")

# Preprocess text
df["text"] = df["text"].apply(preprocess_text)

# Standardize labels (handle inconsistent casing)
df["label"] = df["label"].str.lower().str.strip()
print(f"Unique labels after cleaning: {df['label'].unique()}")

# Encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# -------------------------
# Train / Test split
# -------------------------
X = df["text"]
y = df["label_encoded"]
y_labels = df["label"]

X_train, X_test, y_train, y_test, y_labels_train, y_labels_test = train_test_split(
    X, y, y_labels, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training label distribution:\n{pd.Series(y_labels_train).value_counts()}")

# -------------------------
# Feature Engineering
# -------------------------
print("\nCreating TF-IDF features...")
vectorizer = TfidfVectorizer(
    stop_words="english", 
    max_features=5000,
    ngram_range=(1, 2),  # Include bigrams
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.8  # Ignore terms that appear in more than 80% of documents
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")

# -------------------------
# Model Training and Evaluation (Basic models only)
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    results[name] = {
        'accuracy': accuracy,
        'f1_score': f1,
        'model': model
    }
    
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    if f1 > best_score:
        best_score = f1
        best_model = (name, model)

# Save the best model
print(f"\n Best model: {best_model[0]} with F1 score: {best_score:.4f}")
joblib.dump(best_model[1], "models/best_baseline_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
print(" Best model, vectorizer, and label encoder saved!")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print("Models and artifacts saved:")
print("- models/best_baseline_model.pkl")
print("- models/tfidf_vectorizer.pkl")
print("- models/label_encoder.pkl")