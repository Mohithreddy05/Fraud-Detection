import pandas as pd
import spacy
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

nlp = spacy.load("en_core_web_sm")

# ---------- PREPROCESS ----------
def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if not t.is_stop and t.is_alpha])

# ---------- LOAD ----------
df = pd.read_csv("fraud_dataset_large.csv")

# ---------- SPLIT ----------
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

# ---------- PREPROCESS ----------
train_df["text"] = train_df["text"].apply(preprocess)
test_df["text"] = test_df["text"].apply(preprocess)

# ---------- ENCODE ----------
train_df["label"] = train_df["label"].map({"normal": 0, "fraud": 1})
test_df["label"] = test_df["label"].map({"normal": 0, "fraud": 1})

# ---------- VECTORIZE ----------
vectorizer = TfidfVectorizer(
    ngram_range=(1,1),
    max_features=3000,
    min_df=5,
    max_df=0.8
)

X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

# ---------- MODEL ----------
model = LogisticRegression(max_iter=2000, C=0.5)
model.fit(X_train, train_df["label"])

# ---------- CROSS VALIDATION ----------
scores = cross_val_score(model, X_train, train_df["label"], cv=5)
print("Cross-validation accuracy:", scores.mean())

# ---------- TEST ----------
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(test_df["label"], y_pred))

# ---------- SAVE ----------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel saved successfully!")