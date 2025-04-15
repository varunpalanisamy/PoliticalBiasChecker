import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv("preprocessed.csv")

X = data["content"] 
y = data["bias"] 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

max_features = 50000
print(f"\nTesting max_features={max_features}...")

tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 3), stop_words='english')

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train_tfidf, y_train)

y_pred = logreg_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for max_features={max_features}: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Left", "Center", "Right"]))

joblib.dump(logreg_model, 'logreg_model.pkl') 
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')  
print("\nModel and TF-IDF vectorizer have been saved successfully")

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Left", "Center", "Right"], yticklabels=["Left", "Center", "Right"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (max_features={max_features})")

plt.savefig('confusion_matrix.png')
plt.close()
