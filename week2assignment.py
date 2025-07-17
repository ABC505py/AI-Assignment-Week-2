# disease_outbreak_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load your Kaggle dataset
df = pd.read_csv("your_disease_outbreak_dataset.csv")  # Replace with your actual file name

# Basic preprocessing
df.dropna(inplace=True)  # Clean missing values
X = df.drop(columns=["outbreak"])  # Features
y = df["outbreak"]  # Target variable: 1 = outbreak, 0 = no outbreak

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
