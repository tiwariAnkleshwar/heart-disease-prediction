import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Load dataset
df = pd.read_csv('data/heart.csv')

# One-hot encoding for categorical columns
df = pd.get_dummies(df)

# Define features and label
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Save feature columns for later use in app.py
feature_columns = X.columns.tolist()
print("Feature columns used for training:", feature_columns)

# Save feature columns to a file
with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and print accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)
