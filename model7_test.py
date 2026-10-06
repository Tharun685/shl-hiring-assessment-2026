import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATASET = r"C:\Users\tharu\Downloads\shl-hiring-assessment-2026\Dataset_Final"

# Load original 48 features
original = pd.read_csv(
    os.path.join(DATASET, "train_features.csv")
)

# Load speech features
speech = pd.read_csv(
    os.path.join(DATASET, "train_speech_features.csv")
)

# Remove duplicate columns before merging
speech_features = speech.drop(columns=["label"])

# Merge using filename
df = original.merge(
    speech_features,
    on="filename",
    how="inner"
)

print("Combined dataset shape:", df.shape)

# Separate features and target
X = df.drop(columns=["filename", "label"])
y = df["label"]

print("Number of features:", X.shape[1])

# Same validation split used before
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Best Extra Trees configuration tested so far
model = ExtraTreesRegressor(
    n_estimators=500,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Model #7...")

model.fit(X_train, y_train)

predictions = model.predict(X_val)

# Evaluation
mae = mean_absolute_error(y_val, predictions)
rmse = mean_squared_error(y_val, predictions) ** 0.5
r2 = r2_score(y_val, predictions)

print("\n========== MODEL #7 RESULTS ==========")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

print("\nPrevious best local result:")
print("MAE  : 0.5506")
print("RMSE : 0.6907")
print("R2   : 0.7399")

print("\nCurrent best Kaggle score: 0.7396")