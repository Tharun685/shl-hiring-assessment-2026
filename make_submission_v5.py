import os
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

DATASET = r"C:\Users\tharu\Downloads\shl-hiring-assessment-2026\Dataset_Final"

# Load original features
train_original = pd.read_csv(
    os.path.join(DATASET, "train_features.csv")
)

test_original = pd.read_csv(
    os.path.join(DATASET, "test_features.csv")
)

# Load speech features
train_speech = pd.read_csv(
    os.path.join(DATASET, "train_speech_features.csv")
)

test_speech = pd.read_csv(
    os.path.join(DATASET, "test_speech_features.csv")
)

# Remove duplicate label column from speech features
train_speech_features = train_speech.drop(columns=["label"])

# Merge original + speech features
train = train_original.merge(
    train_speech_features,
    on="filename",
    how="inner"
)

test = test_original.merge(
    test_speech,
    on="filename",
    how="inner"
)

print("Combined training shape:", train.shape)
print("Combined test shape:", test.shape)

# Separate features and target
X_train = train.drop(columns=["filename", "label"])
y_train = train["label"]

X_test = test.drop(columns=["filename"])

print("Training features:", X_train.shape[1])
print("Test features:", X_test.shape[1])

# Model #7
model = ExtraTreesRegressor(
    n_estimators=500,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)

print("\nTraining final Model #7...")

model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    "filename": test["filename"],
    "label": predictions
})

output_file = os.path.join(
    DATASET,
    "submission_v5.csv"
)

submission.to_csv(
    output_file,
    index=False
)

print("\nSubmission created!")
print("Shape:", submission.shape)
print("Saved to:", output_file)

print("\nFirst 10 predictions:")
print(submission.head(10))

print("\nPrediction range:")
print("Minimum:", round(predictions.min(), 3))
print("Maximum:", round(predictions.max(), 3))