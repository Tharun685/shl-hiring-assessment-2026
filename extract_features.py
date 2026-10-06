import os
import pandas as pd
import numpy as np
import librosa

# -----------------------------
# Paths
# -----------------------------
TRAIN_DIR = "train"
CSV_FILE = "train.csv"
OUTPUT_FILE = "train_features.csv"


# -----------------------------
# Extract features from one audio file
# -----------------------------
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=16000)

    features = []

    # 1. MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 2. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)

    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # 3. RMS Energy
    rms = librosa.feature.rms(y=y)

    features.append(np.mean(rms))
    features.append(np.std(rms))

    # 4. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features.append(np.mean(spectral_centroid))
    features.append(np.std(spectral_centroid))

    # 5. Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features.append(np.mean(spectral_bandwidth))
    features.append(np.std(spectral_bandwidth))

    return features


# -----------------------------
# Read training CSV
# -----------------------------
df = pd.read_csv(CSV_FILE)

all_features = []

print("Starting feature extraction...")
print("Total files:", len(df))

# -----------------------------
# Process every audio file
# -----------------------------
for i, filename in enumerate(df["filename"]):

    file_path = os.path.join(TRAIN_DIR, filename)

    try:
        features = extract_features(file_path)

        row = [filename] + features + [df.loc[i, "label"]]

        all_features.append(row)

    except Exception as e:
        print("Error processing:", filename)
        print(e)

    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(df)} files")


# -----------------------------
# Create column names
# -----------------------------
feature_names = []

for i in range(20):
    feature_names.append(f"mfcc_mean_{i+1}")

for i in range(20):
    feature_names.append(f"mfcc_std_{i+1}")

feature_names += [
    "zcr_mean",
    "zcr_std",
    "rms_mean",
    "rms_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std"
]

columns = ["filename"] + feature_names + ["label"]


# -----------------------------
# Save features
# -----------------------------
features_df = pd.DataFrame(
    all_features,
    columns=columns
)

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Feature extraction completed!")
print("Saved:", OUTPUT_FILE)
print("Shape:", features_df.shape)
print(features_df.head())