import os
import pandas as pd
import numpy as np
import librosa

DATASET = r"C:\Users\tharu\Downloads\shl-hiring-assessment-2026\Dataset_Final"


def extract_speech_features(file_path):

    y, sr = librosa.load(file_path, sr=16000)

    features = {}

    # ------------------------------------------------
    # 1. Pitch / Fundamental Frequency
    # ------------------------------------------------

    f0 = librosa.yin(
        y,
        fmin=70,
        fmax=400,
        sr=sr
    )

    # Remove invalid values
    f0 = f0[np.isfinite(f0)]

    if len(f0) > 0:
        features["pitch_mean"] = np.mean(f0)
        features["pitch_std"] = np.std(f0)
        features["pitch_min"] = np.min(f0)
        features["pitch_max"] = np.max(f0)
        features["pitch_median"] = np.median(f0)
    else:
        features["pitch_mean"] = 0
        features["pitch_std"] = 0
        features["pitch_min"] = 0
        features["pitch_max"] = 0
        features["pitch_median"] = 0

    # ------------------------------------------------
    # 2. Spectral Rolloff
    # ------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        roll_percent=0.85
    )

    features["rolloff_mean"] = np.mean(rolloff)
    features["rolloff_std"] = np.std(rolloff)

    # ------------------------------------------------
    # 3. Spectral Flatness
    # ------------------------------------------------

    flatness = librosa.feature.spectral_flatness(y=y)

    features["flatness_mean"] = np.mean(flatness)
    features["flatness_std"] = np.std(flatness)

    # ------------------------------------------------
    # 4. RMS Energy
    # ------------------------------------------------

    rms = librosa.feature.rms(y=y)

    features["energy_mean"] = np.mean(rms)
    features["energy_std"] = np.std(rms)
    features["energy_min"] = np.min(rms)
    features["energy_max"] = np.max(rms)

    # ------------------------------------------------
    # 5. Energy percentiles
    # ------------------------------------------------

    features["energy_25"] = np.percentile(rms, 25)
    features["energy_50"] = np.percentile(rms, 50)
    features["energy_75"] = np.percentile(rms, 75)

    # ------------------------------------------------
    # 6. Zero Crossing Rate statistics
    # ------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(y)

    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)
    features["zcr_min"] = np.min(zcr)
    features["zcr_max"] = np.max(zcr)

    return features


# ====================================================
# TRAIN DATA
# ====================================================

train_csv = os.path.join(DATASET, "train.csv")
train_audio = os.path.join(DATASET, "train")

train_df = pd.read_csv(train_csv)

all_features = []

for i, row in train_df.iterrows():

    file_path = os.path.join(
        train_audio,
        row["filename"]
    )

    try:

        features = extract_speech_features(file_path)

        features["filename"] = row["filename"]
        features["label"] = row["label"]

        all_features.append(features)

    except Exception as e:

        print("Error:", row["filename"], e)

    if (i + 1) % 100 == 0:
        print(f"Train: {i + 1}/{len(train_df)}")


train_speech = pd.DataFrame(all_features)

train_output = os.path.join(
    DATASET,
    "train_speech_features.csv"
)

train_speech.to_csv(
    train_output,
    index=False
)


# ====================================================
# TEST DATA
# ====================================================

test_csv = os.path.join(DATASET, "test.csv")
test_audio = os.path.join(DATASET, "test")

test_df = pd.read_csv(test_csv)

all_features = []

for i, row in test_df.iterrows():

    file_path = os.path.join(
        test_audio,
        row["filename"]
    )

    try:

        features = extract_speech_features(file_path)

        features["filename"] = row["filename"]

        all_features.append(features)

    except Exception as e:

        print("Error:", row["filename"], e)

    if (i + 1) % 50 == 0:
        print(f"Test: {i + 1}/{len(test_df)}")


test_speech = pd.DataFrame(all_features)

test_output = os.path.join(
    DATASET,
    "test_speech_features.csv"
)

test_speech.to_csv(
    test_output,
    index=False
)


print("\n================================")
print("Speech feature extraction done!")
print("================================")

print("Train shape:", train_speech.shape)
print("Test shape:", test_speech.shape)

print("\nTrain features:")
print(train_speech.columns.tolist())