\# SHL Hiring Assessment 2026 – Audio Score Prediction



\## Overview



This project was developed for the \*\*SHL Hiring Assessment 2026\*\* Kaggle challenge.



The objective is to predict a numerical score from audio recordings. The training dataset contains labeled audio files, while the test dataset contains audio files for which predictions must be generated.



The final solution uses audio signal processing and machine learning to predict the scores.



\## Dataset



The dataset contains:



\- 769 training audio files

\- 216 test audio files

\- Training labels ranging from 0 to 5

\- Audio recordings containing human speech



The original audio dataset is not included in this repository because of its large size.



\## Approach



The project was developed in multiple stages.



\### 1. Audio Feature Extraction



Audio files were loaded using `librosa` at a sampling rate of \*\*16 kHz\*\*.



The initial feature set included:



\- MFCC mean

\- MFCC standard deviation

\- Zero Crossing Rate

\- RMS Energy

\- Spectral Centroid

\- Spectral Bandwidth



Additional speech-specific features were then added:



\- Pitch / Fundamental Frequency

\- Spectral Rolloff

\- Spectral Flatness

\- Energy statistics

\- Zero Crossing Rate statistics



The final model used \*\*68 audio features\*\*.



\### 2. Machine Learning Model



Several machine learning models were tested during development:



\- Random Forest Regressor

\- Extra Trees Regressor

\- HistGradientBoosting Regressor

\- Ensemble models



The final selected model is:



\*\*Extra Trees Regressor\*\*



Configuration:



```text

n\_estimators = 500

max\_features = 0.7

random\_state = 42

n\_jobs = -1
```

\## Model Evaluation



The model was evaluated using:



\- Mean Absolute Error (MAE)

\- Root Mean Squared Error (RMSE)

\- R² Score



The final feature set achieved the following local validation results:



| Metric | Score |

|--------|------:|

| MAE | 0.5435 |

| RMSE | 0.6871 |

| R² | 0.7426 |



\## Kaggle Performance



Multiple approaches were tested during the competition.



| Submission | Approach | Kaggle Score |

|------------|----------|-------------:|

| Submission 1 | Original 48 audio features + Extra Trees | 0.7396 |

| Submission 2 | Expanded audio features | 0.7045 |

| Submission 3 | Tuned Extra Trees | 0.7371 |

| Submission 4 | Ensemble model | 0.7371 |

| Submission 5 | Original + speech-specific features | \*\*0.7489\*\* |



\### Best Result



The best Kaggle score achieved was:



\*\*0.7489\*\*



The selected final submission was:



```text

submission\_v5.csv

```

\## Final Model



The final model combines:



\- 48 original audio features

\- 20 additional speech-specific features

\- 68 features in total

\- Extra Trees Regression



The final model was trained on the complete training dataset before generating predictions for the test dataset.



\## Project Files



\- `README.md` – Project documentation

\- `extract\_features.py` – Extracts the original audio features

\- `extract\_test\_features.py` – Extracts features from test audio

\- `speech\_features\_v4.py` – Extracts speech-specific audio features

\- `model7\_test.py` – Evaluates the final feature set

\- `make\_submission\_v5.py` – Trains the final model and creates predictions

\- `submission\_v5.csv` – Final prediction file



\## Requirements



The project uses Python and the following libraries:



```text

pandas

numpy

scikit-learn

librosa

soundfile

```
Install them using:

```bash

pip install pandas numpy scikit-learn librosa soundfile

```

\## How to Run



\### 1. Extract original features



```bash

python extract\_features.py

python extract\_test\_features.py

```

\### 2. Extract speech-specific features



```bash

python speech\_features\_v4.py

```

\### 3. Evaluate the model

```

python model7\_test.py

```

\### 4. Generate predictions



```bash

python make\_submission\_v5.py

```

This creates:



```text

submission\_v5.csv
```

\## Results

The final selected solution achieved a:

Kaggle Score: 0.7489

The solution improved the initial Kaggle score of 0.7396 by combining the original audio features with additional speech-specific features.

\## Conclusion

This project demonstrates an audio-based machine learning pipeline involving:

1\. Audio preprocessing

2\. Feature extraction

3\. Speech-specific feature engineering

4\. Machine learning model comparison

5\. Model evaluation

6\. Kaggle prediction generation

The final Extra Trees model achieved the best Kaggle score of 0.7489.

\## Author

\*\*SINGAM VENKATA THARUN REDDY\*\*

B.Tech – Computer Science and Engineering

