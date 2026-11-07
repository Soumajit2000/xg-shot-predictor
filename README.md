# xG Shot Predictor

Will this shot be a goal? An end-to-end machine learning project that predicts the outcome of football shots, compares five classifiers on six metrics, and serves them through an interactive Streamlit app.

**Live app:** https://xg-shot-predictor.streamlit.app/

![App screenshot](assets/app.png)

---

## a. Problem statement

In football analytics, *expected goals (xG)* measures the quality of a chance: the probability that a shot becomes a goal, judged only from information available at the moment the ball is struck. This project frames xG as a **binary classification problem**: given a shot's location, how it was struck, the game situation and the positions of the defenders, predict whether it results in a goal (`is_goal = 1`) or not (`is_goal = 0`).

The problem is deliberately imbalanced. Only about 9% of shots are goals, so a model that always predicts "no goal" is roughly 91% accurate yet useless. This makes it a good test of why AUC, F1 and MCC matter alongside accuracy.

## b. Dataset description

**Source:** [StatsBomb Open Data](https://github.com/statsbomb/open-data), a public repository of manually tagged football event data, accessed with the `statsbombpy` library.

**Scope:** every match of four international tournaments: FIFA World Cup 2018, FIFA World Cup 2022, UEFA Euro 2020 and UEFA Euro 2024 (230 matches).

| Property | Value |
|---|---|
| Raw shots | 5,829 |
| Shots after cleaning | 5,600 |
| Features | 18 |
| Target | `is_goal` (1 = goal, 0 = no goal) |
| Goal rate | 9.1% |
| Problem type | Binary classification |

**Cleaning decisions**

- Removed penalty shootout kicks (period 5): they are not in-game shots.
- Removed in-game penalties and the 6 direct shots from corners: penalties come from a different process (fixed spot, no defenders), and 6 rows are too few to learn from.
- StatsBomb only records boolean flags when they are true, so empty values were filled as `False`.
- Columns that describe what happened *after* the strike were excluded to prevent data leakage: `shot_end_location`, `shot_statsbomb_xg` (StatsBomb's own model output), `shot_saved_to_post`, `shot_saved_off_target` and `shot_deflected`.

**Features (18)**

| Group | Features | Description |
|---|---|---|
| Geometry | `x`, `y`, `distance_to_goal`, `shot_angle` | Shot location (StatsBomb 120 × 80 pitch), distance to the centre of the goal, and the angle between the two posts as seen from the shot |
| Game state | `minute` | Match minute |
| Defensive pressure | `defenders_in_cone`, `opponents_within_5yd`, `keeper_distance_to_goal` | Built from StatsBomb freeze frames: defenders inside the triangle between shooter and posts, opponents within 5 yards of the shooter, and the goalkeeper's distance from goal |
| Situation flags | `under_pressure`, `first_time`, `one_on_one`, `aerial_won`, `open_goal`, `follows_dribble` | Boolean (0/1) descriptions of the shot |
| Categorical | `body_part`, `technique`, `play_pattern`, `shot_type` | How the shot was taken and the phase of play it came from |

**Train/test split:** 80/20, grouped by match (no match appears in both sets, so a shot and its rebound can never be split across them) and stratified by the target.

| Set | Shots | Matches | Goal rate |
|---|---|---|---|
| Train | 4,478 | 184 | 9.1% |
| Test (`test_data.csv`) | 1,122 | 46 | 9.0% |

## c. GitHub repository

https://github.com/Soumajit2000/xg-shot-predictor

```
xg-shot-predictor/
├── app.py                 # Streamlit app
├── requirements.txt       # pinned dependencies
├── README.md
├── test_data.csv          # held-out test set (18 features + is_goal)
├── model/                 # five saved pipelines, metrics and training metadata
├── src/
│   ├── data_loader.py     # downloads shots from StatsBomb open data (cached per match)
│   ├── features.py        # cleaning, feature engineering, feature contract
│   ├── split_data.py      # match-grouped, stratified train/test split
│   ├── evaluate.py        # shared metric functions (used by training and the app)
│   └── train.py           # trains, evaluates and saves all five models
├── notebooks/
│   └── 01_eda.ipynb       # exploratory analysis
└── .streamlit/config.toml # app theme
```

## d. Models used

All five models are scikit-learn pipelines that bundle preprocessing with the classifier: median imputation and standard scaling for numeric features, and one-hot encoding for categorical features, all fitted on the training data only. Each saved `.joblib` file therefore accepts raw feature columns directly, and the app never retrains.

| Model | Key settings |
|---|---|
| Logistic Regression | `class_weight="balanced"`, `max_iter=1000` |
| Decision Tree | `max_depth=5`, `min_samples_leaf=20`, `class_weight="balanced"` |
| KNN | `n_neighbors=25` |
| Naive Bayes | Gaussian |
| Support Vector Machine | RBF kernel, `class_weight="balanced"` |

### Comparison table (test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7861 | 0.7977 | 0.2544 | 0.7129 | 0.3750 | 0.3336 |
| Decision Tree | 0.6738 | 0.7109 | 0.1576 | 0.6040 | 0.2500 | 0.1714 |
| KNN | 0.9118 | 0.7457 | 0.6000 | 0.0594 | 0.1081 | 0.1690 |
| Naive Bayes | 0.6070 | 0.7954 | 0.1705 | 0.8713 | 0.2853 | 0.2596 |
| Support Vector Machine | 0.8155 | 0.7781 | 0.2675 | 0.6040 | 0.3708 | 0.3132 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression |Logistic Regression performed best overall, achieving the highest AUC (0.798), F1-score (0.375) and MCC (0.334). Its recall of 0.71 indicates that it identifies a good proportion of actual goals while maintaining a reasonable balance between precision and recall. |
| Decision Tree | Decision Tree showed relatively weak performance, with the lowest AUC (0.711) and lowest precision (0.158), along with an MCC of only 0.171. Its limited depth may restrict its ability to capture the gradual relationship between shot distance, angle and goal probability. |
| KNN | KNN achieved the highest accuracy (91.2%), but its recall of just 0.059 shows that it identified very few actual goals. Since the dataset is heavily imbalanced, its accuracy is barely better than the 91% baseline achieved by always predicting non-goals, making it largely ineffective. |
| Naive Bayes | Naive Bayes achieved the highest recall (0.871), identifying most actual goals, but its accuracy (60.7%) was the lowest and its precision was only 0.171. Correlated features such as distance, coordinates and shot angle may violate its feature independence assumption, causing it to overestimate goal probabilities and generate more false positives. |
| Support Vector Machine | SVM was a close second to Logistic Regression, with a similar F1-score (0.371) and MCC (0.313). It also achieved higher accuracy (81.6%) and the best precision among the balanced models, making it a competitive alternative. |
| **Overall winner for this dataset** | Logistic Regression is the overall winner for this dataset. It achieved the highest AUC, F1-score and MCC, making it the strongest model when considering both class discrimination and balanced classification performance rather than accuracy alone. Its ability to capture the relatively smooth relationship between shot distance, angle and goal probability also makes it a suitable choice for this problem. |

---

## Streamlit app

- Upload a test CSV (a sample `test_data.csv` can be downloaded from the sidebar)
- Choose any of the five models from a dropdown
- See all six metrics, a confusion matrix and a classification report
- Compare all five models side by side
- Explore a shot map of every shot, sized by expected goals

## Reproducing the results

```bash
pip install -r requirements.txt
python -m src.data_loader
python -m src.features
python -m src.split_data
python -m src.train
streamlit run app.py
```

All random seeds are fixed (`random_state=42`).

## Data attribution

Data provided by [StatsBomb](https://statsbomb.com/) through their free [open data repository](https://github.com/statsbomb/open-data). Use of the data is subject to the StatsBomb Public Data User Agreement.

## License

Code released under the MIT License. The data is not covered by this license.