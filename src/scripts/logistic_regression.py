import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv(
    "../../data/processed/training_data/final_training_dataset.csv"
)

features = [
    "home_elo",
    "away_elo",
    "elo_diff",

    "home_rank",
    "away_rank",
    "rank_diff",

    "home_points",
    "away_points",
    "points_diff",

    "home_form_5",
    "away_form_5",
    "form_diff",

    "home_avg_goals_scored_5",
    "away_avg_goals_scored_5",

    "home_avg_goals_conceded_5",
    "away_avg_goals_conceded_5",

    "goals_scored_diff",
    "goals_conceded_diff",

    "tournament_importance",
    "home_advantage",

    "home_h2h_wins",
    "away_h2h_wins",
    "h2h_diff"
]

target = "result"

print(df.shape)
df.head()
X = df[features]

y = df[target]

#Holdout Method of Splitting Datasets(Train-Test split)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)


#Scale Features for logistic Regression ( Standarize the value so that bigger value doesnt affect optimization process) 
#fit_transform calculates z=x-mean/std
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

#Train Logistic Regression
log_model = LogisticRegression(
    max_iter=5000,
    random_state=42
)

log_model.fit(
    X_train_scaled,
    y_train
)

y_pred = log_model.predict(
    X_test_scaled
)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        y_pred
    )
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Logistic Regression Accuracy: {accuracy:.4f}")



