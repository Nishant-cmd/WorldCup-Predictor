"""
data_utils.py
=============
Shared config, data loading, train/test split, and result printing.
Imported by random_forest.py and xgboost_model.py — nothing model-specific lives here.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit paths and constants here
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH    = "../../data/processed/training_data/final_training_dataset.csv"
TRAIN_RATIO  = 0.80     # chronological split — 80% past, 20% future
CV_FOLDS     = 5
RANDOM_STATE = 42
TARGET       = "result"
CLASS_NAMES  = ["Away Win", "Draw", "Home Win"]

# Only raw features — engineered extras were tested and made accuracy worse
FEATURES = [
    # Team strength
    "elo_diff", "rank_diff", "points_diff",
    # Recent form (last 5 games)
    "home_form_5", "away_form_5", "form_diff",
    # Goal averages (last 5 games)
    "home_avg_goals_scored_5",   "away_avg_goals_scored_5",
    "home_avg_goals_conceded_5", "away_avg_goals_conceded_5",
    "goals_scored_diff", "goals_conceded_diff",
    # Match context
    "tournament_importance", "home_advantage", "neutral",
    # Head-to-head record
    "home_h2h_wins", "away_h2h_wins", "h2h_diff",
    # ELO momentum (change from last game)
    "change_diff",
]


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Remove 2 corrupt rows where ELO was recorded as 0
    df = df[(df["home_elo"] > 0) & (df["away_elo"] > 0)].reset_index(drop=True)

    print(f"Rows    : {len(df):,}")
    print(f"Period  : {df['date'].min().date()}  →  {df['date'].max().date()}")
    print(f"Classes : {df[TARGET].value_counts().sort_index().to_dict()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame):
    """
    Splits on time — all training data comes before all test data.
    Standard k-fold would let the model 'see the future', inflating scores.
    """
    idx = int(len(df) * TRAIN_RATIO)
    X_train = df.iloc[:idx][FEATURES]
    y_train = df.iloc[:idx][TARGET]
    X_test  = df.iloc[idx:][FEATURES]
    y_test  = df.iloc[idx:][TARGET]

    print(f"\nTrain : {len(X_train):,} matches  (up to {df.iloc[idx - 1]['date'].date()})")
    print(f"Test  : {len(X_test):,}  matches  ({df.iloc[idx]['date'].date()} → {df.iloc[-1]['date'].date()})")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_results(y_true, y_pred):
    """Prints accuracy, macro F1 and a full classification report."""
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1 : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES)}")


def plot_confusion_matrix(y_true, y_pred, title: str, ax, cmap: str):
    """Plots a single confusion matrix onto the given axis."""
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, colorbar=False, cmap=cmap
    )
    acc = accuracy_score(y_true, y_pred)
    ax.set_title(f"{title}  (Acc: {acc:.3f})")