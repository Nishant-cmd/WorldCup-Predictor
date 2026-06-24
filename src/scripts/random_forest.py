"""
Football Match Result Prediction
=================================
Model : Random Forest  
Split  : Chronological 80/20 — no future data leakage
Target : result  →  0 = Away Win  |  1 = Draw  |  2 = Home Win
Run:
    python3 random_forest.py

Output:
    rf_model.pkl       — saved model
    rf_results.png     — confusion matrix + feature importance
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data_utils import (
    DATA_PATH, RANDOM_STATE, FEATURES, CLASS_NAMES,
    load_data, split_data, print_results, plot_confusion_matrix,
)

MODEL_OUT = "../../models/rf_model.pkl"
PLOT_OUT  = "../../output/simulation/rf_result.png"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train(X_train, y_train) -> RandomForestClassifier:
    """
    Params source: RandomizedSearchCV (n_iter=80, 5-fold CV) on this dataset.

      n_estimators=600   more trees → lower variance
      max_depth=8        shallow enough to avoid overfitting noisy match data
      min_samples_split  require at least 10 samples to attempt a split
      min_samples_leaf   each leaf must cover at least 4 samples
      max_features=sqrt  standard for classification (√n features per split)
      max_samples=0.8    each tree trains on 80% of rows (row bagging)

    To re-run the search, see the commented block at the bottom of this file.
    """
    model = RandomForestClassifier(
        n_estimators      = 600,
        max_depth         = 8,
        min_samples_split = 10,
        min_samples_leaf  = 4,
        max_features      = "sqrt",
        bootstrap         = True,
        max_samples       = 0.8,
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
    )
    model.fit(X_train, y_train)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot(model, X_test, y_test, save_path: str):

    y_pred = model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Random Forest — Football Match Prediction", fontsize=13, fontweight="bold")

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, "Random Forest", axes[0], cmap="Blues")

    # Feature importance (mean decrease in impurity)
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fi.tail(15).plot(kind="barh", ax=axes[1], color="#2196F3", alpha=0.85)
    axes[1].set_title("Top 15 Features — Impurity Importance")
    axes[1].set_xlabel("Mean Decrease in Impurity")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 50)
    print("  RANDOM FOREST — FOOTBALL PREDICTION")
    print("=" * 50)

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)

    print("\nTraining model...")
    model  = train(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n--- Results ---")
    print_results(y_test, y_pred)

    plot(model, X_test, y_test, PLOT_OUT)

    joblib.dump(model, MODEL_OUT)
    print(f"Model saved → {MODEL_OUT}")
    print("\n✓ Done.")
    print("  NOTE: ~60-61% is the realistic ceiling for this dataset.")
    print("  Breaking through requires player-level data: lineups,")
    print("  injuries, xG — none of which are in the current dataset.")


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# APPENDIX — Re-run hyperparameter search (slow, ~10 min)
# ══════════════════════════════════════════════════════════════════════════════
#
# from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
#
# param_grid = {
#     "n_estimators":      [200, 300, 400, 500, 600],
#     "max_depth":         [6, 8, 10, 12, None],
#     "min_samples_split": [2, 4, 6, 8, 10],
#     "min_samples_leaf":  [1, 2, 3, 4],
#     "max_features":      ["sqrt", "log2"],
#     "max_samples":       [0.7, 0.8, 0.9, None],
# }
# search = RandomizedSearchCV(
#     RandomForestClassifier(bootstrap=True, random_state=RANDOM_STATE, n_jobs=-1),
#     param_distributions=param_grid,
#     n_iter=80, scoring="accuracy",
#     cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
#     random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
# )
# search.fit(X_train, y_train)
# print(f"Best CV : {search.best_score_:.4f}")
# print(f"Params  : {search.best_params_}")





