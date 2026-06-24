"""
xgboost_model.py
================
Football match prediction using XGBoost.

Run:
    python3 xgboost_model.py

Output:
    xgb_model.pkl      — saved model
    xgb_results.png    — confusion matrix + feature importance
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from data_utils import (
    DATA_PATH, CV_FOLDS, RANDOM_STATE, FEATURES,
    load_data, split_data, print_results, plot_confusion_matrix,
)


MODEL_OUT = "../../models/xgb_model.pkl"
PLOT_OUT  = "../../output/simulation/rf_result.png"

# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train(X_train, y_train) -> XGBClassifier:
    """
    Param rationale — each value chosen for noisy, aggregate-stats football data:

      n_estimators=500    enough trees at lr=0.05 to fully converge
      max_depth=4         shallow trees — prevents memorising match noise
      learning_rate=0.05  slow learning + more trees generalises better than fast + few
      subsample=0.8       row bagging per tree — reduces variance
      colsample_bytree    random feature subset per tree (same idea as RF's max_features)
      min_child_weight=3  ignore splits covering fewer than 3 samples — avoids outliers
      gamma=0.1           minimum loss reduction to make a split — built-in pruning
      reg_alpha=0.1       L1 regularisation — pushes weak feature weights toward zero
      reg_lambda=1.5      L2 regularisation — prevents any single feature dominating
      tree_method='hist'  fast histogram algorithm — same accuracy, much faster training
    """
    model = XGBClassifier(
        n_estimators      = 500,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.7,
        min_child_weight  = 3,
        gamma             = 0.1,
        reg_alpha         = 0.1,
        reg_lambda        = 1.5,
        eval_metric       = "mlogloss",
        tree_method       = "hist",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
    )
    model.fit(X_train, y_train)
    return model


def cross_validate(model, X_train, y_train):
    """Runs stratified k-fold CV on the training set and prints the result."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"5-Fold CV  →  Mean: {scores.mean():.4f}  |  Std: ±{scores.std():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot(model, X_test, y_test, save_path: str):
    y_pred = model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("XGBoost — Football Match Prediction", fontsize=13, fontweight="bold")

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, "XGBoost", axes[0], cmap="Oranges")

    # Feature importance (gain — how much each feature reduces loss across all splits)
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fi.tail(15).plot(kind="barh", ax=axes[1], color="#F5871F", alpha=0.85)
    axes[1].set_title("Top 15 Features — Gain Importance")
    axes[1].set_xlabel("Gain")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 50)
    print("  XGBOOST — FOOTBALL PREDICTION")
    print("=" * 50)

    df                           = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)

    print("\nTraining model...")
    model  = train(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n--- Results ---")
    print_results(y_test, y_pred)

    print("--- Cross-Validation ---")
    cross_validate(model, X_train, y_train)

    plot(model, X_test, y_test, PLOT_OUT)

    joblib.dump(model, MODEL_OUT)
    print(f"Model saved → {MODEL_OUT}")
    print("\n✓ Done.")


if __name__ == "__main__":
    main()