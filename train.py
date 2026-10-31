"""Trains the credit model on the first three months of data and saves it."""

import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

FEATURES = ["income", "dti", "utilisation", "delinquencies", "inquiries", "loan_amount", "years_employed"]
TRAIN_MONTHS = [1, 2, 3]


def train(df):
    train_df = df[df["month"].isin(TRAIN_MONTHS)]
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)
    model.fit(train_df[FEATURES], train_df["repaid"])
    return model


if __name__ == "__main__":
    df = pd.read_csv("data/applications.csv")
    model = train(df)
    with open("outputs/model.pkl", "wb") as f:
        pickle.dump(model, f)
    train_df = df[df["month"].isin(TRAIN_MONTHS)]
    acc = model.score(train_df[FEATURES], train_df["repaid"])
    print(f"trained on months {TRAIN_MONTHS}, train accuracy {acc:.3f}")
