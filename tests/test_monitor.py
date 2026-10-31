import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_data import generate
from monitor import disparate_impact, equalized_odds_diff, check_thresholds, run
from train import train, FEATURES


def test_generate_shape():
    df = generate()
    assert df["month"].nunique() == 12
    assert set(FEATURES).issubset(df.columns)
    assert df["repaid"].between(0, 1).all()


def test_generate_is_deterministic():
    a = generate(seed=3)
    b = generate(seed=3)
    assert a.equals(b)


def test_disparate_impact_equal_groups():
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "approved": [1, 0, 1, 0]})
    assert disparate_impact(df, "g") == 1.0


def test_equalized_odds_perfect_classifier():
    df = pd.DataFrame({
        "g": ["a", "a", "b", "b"],
        "repaid": [1, 0, 1, 0],
        "approved": [1, 0, 1, 0],
    })
    assert equalized_odds_diff(df, "g") == 0.0


def test_pipeline_flags_drift():
    df = generate()
    model = train(df)
    config = {
        "approval_threshold": 0.5,
        "sensitive_attributes": ["gender"],
        "thresholds": {"disparate_impact_min": 0.8, "equalized_odds_diff_max": 0.1, "auc_gap_max": 0.05},
    }
    history, alerts = run(df, model, config)
    assert len(history) == 12
    early = history[history["month"] <= 6]
    assert (early["disparate_impact"] >= 0.8).all()
    late_alerts = [a for a in alerts if a[0] >= 10 and a[2] == "disparate_impact"]
    assert len(late_alerts) >= 1
