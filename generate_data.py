"""Builds a synthetic credit application dataset with twelve monthly batches.

From month 7 onward the income distribution for female applicants drifts
downward, which pushes the model toward denying that group more often.
The drift is deliberate so the monitor has something to catch.
"""

import numpy as np
import pandas as pd

SEED = 7
N_PER_MONTH = 3000


def make_month(rng, month):
    n = N_PER_MONTH
    gender = rng.choice(["female", "male"], size=n, p=[0.48, 0.52])
    age_group = rng.choice(["18_34", "35_54", "55_plus"], size=n, p=[0.35, 0.45, 0.20])

    income = rng.normal(62000, 18000, size=n)
    if month >= 7:
        # drift: female incomes in the data slide down a little more each month
        shift = 2500 * (month - 6)
        income = np.where(gender == "female", income - shift, income)
    income = np.clip(income, 12000, 250000)

    dti = np.clip(rng.normal(0.32, 0.12, size=n), 0.02, 0.95)
    utilisation = np.clip(rng.beta(2, 4, size=n), 0, 1)
    delinquencies = rng.poisson(0.4, size=n)
    inquiries = rng.poisson(1.5, size=n)
    loan_amount = np.clip(rng.normal(18000, 9000, size=n), 1000, 80000)
    years_employed = np.clip(rng.normal(6, 4, size=n), 0, 40)

    logit = (
        0.6
        + 0.000035 * (income - 60000)
        - 4.0 * (dti - 0.32)
        - 2.5 * (utilisation - 0.33)
        - 0.9 * delinquencies
        - 0.25 * inquiries
        - 0.00002 * (loan_amount - 18000)
        + 0.06 * years_employed
    )
    p_good = 1 / (1 + np.exp(-logit))
    repaid = rng.binomial(1, p_good)

    return pd.DataFrame({
        "month": month,
        "gender": gender,
        "age_group": age_group,
        "income": income.round(0),
        "dti": dti.round(3),
        "utilisation": utilisation.round(3),
        "delinquencies": delinquencies,
        "inquiries": inquiries,
        "loan_amount": loan_amount.round(0),
        "years_employed": years_employed.round(1),
        "repaid": repaid,
    })


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    frames = [make_month(rng, m) for m in range(1, 13)]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/applications.csv", index=False)
    print(f"wrote {len(df)} rows across {df['month'].nunique()} months")
