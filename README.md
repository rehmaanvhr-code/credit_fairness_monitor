# Credit fairness monitor

This repo turns a one time fairness audit into an ongoing monitoring pipeline for a credit decision model. It generates twelve monthly batches of synthetic credit applications, trains a gradient boosting model on the first three months, and then scores every month in turn. For each month it records approval rates by group, the disparate impact ratio under the four fifths rule, the equalized odds difference, and the AUC gap between groups, for two sensitive attributes (gender and age group). Thresholds live in config.json. When a metric crosses a threshold the month and metric are written to an alerts file. The synthetic data contains a deliberate drift from month 7 onward, where the income distribution of female applicants slides downward, so the monitor has a real problem to catch. The drift becomes visible in the metrics around month 10.

It extends the earlier credit-fairness-audit repo on this account. That project audited a model once at a single point in time. The argument in "Who Bears the Cost of Waiting: The Uneven Price of Slow AI Adoption" (Rehman, 2026) is that the costs of delay and of poor governance land unevenly across groups, and that this unevenness compounds over time rather than showing up at a single moment. A one time audit cannot see that. Monitoring can.

In financial services this is now a standard control rather than a research exercise. Fair lending obligations under ECOA and Regulation B, the EU AI Act's treatment of creditworthiness assessment as high risk, and model risk guidance such as OSFI Guideline E23 all expect that fairness is checked after deployment and not only before it. A monitor like this one is the piece of a governance framework that makes the post deployment check concrete: a metric, a threshold, a history and an alert.

## How to run

```
pip install -r requirements.txt
python generate_data.py
python train.py
python monitor.py
pytest
```

generate_data.py writes data/applications.csv (36,000 rows, seed fixed). train.py fits the model on months 1 to 3 and saves outputs/model.pkl. monitor.py scores every month and writes three files to outputs/: metrics_history.csv with one row per month and attribute, alerts.md listing every breach, and fairness_over_time.png which plots the three fairness metrics by month with the thresholds drawn as dotted lines.

## Reading the outputs

Disparate impact is the lowest group approval rate divided by the highest. Values under 0.8 fail the four fifths rule used in US employment and lending practice. Equalized odds difference is the larger of the true positive rate gap and the false positive rate gap between groups; it asks whether the model makes errors at the same rate for everyone. The AUC gap asks whether the model ranks applicants equally well within each group. In the shipped run, gender stays within every threshold through month 9 and then breaches disparate impact and equalized odds from month 10, which is what the injected drift should produce. Age group stays within limits throughout, which is the expected result because nothing in the data generator ties age to outcome.

## Data and sources

All data is synthetic and generated with a fixed seed, so runs are reproducible and the repo contains no personal data. The feature set (income, debt to income, utilisation, delinquencies, inquiries, loan amount, years employed) mirrors the kind of variables found in the German Credit dataset (UCI) and the Home Credit Default Risk dataset (Kaggle). The pipeline can be pointed at either by replacing data/applications.csv with a file that has the same column names plus a month column and a binary outcome column.

## Limitations

The drift is a simple income shift and real drift is messier. The thresholds are illustrative, not regulatory. With smaller monthly batches the age group metrics become noisy because the 55 plus group is small, which is itself a lesson: a monitor needs minimum group sizes before an alert is trustworthy. The monitor measures outcomes, not the cause of them, so an alert is a reason to investigate rather than a conclusion.

Author: Saif Rehman, MMAI, Smith School of Business, Queen's University.
