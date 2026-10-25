# Late Deliveries

This project predicts whether a QuickBite food delivery order will arrive more than 10 minutes late. The objective is to identify deliveries that should receive a voucher while minimizing the total cost of unnecessary vouchers and missed late orders.

## Setup

Create the Conda environment using the `conda-forge` channel specified in `environment.yml`:

```bash
git clone https://github.com/Arte8/QuickBite
cd Quickbite
conda env create -f environment.yml
conda activate lab05-late-deliveries
python -m pip install -r requirements.txt
python -m pip install -e .

```
## Run

```bash
python -m latedeliveries.train
```

The project uses Python 3.11. The requirements include scikit-learn, pandas, matplotlib, pytest, Ruff, and ipykernel.

## Current Features

| Column | Known when order is placed? | Decision |
|---|---|---|
| `distance_km` | Yes, if calculated from the delivery address | Keep |
| `items` | Yes, from the order details | Keep |
| `hour` | Yes, from the order timestamp | Keep |
| `weather` | Yes, if current weather is available | Keep |
| `day_type` | Yes, from the order date | Keep |
| `vehicle` | Only if the vehicle is already assigned | Exclude |
| `courier_months` | Only if the courier is already assigned | Exclude |
| `prep_minutes` | No, if it represents actual preparation duration | Exclude |
| `customer_rating` | No, recorded after delivery | Exclude |
| `order_id` | Identifier, not a useful predictor | Exclude |
| `late` | Target variable being predicted | Exclude |



This command generates the synthetic dataset, compares classification models, performs hyperparameter tuning, selects a cost-based threshold, evaluates the final model, and saves the outputs.

## Check it

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

All 12 automated tests passed in the latest verification.

## Decisions


### Feature selection

The dataset contains 3,000 synthetic delivery orders generated
using `make_orders(n=3000, seed=7)`.

The model predicts whether an order will be late at the moment
the order is placed. Only information available at that time
is used as a predictor.

Columns included:

| Column | Reason for inclusion |
|---|---|
| `distance_km` | Delivery distance can be calculated from the delivery address. |
| `items` | Number of items is known from the order details. |
| `hour` | Order placement time is known. |
| `weather` | Current weather can be obtained at order placement. |
| `day_type` | Weekday or weekend is known from the order date. |

Columns excluded:

| Column | Reason for exclusion |
|---|---|
| `prep_minutes` | Actual preparation duration is not known when the order is placed. |
| `courier_months` | Courier experience is unavailable unless a courier has already been assigned. |
| `vehicle` | Delivery vehicle may not be assigned at order placement. |
| `customer_rating` | Recorded after delivery and creates target leakage. |
| `order_id` | Identifier rather than a meaningful predictor. |
| `late` | Target variable, not a predictor. |

Numerical features use median imputation and scaling.
Categorical features use one-hot encoding with
`handle_unknown="ignore"`.

The dataset is split into 75% training data and 25%
testing data using stratification and `random_state=42`.
The test set is reserved for final evaluation.

### Model selection


### Model comparison


| Model | Accuracy (mean ± SD) | Precision (mean ± SD) | Recall (mean ± SD) | F1 (mean ± SD) | ROC-AUC (mean ± SD) |
|---|---|---|---|---|---|
| DummyClassifier | 0.844 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| LogisticRegression | 0.855 ± 0.014 | 0.607 ± 0.162 | 0.199 ± 0.058 | 0.298 ± 0.083 | 0.769 ± 0.028 |
| BalancedLogisticRegression | 0.730 ± 0.024 | 0.322 ± 0.031 | 0.655 ± 0.054 | 0.432 ± 0.038 | 0.768 ± 0.028 |
| RandomForest | 0.834 ± 0.013 | 0.447 ± 0.069 | 0.245 ± 0.058 | 0.313 ± 0.061 | 0.699 ± 0.023 |


Logistic Regression was selected because it achieved the highest mean ROC-AUC among the evaluated models, although its advantage over Balanced Logistic Regression was negligible. It also provided higher accuracy and precision at the default threshold.

Balanced Logistic Regression achieved better recall and F1, so the model comparison shows that accuracy alone is insufficient for this problem. A separate cost-based threshold was therefore selected for the chosen model.


### Hyperparameter tuning

GridSearchCV evaluated eight combinations of regularization
strength (`C`) and solver using five-fold cross-validation.

ROC-AUC was selected as the scoring metric because it measures
how well the model ranks late orders across classification
thresholds. The final decision threshold was selected
separately using the business cost function.

The best hyperparameters were:

- `C = 0.01`
- `solver = liblinear`

The untuned cross-validation ROC-AUC was 0.768543,
while the tuned ROC-AUC was 0.769897.

The improvement was approximately 0.001354, smaller than
the fold-to-fold standard deviation of approximately 0.028196.

Therefore, tuning did not produce a clearly meaningful
performance improvement relative to cross-validation variation.


### Threshold selection and cost rule

The business cost function assigns:

- **$4** for every flagged order, representing a voucher.
- **$15** for every actual late order that is not flagged.

The total cost is:

`Total cost = 4 × flagged orders + 15 × missed late orders`

The default classification threshold of 0.50 does not necessarily minimize this cost.

Out-of-fold probabilities from the training data were evaluated across thresholds from 0.05 to 0.95 in increments of 0.05.

The selected threshold was 0.30, producing the lowest measured out-of-fold training cost of $4,316. This threshold was selected using training data only; the held-out test set was reserved for final evaluation.

The threshold was selected using training data only. The held-out test set was reserved for final evaluation.


## Result

### Final test-set performance

The final evaluation used 750 held-out orders.

| Metric | Result |
|---|---|
| Accuracy | 0.844 |
| Precision | 0.500 |
| Recall | 0.410 |
| F1-score | 0.451 |
| ROC-AUC | 0.765 |
| Classification threshold | 0.30 |

### Confusion matrix

| Actual / Predicted | On time | Late |
|---|---|---|
| On time | 585 | 48 |
| Late | 69 | 48 |

The model correctly identified 48 late orders and missed
69 late orders. It also flagged 48 on-time orders.

### Business cost comparison

| Policy | Total test cost |
|---|---|
| Flag nobody | $1,755 |
| Flag everybody | $3,000 |
| Trained model (threshold 0.30) | $1,419 |

The trained model saved $336, or approximately 19.1%,
compared with flagging nobody.

This result supports further pilot evaluation, but the
cost advantage should not be treated as guaranteed because
it was measured on one synthetic test split and may vary
on new orders.

## What the tests cover

The project includes 12 automated tests covering:

- Correct training and testing split sizes.
- Exclusion of target leakage and identifier columns.
- Valid prediction probabilities.
- Handling missing numerical values.
- Handling previously unseen categorical values.
- Cost calculations using small, manually verifiable examples.
- Threshold selection based on minimum intervention cost.
- Cases where flagging nobody or flagging everybody is the cheapest policy.
- Saving and reloading a trained model without changing its predictions.
-Cost calculation tests use pytest parametrization to verify multiple manually calculated voucher and missed-delivery scenarios.

One aspect not directly tested is the model's performance on real-world delivery data, because the assignment uses synthetic data and no real operational dataset is available.


## Known limitations

The model was trained on synthetic orders, so its performance
may not generalize to real deliveries.

The final model's recall is approximately 41.0%, meaning
it misses more than half of the actual late orders.

Only features available at order placement were included.
Potentially predictive information such as actual preparation
duration, courier experience, and vehicle type was excluded
because its availability at prediction time was not established.

Hyperparameter tuning produced only a marginal ROC-AUC
improvement. The threshold-selection process may also be
somewhat optimistic because it uses out-of-fold probabilities
from a model configuration selected on the same training data.

The reported cost savings come from a single held-out test
split. Additional validation across different samples would
provide greater confidence.

With more time, I would evaluate additional models,
improve recall, assess probability calibration, validate
the system on real deliveries, and monitor model performance
and intervention costs after deployment.
