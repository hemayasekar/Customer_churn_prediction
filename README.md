# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will churn (leave) based on their account and service information.

## Dataset

- **Source:** WA*Fn-UseC*-Telco-Customer-Churn.csv
- **Records:** 7,043 customers
- **Features:** 19 features including tenure, contract type, monthly charges, internet service, etc.
- **Target:** Churn (Yes/No)

## Preprocessing Steps

1. Converted `TotalCharges` from string to numeric (filled missing values with median)
2. Dropped `customerID` (not useful for prediction)
3. Label encoded all categorical columns
4. Applied SMOTE to handle class imbalance

## Models Used

| Model               | Description                          |
| ------------------- | ------------------------------------ |
| Logistic Regression | Linear classifier with max_iter=1000 |
| Random Forest       | Ensemble of 100 decision trees       |

## Evaluation

- Classification Report (precision, recall, F1-score)
- Confusion Matrix (visualized as heatmap)

## Requirements

- pandas
- scikit-learn
- imbalanced-learn
- seaborn
- matplotlib

## Usage

Run the notebook `Customer_churn_pre.ipynb` sequentially from top to bottom.
