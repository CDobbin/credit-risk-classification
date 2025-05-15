# credit-risk-classification

## Overview of the Analysis

The purpose of this analysis was to develop a machine learning model to predict the creditworthiness of borrowers using historical lending data from a peer-to-peer lending company. The goal was to classify loans as either healthy (0) or high-risk (1) to aid in informed lending decisions. The dataset, `lending_data.csv`, contained financial information such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, total debt, and loan status.

- **Prediction Target**: The variable to predict was `loan_status`, where 0 indicates a healthy loan and 1 indicates a high-risk loan prone to defaulting.
- **Variable Distribution**:
  ```python
  # Value counts for loan_status
  df['loan_status'].value_counts()
  ```
  - Healthy Loans (0): 18,765
  - High-Risk Loans (1): 619
  This shows a significant class imbalance, with healthy loans dominating the dataset.
- **Machine Learning Process**:
  1. **Data Preparation**: Loaded the dataset into a Pandas DataFrame, separated features (`X`) from the target (`y`), and split the data into training (75%) and testing (25%) sets using `train_test_split` with `random_state=1`.
  2. **Model Training**: Trained a logistic regression model on the training data using `LogisticRegression` with `random_state=1`.
  3. **Prediction**: Generated predictions on the testing data.
  4. **Evaluation**: Assessed model performance using a confusion matrix and classification report, focusing on accuracy, precision, and recall scores.
- **Methods Used**: The analysis utilized a single machine learning algorithm, `LogisticRegression`, due to its suitability for binary classification tasks and interpretability in financial contexts.

## Results

* **Logistic Regression Model**:
  - **Accuracy Score**: 0.99 – The model correctly predicted 99% of all loan statuses, indicating strong overall performance.
  - **Precision Score**:
    - Healthy Loans (0): 1.00 – 100% of loans predicted as healthy were actually healthy, showing no false positives.
    - High-Risk Loans (1): 0.85 – 85% of loans predicted as high-risk were actually high-risk, with 15% false positives.
  - **Recall Score**:
    - Healthy Loans (0): 0.99 – The model identified 99% of all healthy loans, missing only 1%.
    - High-Risk Loans (1): 0.95 – The model identified 95% of all high-risk loans, missing only 5%.

## Summary

The logistic regression model performed exceptionally well, achieving an overall accuracy of 99%. It excelled at predicting healthy loans, with near-perfect precision (1.00) and recall (0.99), ensuring reliable identification of low-risk borrowers. For high-risk loans, the model’s recall of 0.95 was particularly strong, capturing 95% of risky loans, which is critical for minimizing lending risk. The precision of 0.85 for high-risk loans indicates some false positives, likely due to the class imbalance (18,765 healthy vs. 619 high-risk loans).

The model’s high recall for high-risk loans makes it well-suited for the problem, as predicting the `1`’s (high-risk loans) is more important than predicting the `0`’s in a lending context. Missing a high-risk loan (false negative) could lead to significant financial loss, whereas false positives (predicting a healthy loan as high-risk) are less costly. The logistic regression model’s performance, especially its ability to identify nearly all high-risk loans, supports its use.

**Recommendation**: I recommend using the logistic regression model for assessing borrower creditworthiness due to its high accuracy and excellent recall for high-risk loans. To further enhance performance, techniques like oversampling the high-risk class or exploring ensemble methods (e.g., Random Forest) could improve precision for high-risk loans, addressing the class imbalance.