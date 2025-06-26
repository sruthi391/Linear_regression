# Linear_regression
# 🏠 Housing Price Prediction with Linear Regression

This project demonstrates the implementation of **Simple** and **Multiple Linear Regression** using the [Housing dataset](Housing.csv). It covers preprocessing, model training, evaluation, and visualization.

---

## 📁 Files Included

| File | Description |
|------|-------------|
| `Housing.csv` | Raw dataset |
| `Housing_preprocessed.csv` | Cleaned & preprocessed dataset |
| `regression_model.py` | Python script to train and evaluate the model |
| `metrics.txt` | Model evaluation metrics (MAE, MSE, R²) |
| `coefficients.txt` | Linear regression coefficients and intercept |
| `regression_plot.png` | Plot of predicted vs actual prices based on area |

---

## 🧪 Tools Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## 📌 How to Run

```bash
# 1. Install dependencies
pip install pandas scikit-learn matplotlib

# 2. Run the model script
python regression_model.py

---

## 📖 Interview Questions & Answers
1. What assumptions does linear regression make?
Linearity

Independence of errors

Homoscedasticity (constant variance of errors)

Normality of residuals

No multicollinearity

2. How do you interpret the coefficients?
Each coefficient represents the expected change in the target variable for a one-unit increase in the corresponding feature, holding all other features constant.

3. What is R² score and its significance?
R² (coefficient of determination) tells how well the model explains the variance in the target. Closer to 1 means better fit.

4. When would you prefer MSE over MAE?
MSE is preferred when larger errors should be penalized more heavily than smaller ones.

5. How do you detect multicollinearity?
Using Variance Inflation Factor (VIF) or examining correlation matrices.

6. Difference between simple and multiple regression?
Simple regression uses one feature.

Multiple regression uses two or more features.

7. Can linear regression be used for classification?
No, linear regression is for regression tasks. For classification, use logistic regression or other classifiers.

8. What happens if you violate regression assumptions?
Violating assumptions can lead to biased or inefficient estimates, incorrect inferences, and poor model performance.
