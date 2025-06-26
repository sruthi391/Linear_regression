import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load and Preprocess the Dataset ---

# Load the dataset
df = pd.read_csv('Housing.csv')

# Convert binary categorical columns to numerical (0s and 1s)
binary_map = {'yes': 1, 'no': 0}
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map(binary_map)

# One-hot encode the 'furnishingstatus' column
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

print("--- Preprocessed Data (Head) ---")
print(df.head())
print("\n--- Preprocessed Data Info ---")
print(df.info())

# Save the preprocessed DataFrame to a new CSV file
df.to_csv('Housing_preprocessed.csv', index=False)
print("\nPreprocessed data saved to Housing_preprocessed.csv")


# --- 2. Simple Linear Regression (Area vs. Price) ---

print("\n--- Simple Linear Regression ---")
X_simple = df[['area']]
y_simple = df['price']

# Split data into train and test sets
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

# Fit the Linear Regression model
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train_simple)

# Make predictions on the test set
y_pred_simple = model_simple.predict(X_test_simple)

# Evaluate the model
mae_simple = mean_absolute_error(y_test_simple, y_pred_simple)
mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
r2_simple = r2_score(y_test_simple, y_pred_simple)

print(f'Simple Linear Regression (Area vs. Price):')
print(f'  MAE: {mae_simple:.2f}')
print(f'  MSE: {mse_simple:.2f}')
print(f'  R-squared: {r2_simple:.2f}')
print(f'  Coefficient (Area): {model_simple.coef_[0]:.2f}')
print(f'  Intercept: {model_simple.intercept_:.2f}')

# Plotting the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_test_simple, y_test_simple, color='blue', label='Actual Prices')
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression: Price vs. Area')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig('simple_linear_regression_plot.png')
plt.show()
print("Simple linear regression plot saved as simple_linear_regression_plot.png")


# --- 3. Multiple Linear Regression (All Features vs. Price) ---

print("\n--- Multiple Linear Regression ---")
X_multiple = df.drop('price', axis=1)
y_multiple = df['price']

# Split data into train and test sets
X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(X_multiple, y_multiple, test_size=0.2, random_state=42)

# Fit the Linear Regression model
model_multiple = LinearRegression()
model_multiple.fit(X_train_multiple, y_train_multiple)

# Make predictions on the test set
y_pred_multiple = model_multiple.predict(X_test_multiple)

# Evaluate the model
mae_multiple = mean_absolute_error(y_test_multiple, y_pred_multiple)
mse_multiple = mean_squared_error(y_test_multiple, y_pred_multiple)
r2_multiple = r2_score(y_test_multiple, y_pred_multiple)

print(f'\nMultiple Linear Regression (All Features vs. Price):')
print(f'  MAE: {mae_multiple:.2f}')
print(f'  MSE: {mse_multiple:.2f}')
print(f'  R-squared: {r2_multiple:.2f}')

# Interpret coefficients
coefficients = pd.DataFrame({'Feature': X_multiple.columns, 'Coefficient': model_multiple.coef_})
print('\nCoefficients for Multiple Linear Regression:')
print(coefficients)