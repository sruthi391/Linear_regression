import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Housing.csv")

# Preprocess categorical variables
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if x == 'yes' else 0)
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Define features and target
X = df.drop('price', axis=1)
y = df['price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output evaluation metrics
with open("metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.2f}\\n")
    f.write(f"MSE: {mse:.2f}\\n")
    f.write(f"R²: {r2:.4f}\\n")

# Output model coefficients
coefficients = pd.Series(model.coef_, index=X.columns)
intercept = model.intercept_

with open("coefficients.txt", "w") as f:
    f.write(f"Intercept: {intercept:.2f}\\n")
    for feature, coef in coefficients.items():
        f.write(f"{feature}: {coef:.2f}\\n")

# Plot regression (simple example using 'area' feature)
plt.figure(figsize=(8, 5))
plt.scatter(X_test['area'], y_test, color='blue', label='Actual Price')
plt.scatter(X_test['area'], y_pred, color='red', label='Predicted Price', alpha=0.6)
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Actual vs Predicted Price based on Area')
plt.legend()
plt.savefig("regression_plot.png")
plt.close()