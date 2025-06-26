# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Housing.csv")

# Binary encoding for 'yes'/'no' columns
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_cols] = df[binary_cols].applymap(lambda x: 1 if x == 'yes' else 0)

# One-hot encoding for 'furnishingstatus'
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# ==========================================================
# 🔹 SIMPLE LINEAR REGRESSION (Using 'area' only)
# ==========================================================

X_simple = df[['area']]
y = df['price']

model_simple = LinearRegression()
model_simple.fit(X_simple, y)
y_pred_simple = model_simple.predict(X_simple)

# Plot regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_simple, y, color='blue', label='Actual Prices')
plt.plot(X_simple, y_pred_simple, color='red', label='Regression Line')
plt.title("Simple Linear Regression: Price vs. Area")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================================
# 🔸 MULTIPLE LINEAR REGRESSION (Using all features)
# ==========================================================

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)
y_pred_multi = model_multi.predict(X_test)

# Plot predicted vs actual prices based on area
plt.figure(figsize=(10, 6))
plt.scatter(X_test['area'], y_test, color='blue', label='Actual Price')
plt.scatter(X_test['area'], y_pred_multi, color='red', alpha=0.6, label='Predicted Price')
plt.title("Actual vs Predicted Price based on Area (Multiple Regression)")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================================
# 📊 Evaluation Metrics
# ==========================================================

print("🔸 Multiple Linear Regression Evaluation:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_multi):,.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_multi):,.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_multi):.4f}")