'''Regression example (House price prediction)

Now we do a regression example.

Inputs: [area (sq ft), number of bedrooms]

Output: house price in lakhs'''

# ----- Regression: House Price Prediction -----

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 1. Sample dataset (features: [area_sqft, bedrooms], target: price in lakhs)
X = [
    [600, 1],
    [800, 1],
    [1000, 2],
    [1200, 2],
    [1500, 3],
    [1800, 3],
    [2000, 3],
    [2200, 4],
    [2500, 4],
    [2800, 5]
]

y = [20, 25, 30, 35, 45, 50, 55, 60, 70, 80]  # prices in lakhs

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Create model (Linear Regression for regression problems)
reg_model = LinearRegression()

# 4. Train the model
reg_model.fit(X_train, y_train)

# 5. Predict on test data
y_pred = reg_model.predict(X_test)

# 6. Evaluate using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# 7. Predict price for a new house
new_house = [[1600, 3]]  # 1600 sqft, 3 bedrooms
predicted_price = reg_model.predict(new_house)
print("Predicted price (in lakhs):", predicted_price[0])