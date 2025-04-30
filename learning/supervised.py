"""
Supervised learning: The model learns from labeled data (you give it both inputs and correct outputs).

Commong models:
- Linear regression
- Logistic regression
- Deciosion trees
- Random forest
- Support Vector Machine
- K-Nearest neighbor
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression


# ------------- Linear Regression -------------------------
# Data
X = np.array([[1000], [1500], [1700], [2000], [2200], [2500], [2600], [3000]])
y = np.array([200000, 250000, 265000, 300000, 320000, 350000, 375000, 400000])

# Creating the model
model = LinearRegression()

# Traing the model: fit
model.fit(X, y)

# Make predictions
predicted_price = model.predict([[1750]])
print(f"Predicted price for {1750} sqft: ${predicted_price[0]:,.2f}")

# Visualize
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, model.predict(X), color="red", label="Linear regression line")
plt.xlabel("House Size (sqft)")
plt.ylabel("Price ($)")
plt.title("Linear Regression - House Price Prediction")
plt.legend()
plt.show()

# ------------- Logistic Regression -------------------------
# Data
Xlog = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
ylog = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Train model
model = LogisticRegression()
model.fit(Xlog, ylog)

study_hours = 4.5
probability = model.predict_proba([[study_hours]])[0][1]
prediction = model.predict([[study_hours]])[0]

print(f"Probability of passing with {study_hours} hours of study: {probability:.2f}")
print("Prediction:", "Pass" if prediction == 1 else "Fail")

X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_test)[:, 1]

plt.plot(X_test, y_prob, color="blue", label="Pass Probability")
plt.scatter(Xlog, ylog, color="red", label="Training Data")
plt.axhline(y=0.5, color="gray", linestyle="--", label="0.5 Threshold")
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression - Pass Prediction")
plt.legend()
plt.grid(True)
plt.show()
