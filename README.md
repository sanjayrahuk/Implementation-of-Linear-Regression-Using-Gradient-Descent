# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed.

2.Scale the features using a standard scaler to normalize the data.

3.Initialize model parameters (theta) and add an intercept term to the feature set.

4.Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function.

5.Make predictions on new data by transforming it using the same scaling and encoding applied to the training data.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SANJAY RAHUL A
RegisterNumber:  212224060234
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Load dataset
data = pd.read_csv("exp_3_50_Startups.csv")

# Display first few rows
print("First 5 rows of the dataset:")
display(data.head())
# One-hot encode the 'State' column
data_encoded = pd.get_dummies(data, columns=['State'], drop_first=True)

# Display the encoded dataset
print("Encoded dataset:")
display(data_encoded.head())
# Independent features (all columns except Profit)
X_raw = data_encoded.drop('Profit', axis=1).values

# Target variable (Profit)
y_raw = data_encoded['Profit'].values.reshape(-1, 1)

print("Shape of features:", X_raw.shape)
print("Shape of target:", y_raw.shape)
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X = X_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw)
# Add a column of ones to X for bias term
m = X.shape[0]
X = np.hstack((np.ones((m, 1)), X))
print("Shape of X after adding bias term:", X.shape)
def compute_cost(X, y, theta):
    """
    Compute Mean Squared Error cost.
    """
    m = len(y)
    preds = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((preds - y) ** 2)
    return cost
def gradient_descent(X, y, learning_rate=0.01, num_iters=2000, tol=1e-8, verbose=False):
    """
    Batch Gradient Descent for Linear Regression.
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []

    prev_cost = compute_cost(X, y, theta)

    for i in range(num_iters):
        preds = X.dot(theta)
        errors = preds - y
        grad = (1 / m) * (X.T.dot(errors))
        theta -= learning_rate * grad

        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        if abs(prev_cost - cost) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break
        prev_cost = cost

        if verbose and (i % 500 == 0 or i < 5):
            print(f"Iteration {i:4d}, Cost: {cost:.6f}")

    return theta, J_history
alpha = 0.01
theta, J_hist = gradient_descent(X, y, learning_rate=alpha, num_iters=5000, tol=1e-9, verbose=True)

print("\nLearned Parameters (Theta):")
print(theta.flatten())
plt.figure(figsize=(7,4))
plt.plot(J_hist)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE/2)')
plt.title('Cost Function Convergence')
plt.grid(True)
plt.show()
# Create new sample (same feature names as original)
new_sample = pd.DataFrame([{
    'R&D Spend': 165349.2,
    'Administration': 136897.8,
    'Marketing Spend': 471784.1,
    'State': 'New York'
}])

# Apply one-hot encoding
new_encoded = pd.get_dummies(new_sample, columns=['State'], drop_first=True)

# Align columns with training data (add missing ones as 0)
new_encoded = new_encoded.reindex(columns=data_encoded.drop('Profit', axis=1).columns, fill_value=0)

# Scale new data
new_scaled = X_scaler.transform(new_encoded)

# Add bias
new_design = np.hstack((np.ones((new_scaled.shape[0], 1)), new_scaled))

# Predict (scaled)
scaled_pred = new_design.dot(theta)

# Inverse transform to original units
pred_original = y_scaler.inverse_transform(scaled_pred)

print(f"\nPredicted Profit for the new startup: ₹{pred_original[0][0]:,.2f}")
```

## Output:
<img width="501" height="197" alt="image" src="https://github.com/user-attachments/assets/d815af06-6794-4517-a74b-dd7f822d44f3" /><br><br>


<img width="642" height="197" alt="image" src="https://github.com/user-attachments/assets/64bae2b5-1112-41d2-8286-6c5973f61c80" /><br><br>


<img width="217" height="47" alt="image" src="https://github.com/user-attachments/assets/33e5c4a2-1771-41f1-86c7-c89c00ff2afe" /><br><br>


<img width="342" height="37" alt="image" src="https://github.com/user-attachments/assets/9ca6ec7b-db08-44b7-a32f-ca8d609c5dde" /><br><br>


<img width="500" height="262" alt="image" src="https://github.com/user-attachments/assets/119bc4b1-6816-47ec-9626-e246945e820f" /><br><br>


<img width="571" height="367" alt="image" src="https://github.com/user-attachments/assets/7def8871-96be-4e27-968c-a1ca0f6a8e55" /><br><br>


<img width="382" height="17" alt="image" src="https://github.com/user-attachments/assets/febf4acf-b066-410e-98f0-41e983623d4f" /><br><br>

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
