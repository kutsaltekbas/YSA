import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# Load the dataset
dataset = pd.read_csv("data.csv")

# Define features and target
X = dataset.drop('Salary', axis=1)
Y = dataset['Salary']

# Define categorical and numerical features
categorical_features = ["Job Title", "Employment Type", "Experience Level", "Expertise Level", "Company Location", "Employee Residence", "Company Size", "Year", "Salary Currency"]
numerical_features = dataset.columns.difference(categorical_features + ['Salary'])

# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the training data
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Scale the target variable
scaler = StandardScaler()
Y_train_scaled = scaler.fit_transform(Y_train.values.reshape(-1, 1)).ravel()
Y_test_scaled = scaler.transform(Y_test.values.reshape(-1, 1)).ravel()

# Train the Random Forest Regression model
from sklearn.svm import SVR

kernels = ["linear", "poly", "rbf", "sigmoid"]

for kernel in kernels:
    svr = SVR(kernel=kernel)
    svr.fit(X_train_processed, Y_train_scaled)

    # Predicting the test set results
    Y_pred = svr.predict(X_test_processed)

    # Predict the test set results
    Y_pred_scaled = svr.predict(X_test_processed)

    # Inverse transform the scaled predictions and actual values
    Y_pred = scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1))
    Y_test = scaler.inverse_transform(Y_test_scaled.reshape(-1, 1))

    # Calculate R-squared
    r_squared = r2_score(Y_test, Y_pred)

    print(f"{kernel} R-squared: {r_squared:.2f}")
    
    # Flatten Y_test for comparison
    Y_test_flat = Y_test.ravel()

    # Plot predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test_flat, Y_pred, edgecolors=(0, 0, 0), alpha=0.6)
    plt.plot([Y_test_flat.min(), Y_test_flat.max()], [Y_test_flat.min(), Y_test_flat.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.5, 0.95, f'R-squared: {r_squared:.2f}', horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes)
    plt.title(f'{kernel} Actual vs Predicted Salary')
    plt.show()
