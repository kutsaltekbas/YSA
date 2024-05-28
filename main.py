import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Veri setini yükleyelim
file_path = rf'./data.csv'  # Dosya yolunuza göre güncelleyin
dataset = pd.read_csv(file_path)

X = dataset.drop('Salary', axis=1)
Y = dataset['Salary']

categorical_features = ["Job Title", "Employment Type", "Experience Level", "Expertise Level", "Company Location", "Employee Residence", "Company Size", "Year", "Salary Currency"]
numerical_features = dataset.columns.difference(categorical_features + ['Salary'])

# Pipeline oluşturulması
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Verileri eğitim ve test setlerine bölelim
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Verileri ölçeklendirme ve dönüştürme pipeline'ı
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Train-validation split manually
X_train_processed, X_val_processed, y_train, y_val = train_test_split(X_train_processed, Y_train, test_size=0.2, random_state=42)

# Scale the target variable
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = scaler.transform(y_val.values.reshape(-1, 1))
y_test_scaled = scaler.transform(Y_test.values.reshape(-1, 1))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=400, restore_best_weights=True)

def train(learning_rate):
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=150, activation="relu", input_shape=(X_train_processed.shape[1],)))
    ann.add(tf.keras.layers.Dropout(0.2))
    ann.add(tf.keras.layers.Dense(units=300, activation="relu"))
    ann.add(tf.keras.layers.Dropout(0.2))
    ann.add(tf.keras.layers.Dense(units=500, activation="relu"))
    ann.add(tf.keras.layers.Dropout(0.2))
    ann.add(tf.keras.layers.Dense(units=300, activation="relu"))
    ann.add(tf.keras.layers.Dropout(0.2))
    ann.add(tf.keras.layers.Dense(units=150, activation="relu"))
    ann.add(tf.keras.layers.Dropout(0.2))
    ann.add(tf.keras.layers.Dense(units=1, activation="linear"))

    ann.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mean_absolute_error", metrics=["mean_absolute_error"])

    history = ann.fit(X_train_processed, y_train_scaled, batch_size=32, epochs=20000, validation_data=(X_val_processed, y_val_scaled), callbacks=[early_stopping])
    
    return ann, history

def calculate_r_squared(ann):
    y_pred_scaled = ann.predict(X_test_processed)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test_scaled, y_pred_scaled)
    return r2

ann, history = train(0.001)

# Plot the training history
plt.figure(figsize=(14, 5))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('YSA (0.0001) Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

# Predict on the test set
y_pred_scaled = ann.predict(X_test_processed)
y_pred = scaler.inverse_transform(y_pred_scaled)

# Plot the actual vs predicted values
plt.subplot(1, 2, 2)
plt.scatter(Y_test, y_pred, edgecolors=(0, 0, 0), alpha=0.6)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('YSA (0.001) Actual vs Predicted Salary')

# Calculate R-squared
r_squared = calculate_r_squared(ann)
plt.text(0.5, 0.95, f'R-squared: {r_squared:.2f}', horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

ann, history = train(0.001)

# Plot the training history
plt.figure(figsize=(14, 5))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('YSA (0.0001) Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

# Predict on the test set
y_pred_scaled = ann.predict(X_test_processed)
y_pred = scaler.inverse_transform(y_pred_scaled)

# Plot the actual vs predicted values
plt.subplot(1, 2, 2)
plt.scatter(Y_test, y_pred, edgecolors=(0, 0, 0), alpha=0.6)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('YSA (0.0001) Actual vs Predicted Salary')

# Calculate R-squared
r_squared = calculate_r_squared(ann)
plt.text(0.5, 0.95, f'R-squared: {r_squared:.2f}', horizontalalignment='center', verticalalignment='top', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
