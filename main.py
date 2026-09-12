import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# House Price Prediction - Model Training
# --------------------------------------------------
# This script is used only to train the ML model.
# The trained model is later used by app.py.
# --------------------------------------------------


DATA_PATH = "house-prices/train.csv"

MODEL_PATH = "house_price_selected_model.pkl"
FEATURES_PATH = "selected_features.pkl"


# Selected features used by the final model
selected_features = [
    "OverallQual",
    "YearBuilt",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "GarageArea",
    "2ndFlrSF",
    "YearRemodAdd",
    "Fireplaces",
    "WoodDeckSF",
    "OpenPorchSF"
]


# --------------------------------------------------
# Check dataset
# --------------------------------------------------

if not os.path.exists(DATA_PATH):
    print("Dataset not found!")
    print(f"Please place train.csv inside: {DATA_PATH}")
    print("This script is only required when training/retraining the model.")
    raise SystemExit


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# --------------------------------------------------
# Features and target
# --------------------------------------------------

X = df[selected_features]
y = df["SalePrice"]


# --------------------------------------------------
# Train / Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# Random Forest Model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=40,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\nSelected Feature Model")
print("----------------------")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)


# --------------------------------------------------
# Save model and selected features
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)

joblib.dump(selected_features, FEATURES_PATH)


print("\nModel saved successfully!")
print("Model:", MODEL_PATH)
print("Features:", FEATURES_PATH)