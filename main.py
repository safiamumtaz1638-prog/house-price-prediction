from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import pandas as pd

df = pd.read_csv("house-prices/train.csv")

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

X = df[selected_features]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=40,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Selected Feature Model")
print("----------------------")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

joblib.dump(model, "house_price_selected_model.pkl")
joblib.dump(selected_features, "selected_features.pkl")

print("Model saved successfully!")