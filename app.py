from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_selected_model.pkl")
# Load feature columns
feature_columns = joblib.load("selected_features.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        new_house = {
            "OverallQual": float(request.form["OverallQual"]),
            "YearBuilt": float(request.form["YearBuilt"]),
            "GrLivArea": float(request.form["GrLivArea"]),
            "GarageCars": float(request.form["GarageCars"]),
            "TotalBsmtSF": float(request.form["TotalBsmtSF"]),
            "1stFlrSF": float(request.form["1stFlrSF"]),
            "FullBath": float(request.form["FullBath"]),
            "BedroomAbvGr": float(request.form["BedroomAbvGr"]),
            "TotRmsAbvGrd": float(request.form["TotRmsAbvGrd"]),
            "GarageArea": float(request.form["GarageArea"]),
            "2ndFlrSF": float(request.form["2ndFlrSF"]),
            "YearRemodAdd": float(request.form["YearRemodAdd"]),
            "Fireplaces": float(request.form["Fireplaces"]),
            "WoodDeckSF": float(request.form["WoodDeckSF"]),
            "OpenPorchSF": float(request.form["OpenPorchSF"])
        }

        # Create DataFrame with selected features
        new_house_df = pd.DataFrame(
            0,
            index=[0],
            columns=feature_columns
        )

        # Add user values
        for feature, value in new_house.items():
            if feature in new_house_df.columns:
                new_house_df[feature] = value

        # Prediction
        prediction = model.predict(new_house_df)[0]

        return render_template(
            "index.html",
            prediction=round(prediction, 2),
            form_data=new_house
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            error="Please enter valid values in all fields."
        )

    except Exception as e:
        return render_template(
            "index.html",
            error="Something went wrong. Please try again."
        )

if __name__ == "__main__":
    app.run()