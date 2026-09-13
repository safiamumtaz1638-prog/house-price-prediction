# 🏠 House Price Prediction

An end-to-end **Machine Learning web application** that predicts house prices based on important house characteristics. The project uses a **Random Forest Regressor** and is deployed as a Flask web application.

## 🚀 Live Demo

**Live Website:** https://safiamumtaz1638.pythonanywhere.com/

## 📌 Project Overview

This project demonstrates the complete Machine Learning workflow:

**Data → Model Training → Evaluation → Model Saving → Flask Web App → Deployment**

Instead of asking users to provide hundreds of technical features, the final application uses **15 important and user-friendly house features** to make predictions.

## ✨ Features

* Predict house prices using Machine Learning
* Random Forest Regression model
* 15 selected house features
* User-friendly web form
* Input validation
* Error handling
* Prediction result displayed directly on the website
* Clear Form functionality
* Responsive and attractive UI
* Model saved and loaded using Joblib
* Flask backend
* Deployed online using PythonAnywhere

## 🧠 Machine Learning Model

The final model is a **RandomForestRegressor**.

### Selected Features

1. Overall Quality
2. Year Built
3. Living Area
4. Garage Cars
5. Total Basement Area
6. 1st Floor Area
7. Full Bathrooms
8. Bedrooms
9. Total Rooms
10. Garage Area
11. 2nd Floor Area
12. Year Remodeled
13. Fireplaces
14. Wood Deck Area
15. Open Porch Area

## 📊 Model Performance

The model was evaluated using a train/test split with `random_state=42`.

| Metric |     Result |
| ------ | ---------: |
| MAE    |  18,226.06 |
| RMSE   |  28,747.61 |
| R²     | **0.8923** |

The selected-feature model achieved an R² of approximately **0.89** while using only 15 features, making it much more practical for a user-facing application.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Flask
* HTML
* CSS
* Git
* GitHub
* PythonAnywhere

## 📁 Project Structure

```text
house-price-prediction/
│
├── app.py
├── main.py
├── house_price_selected_model.pkl
├── selected_features.pkl
├── requirements.txt
├── .gitignore
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

## ⚙️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/safiamumtaz1638-prog/house-price-prediction.git
```

### 2. Open the project

```bash
cd house-price-prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask application

```bash
python app.py
```

### 5. Open in browser

```text
http://127.0.0.1:5000
```

## 🔮 How the Application Works

The user enters 15 house details through the web form.

```text
User Input
    ↓
HTML Form
    ↓
Flask Backend
    ↓
Selected Features
    ↓
Random Forest Model
    ↓
Predicted House Price
    ↓
Result on Website
```

## 💡 Why 15 Features?

The original dataset contains many features, but requiring users to manually enter hundreds of values would make the application difficult to use.

A selected-feature model was therefore trained using 15 important and easily understandable features.

The original 260-feature model achieved:

**R² = 0.8948**

The selected 15-feature model achieved:

**R² = 0.8923**

The very small difference in performance made the 15-feature model a better choice for the final user-facing application.

## 🌐 Deployment

The Flask application is deployed using **PythonAnywhere**.

Live application:

https://safiamumtaz1638.pythonanywhere.com/

## 📌 Future Improvements

* Add more advanced feature selection
* Improve model accuracy with additional experimentation
* Add prediction history
* Add visual analytics
* Improve UI/UX
* Add multiple regression models for comparison
* Add confidence or prediction range
* Improve deployment configuration

## 👩‍💻 Author

**Safia Mumtaz**

GitHub: https://github.com/safiamumtaz1638-prog
