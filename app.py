from flask import Flask, render_template, request
import joblib
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# 📌 Load trained model
model = joblib.load("model.pkl")
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 📌 Get input values from form
        features = [
            int(request.form["Store"]),
            int(request.form["Dept"]),
            float(request.form["Temperature"]),
            float(request.form["Fuel_Price"]),
            float(request.form["MarkDown1"]),
            float(request.form["MarkDown2"]),
            float(request.form["MarkDown3"]),
            float(request.form["MarkDown4"]),
            float(request.form["MarkDown5"]),
            float(request.form["Unemployment"]),
            int(request.form["Size"]),
            int(request.form["Year"]),
            int(request.form["Month"]),
            int(request.form["Week"]),
            int(request.form["IsHoliday"]),
            int(request.form["Type"]),
            int(request.form["up"]),

        ]
        
        # 🛑 Debug: Print features
        print("Features received:", features)
        print("Total features received:", len(features))
        print("Model expects features:", model.get_booster().feature_names)


        # 📌 Convert to NumPy array
        features_array = np.array([features])

        # 📌 Make prediction
        predicted_cpi = model.predict(features_array)[0]

        return render_template("result.html", prediction=predicted_cpi)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
