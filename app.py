from flask import Flask, request
import pandas as pd
from _collections import OrderedDict
import joblib

app = Flask(__name__)


@app.route("/api/actual")
def get():
    sex = float(request.args["Male"])
    age = float(request.args["Age"])
    blood_pressure = float(request.args["BP"])
    heart_rate = float(request.args["HR"])
    ecg = float(request.args["Ecg"])
    temp = float(request.args["Temp"])
    model_folder_name = "IdentifyHeartAttackModel/"
    model_file_path = (
        model_folder_name + "identify_heart_attack_randomforest_model.joblib"
    )
    model_file = open(model_file_path, "rb")
    load_trained_model = joblib.load(model_file)
    request_data = OrderedDict(
        [
            ("age", age),
            ("sex", sex),
            ("trestbps", blood_pressure),
            ("restecg", ecg),
            ("thalach", heart_rate),
            ("temp", temp),
        ]
    )
    reshaped_request_data = pd.Series(request_data).values.reshape(1, -1)
    actual_heart_attack_prediction = load_trained_model.predict(reshaped_request_data)
    return str(actual_heart_attack_prediction)


if __name__ == "__main__":
    app.run()
