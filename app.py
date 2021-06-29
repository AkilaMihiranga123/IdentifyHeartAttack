from flask import Flask, request
import pandas as pd
from _collections import OrderedDict
import joblib

app=Flask(__name__)

@app.route('/api')
def get():
    male=float(request.args['Male'])
    age=float(request.args['Age'])
    bp=float(request.args['BP'])
    hp=float(request.args['HP'])
    ecg=float(request.args['Ecg'])
    temp=float(request.args['Temp'])
    outFileFolder = 'output/'
    filePath = outFileFolder + 'randomforest_model.joblib'
    file = open(filePath, "rb")
    trained_model = joblib.load(file)
    new_data=OrderedDict([('age',age),('sex',male),('trestbps',bp),('restecg',ecg),('thalach',hp),('temp',temp)])
    new_data=pd.Series(new_data).values.reshape(1,-1)
    prediction = trained_model.predict(new_data)
    return str(prediction)


if __name__ == '__main__':
    app.run()
