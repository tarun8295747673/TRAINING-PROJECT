from flask import Flask, render_template, request , jsonify 

import google.generativeai as genai

from sklearn.linear_model import LinearRegression
import pandas as pd

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

app = Flask(__name__)
app.secret_key = 'your_secret_key'


csv_path = 'datasetFile.csv'  
data = pd.read_csv(csv_path)

X = data[['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4', 'Parameter 5', 'Parameter 6', 'Parameter 7', 'Parameter 8']]
y = data['Parameter 9']
X = X.to_numpy()


model1 = LinearRegression()
model1.fit(X, y)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        inputs = [float(request.form[field]) for field in ['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4', 'Parameter 5', 'Parameter 6', 'Parameter 7', 'Parameter 8']]
        prediction = model1.predict([inputs])
        output = "Argument 1" if prediction[0] >= 0.5 else "Argument 2"
        return render_template('index1.html', prediction_text=f'{output}')
    return render_template('index1.html')
