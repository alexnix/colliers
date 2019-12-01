from flask import Flask
from flask import Response
from flask import render_template
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np

from main import compute
compute_result = compute("db_comparables.csv")

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/train')
def train():
    data = pd.read_csv("db_comparables.csv")
    data = data.loc[data['Judet'] == 'Bucuresti']
    return render_template("analysis.html", name="Bucuresti", data=data.to_html())

@app.route('/test')
def test():
    data = pd.read_csv("eval.csv")
    data = data.loc[data['Judet'] == 'Bucuresti']
    return render_template("analysis.html", name="Bucuresti", data=data.to_html())

@app.route('/result')
def result():
    global compute_result
    return render_template("analysis.html", name="Bucuresti", data=compute_result.to_html())

@app.route('/api')
@cross_origin()
def api():
    global compute_result
    return compute_result.to_json()


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="3030", debug=True,
    	threaded=True, use_reloader=False)
