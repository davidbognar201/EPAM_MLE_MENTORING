from flask import Flask, render_template, request, flash
import numpy as np
import pickle
import os
import src.preprocessing as pre

app = Flask(__name__)



@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    input_dict = {
        "age" : float(request.form["age"]),
        "TSH" : float(request.form["tsh"]),
        "T3" : float(request.form["t3"]),
        "TT4" : float(request.form["tt4"]),
        "T4U" : float(request.form["t4u"]),
        "FTI" : float(request.form["fti"]),
        "sex" : request.form.get("sex"),
        "on_thyroxine" : request.form.getlist("on_thyroxine")[-1],
        "on_antithyroid_medication" : request.form.getlist("on_antithyroid_medication")[-1],
        "sick" : request.form.getlist("sick")[-1],
        "pregnant" : request.form.getlist("pregnant")[-1],
        "thyroid_surgery" : request.form.getlist("thyroid_surgery")[-1],
        "I131_treatment" : request.form.getlist("i131_surgery")[-1],
        "lithium" : request.form.getlist("lithium")[-1],
        "goitre" : request.form.getlist("goitre")[-1],
        "tumor" : request.form.getlist("tumor")[-1],
        "hypopituitary" : request.form.getlist("hypopituitary")[-1],
        "psych" : request.form.getlist("psych")[-1]
    }
    result, prob = pre.makePrediction(input_dict)
    return render_template('index.html', **locals())
    
    
    
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)