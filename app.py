#Important Modules
from flask import Flask,render_template, url_for ,flash , redirect
import joblib
from flask import request
import numpy as np
import tensorflow

import os
from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

#from this import SQLAlchemy
app=Flask(__name__,template_folder='template')

dir_path = os.path.dirname(os.path.realpath(__file__))

def get_project_root():
    return dir_path

STATIC_FOLDER = 'static'

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")
 
@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/cancer")
def cancer():
    return render_template("cancer.html")


@app.route("/diabetes")
def diabetes():
    #if form.validate_on_submit():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")


@app.route("/liver")
def liver():
    #if form.validate_on_submit():
    return render_template("liver.html")



def ValuePredictor(to_predict_list, diseaseType):
    to_predict = np.array(to_predict_list).reshape(1,to_predict_list.__len__())
    if(diseaseType=="diabetes"):
        loaded_model = joblib.load("models/diabetes")
        result = loaded_model.predict(to_predict)
    elif(diseaseType=="liver"):
        loaded_model = joblib.load("models/liver")
        result = loaded_model.predict(to_predict)
    elif(diseaseType=="heart"):
        loaded_model = joblib.load("models/heart")
        result =loaded_model.predict(to_predict)
    return result[0]

@app.route('/result',methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        diseaseType = request.args.get('disease')
        result = ValuePredictor(to_predict_list, diseaseType)
    if(int(result)==1):
        prediction='We predict that you are suffering from this disease.\nPlease consult a doctor immediately'
    else:
        prediction='We predict that you are healthy!' 
    return(render_template("result.html", prediction=prediction))


if __name__ == "__main__":
    app.run(debug=True)
