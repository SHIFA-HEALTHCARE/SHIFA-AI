#Important Modules
from flask import Flask,render_template,jsonify,request
import joblib
import numpy as np
import os

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


def prediction_response(result):
    if(int(result)==1):
        prediction = {
            "Suffering" : True
            }
        response =  jsonify(prediction)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    else:
        prediction = {
            "Suffering" : False
            }
        response = jsonify(prediction)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/p_diabetes',methods = ["GET"])
def predict_diabetes():
    if request.method == 'GET':
        diseaseType = "diabetes"
        
        Pregnancies = request.args.get('pregnancies', -1, int)
        Glucose = request.args.get('glucose', -1, int)
        BloodPressure = request.args.get('blood-pressure', -1, int)
        SkinThickness = request.args.get('skin-thickness', -1, int)
        Insulin = request.args.get('insulin', -1, int)
        Bmi = request.args.get('bmi', -1, float)
        DiabetesPedigreeFunction = request.args.get('dpf', -1, float)
        Age = request.args.get('age', -1, int)
    
        to_predict_list = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,Bmi,DiabetesPedigreeFunction,Age,]

        if any(value < 0 for value in to_predict_list):
            response = {
            "Error" : "Invalid Arguments"
            }
  
            return jsonify(response)

        result = ValuePredictor(to_predict_list, diseaseType)

    return prediction_response(result)

@app.route('/p_heart',methods = ["GET"])
def predict_heart():
    if request.method == 'GET':
        diseaseType = "heart"
        
        Age = request.args.get('age', -1, int)
        isMale = request.args.get('male', False, type=lambda v: v.lower() == 'true')
        Cpt = request.args.get('chest-pain-type', -1, int)
        TrestBPS = request.args.get('trest-bps', -1, int)
        Cholestrol = request.args.get('cholestrol', -1, int)
        RestECG = request.args.get('rest-ecg', -1, int)
        Thalach = request.args.get('thalach', -1, int)
        Exang = request.args.get('exang', False, type=lambda v: v.lower() == 'true')
        OldPeak = request.args.get('old-peak', -1, float)
        Slope = request.args.get('slope', -1, int)
        Thal = request.args.get('thal', -1, int)
        
        to_predict_list = [Age,int(isMale == True),Cpt,TrestBPS,Cholestrol,RestECG,Thalach,int(Exang == True),OldPeak,Slope,Thal]

        if any(value < 0 for value in to_predict_list) or (Cpt > 3) or (RestECG > 2) or (Slope > 2) or (Thal > 3):
            response = {
            "Error" : "Invalid Arguments"
            }
  
            return jsonify(response)

        result = ValuePredictor(to_predict_list, diseaseType)

    return prediction_response(result)

@app.route('/p_liver',methods = ["GET"])
def predict_liver():
    if request.method == 'GET':
        diseaseType = "liver"
        
        TotalBilirubin = request.args.get('total-bilirubin', -1, float)
        DirectBilirubin = request.args.get('direct-bilirubin', -1, float)
        AlkalinePhosphotase = request.args.get('alkaline-phosphotase', -1, int)
        AlamineAminotransferase = request.args.get('alamine-aminotransferase', -1, int)
        TotalProtiens = request.args.get('total-protiens', -1, float)
        Albumin = request.args.get('albumin', -1, float)
        AlbuminGlobulinRatio = request.args.get('albumin-globulin-ratio', -1, float)
    
        to_predict_list = [TotalBilirubin,DirectBilirubin,AlkalinePhosphotase,
        AlamineAminotransferase,TotalProtiens,Albumin,AlbuminGlobulinRatio]

        if any(value < 0 for value in to_predict_list):
            response = {
            "Error" : "Invalid Arguments"
            }
  
            return jsonify(response)

        result = ValuePredictor(to_predict_list, diseaseType)

    return prediction_response(result)

if __name__ == "__main__":
    app.run(debug=True)
