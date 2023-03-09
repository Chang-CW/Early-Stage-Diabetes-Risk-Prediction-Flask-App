from flask import Flask, request, jsonify
import pickle
import numpy as np


app = Flask(__name__)

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 16)
    loaded_model = pickle.load(open("pkl/RandomForestClassifier.pkl", "rb"))
    result = loaded_model.predict_proba(to_predict)
    return round(result[0][1]*100, 2)

@app.route('/api', methods = ['GET'])
def returnProb():
    # d = {}
    # inputchr = str(request.args['query'])
    # answer = str(ord(inputchr))
    # d['output'] = answer
    # return d
    d = {}
    X = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden_weight_loss',
         'weakness', 'Polyphagia', 'Genital_thrush', 'visual_blurring',
         'Itching', 'Irritability', 'delayed_healing', 'partial_paresis',
         'muscle_stiffness', 'Alopecia', 'Obesity']
    to_predict_list = []
    for x in X:
        to_predict_list.append(int(request.args['cough']))
    # to_predict_list.append(int(request.args['cough']))    
    # to_predict_list.append(int(request.args['fever']))
    # to_predict_list.append(int(request.args['sore_throat']))
    # to_predict_list.append(int(request.args['shortness_of_breath']))
    # to_predict_list.append(int(request.args['head_ache']))
    # to_predict_list.append(int(request.args['age_60_and_above']))
    # to_predict_list.append(int(request.args['gender']))
    # to_predict_list.append(int(request.args['test_indication']))
    d['output'] = str(ValuePredictor(to_predict_list))
    # return str(to_predict_list)
    return d

if __name__ =="__main__":
    app.run()
