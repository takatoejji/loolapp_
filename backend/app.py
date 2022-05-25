
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
import traceback
from flask_restful import reqparse
app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    return "hey"

@app.route('/predict', methods=['POST'])
def predict():
	model = joblib.load("model.pkl")
	if model:
		try:
			json = request.get_json()	 
			temp=list(json[0].values())
			vals=np.array(temp)
			prediction = model.predict(temp)
			print("here:",prediction)        
			return jsonify({'prediction': str(prediction[0])})

		except:        
			return jsonify({'trace': traceback.format_exc()})
	else:
		return (' Sorry no model here to use')
    


if __name__ == '__main__':
    app.run(debug=True)
    