# Flask
import flask
from flask import request, jsonify, abort
from flask_cors import CORS
from StockPricePrediction import get_predicted_data

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

@app.route('/', methods=['POST'])
def get_data():
    print("************************************")
    print(request.get_json())
    print("************************************")
    query = request.get_json()["query"]
    last_val, nextdata, accuracy = get_predicted_data(query)
    print(jsonify( {'last' : last_val, 'next' : nextdata, 'accuracy' : accuracy } ))
    return jsonify( {'last' : last_val, 'next' : nextdata, 'accuracy' : accuracy } )
app.run()