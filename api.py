from flask import Flask,request,jsonify
from classifier import get_prediction

app=Flask(__name__)

@app.route('/')
def get_data():
    return("hello world")

@app.route('/predict-alphabet',methods=["POST"])
def predict_data():
    image=request.files.get("alphabet")
    predicted_value=get_prediction(image)
    return jsonify({
        'prediction':predicted_value
    },200)

if __name__=="__main__":
    app.run(debug=True)