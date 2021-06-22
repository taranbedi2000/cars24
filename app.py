import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
myjson=open('columns.json','r')
jsonData=myjson.read()
obj=json.loads(jsonData)
r=0
for x in obj:
        if isinstance(obj[x], list):
            r += len(obj[x])

def predict_price(year, kilometers_driven, mileage, engine, power, seats, company, name, location, fuel_type, transmission, owner_type):
    x2 = [0]*r
    x2[0] = int(year)
    x2[1] = kilometers_driven
    x2[2] = mileage
    x2[3] = engine
    x2[4] = power
    x2[5] = seats
    l=[]
    for w in obj:
        for y in obj[w]:
           l.append(y)
    w=l
    for i in range(6,r):
        if w[i]==company:
            x2[i] = 1
        if w[i]==name:
            x2[i] = 1
        if w[i]==location:
            x2[i] = 1
        if w[i]==fuel_type:
            x2[i] = 1
        if w[i]==transmission:
            x2[i] = 1
        if w[i]==owner_type:
            x2[i] = 1
    result = ([x2])
    return result

def xyz(ash):
    c=predict_price(ash[0],ash[1],ash[2],ash[3],ash[4],ash[6],ash[5],ash[7],ash[8],ash[9],ash[10],ash[11])
    return c
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   
    int_features = [x for x in request.form.values()]
    final_features = list(int_features)
    k=xyz(final_features)
    
    prediction = model.predict(k)

    output = prediction[0]

    return render_template('index.html',prediction_text='Price is ₹ {} lakhs'.format(output))


if __name__ == "__main__":
    app.run(debug=True)