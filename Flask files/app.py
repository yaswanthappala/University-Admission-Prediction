import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
app = Flask(__name__)
model= load('proj.save')
trans=load('datatransform')

@app.route('/')
def home():
    return render_template('Graduates.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]
    print(x_test)
    x_test=np.delete(x_test,[2,3],axis=1)
    test=trans.transform(x_test)
    print(test)
    prediction = model.predict(test)
    print(prediction)
    output=prediction[0]
    return render_template('Graduates.html', prediction_text='Chance of Admit {}'.format(output))

'''@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    #For direct API calls trought request
    
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)'''

if __name__ == "__main__":
    app.run(debug=True)
