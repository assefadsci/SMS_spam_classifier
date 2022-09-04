# import libraries
import pickle
from flask import Flask,request, render_template
import pandas as pd

#create flask app
app= Flask(__name__)

#Load the pickle model
classifier= pickle.load(open('classifier.pkl', 'rb'))
vectorize= pickle.load(open('vectorize.pkl', 'rb'))


# define homepage
@app.route('/')
def home():

        return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():

    sms= request.form['sms']
    sms= [sms]
    sms_vec= vectorize.transform(sms)
    spam_prediction= classifier.predict(sms_vec)
    return render_template('result.html', prediction= spam_prediction)

if __name__=='__main__':
    app.run(debug= True)
