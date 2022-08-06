import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
import os

app = Flask(__name__)

model = load_model("fruit.h5")
model1 = load_model("vegetable.h5")


@app.route('/')
def home():
    return render_template('/home.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():
    return render_template('/predict.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(128,128))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        plant=request.form['plant']
        print(plant)
        if(plant=="vegetable"):
            y=np.argmax(model.predict(x),axis=1)
            i=y[0];
            index=['Pepper_bell_Bacterial_spot','Pepper_bell_healthy','Potato_Early_blight','Potato_healthy','Potato_:ate_blight','Tomato_Bacterial_spot','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot']
            preds=index[i]
            print(preds)
            df=pd.read_excel('E:\\IBMProj\\precautions-veg.xlsx')
            print(df.iloc[y[0]]['caution'])
        else:
            y=np.argmax(model.predict(x),axis=1)
            i=y[0];
            index=['Apple_Black_Rot','Apple_Healthy','Corn_Healthy','Corn_Northern_Leaf_Blight','Peach_Bacterial_spot','Peach_Healthy']
            preds=index[i]
            print(preds)
            df=pd.read_excel('E:\\IBMProj\\precautions-veg.xlsx')
            print(df.iloc[y[0]]['caution'])   
        # pred=np.argmax(model.predict(x),axis=1)
        # index=['Bear','Crow','Elephant','Rat']
        # text="The Classified Animal is : " +str(index[pred[0]])
        return df.iloc[y[0]]['caution']
if __name__=='__main__':
    app.run(debug=False)