# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 12:29:05 2022

@author: Admin
"""
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#load the model
model=load_model('fruit.h5')
img=image.load_img(r'E:\\IBMProj\\uploads\\AppleHealthy2.JPG',target_size=(128,128))
print(img)

x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
y=np.argmax(model.predict(x),axis=1)
i=y[0];
index=['Pepper_bell_Bacterial_spot','Pepper_bell_healthy','Potato_Early_blight','Potato_healthy','Potato_:ate_blight','Tomato_Bacterial_spot','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot']
print(index[i])