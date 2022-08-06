# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 22:40:14 2022

@author: JER-JESI
"""
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.models.keras.models import load_model
import numpy as np

model=load_model("fruit.h5")
img=image.load_model("E:\\IBMProj\\uploads\\apple.jpg",target_size=(128,128))
x=img_to_array(img)
x=np.expand_dims(x,axis=0)
pred=model.predict_classes(x)
pred