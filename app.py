import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
from keras_preprocessing.image import load_img, img_to_array

img_size = 48

model = tf.keras.models.load_model("AIGeneratedModel.h5")

st.title("AIgeeks Image Classifier")       
        
img = st.file_uploader("Upload your Image")

if img:
    image = Image.open(img)
    st.image(img)
    image = ImageOps.fit(image, (48,48), Image.Resampling.LANCZOS)
    img_array = img_to_array(image)
    new_arr = img_array/255
    test = []
    test.append(new_arr)
    test = np.array(test)
    y = model.predict(test)
    if y[0] <= 0.5:
        st.write("The given image is 99% Real.")
    elif y[0] <= 0.6 :
        st.write("The given image is 83% Real.")
    elif y[0] <= 0.7 :
        st.write("The given image is 67% Real.")
    elif y[0] <= 0.8 :
        st.write("The given image is 55% Real.")
    elif y[0] <= 0.9 :
        st.write("The given image is 49% Real.")
    elif y[0] <= 1 :
        st.write("The given image is 19% Real.")
    else :
        st.write("The given image is 3% Real.")

    
