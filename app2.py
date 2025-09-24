import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import os

# Header
st.header('Image Classification Model')

# Load the pre-trained model
model = load_model('fruit_vege_classify.keras')

# Categories of fruits and vegetables
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
            'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 
            'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 
            'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 
            'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Create 'Images' folder if it doesn't exist
if not os.path.exists('Images'):
    os.makedirs('Images')

# File uploader for image
image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if image_file is not None:
    # Save the uploaded image to the 'Images' folder
    img_path = os.path.join('Images', image_file.name)
    with open(img_path, "wb") as f:
        f.write(image_file.getbuffer())

    # Open the uploaded image
    image = Image.open(img_path)
    
    #Display the image 
    st.image(image, caption="Uploaded Image")

    # Resize the image to match the input size expected by the model (e.g., 180x180)
    img_height = 180
    img_width = 180
    image_resized = image.resize((img_width, img_height))
    
    # Convert image to numpy array
    img_arr = np.array(image_resized)
    
    # Check if the image has 3 channels (RGB), if not, convert it to RGB
    if img_arr.shape[-1] != 3:
        img_arr = np.stack([img_arr] * 3, axis=-1)  # Convert grayscale to RGB
    
    # Normalize the image
    img_arr = tf.keras.utils.array_to_img(img_arr)
    img_bat=tf.expand_dims(img_arr,0)
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
 
    st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
    #st.write('With accuracy of ' + str(np.max(score)*100))