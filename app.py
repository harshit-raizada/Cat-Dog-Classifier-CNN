import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model('final_model.h5')

# Custom styling with CSS
st.markdown(
    """
    <style>
        .big-font {
            font-weight:bold;
            font-size:50px !important;
            color: dodgerblue;
            text-align: center;
        }
        .desc-font {
            font-size:20px !important;
            text-align: center;
        }
        .result-font {
            font-weight:bold;
            font-size:30px !important;
            text-align: center;
        }
        .streamlit-upload-button {
            color: white !important;
            background-color: dodgerblue !important;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# App title
st.markdown('<p class="big-font">Dog 🐶 Vs Cat 🐱 Classification</p>', unsafe_allow_html=True)

# Add some space and description
st.write(' ')
st.markdown(
    """
    <p class="desc-font">Upload an image of a dog or cat, and the model will classify it for you!</p>
    """, 
    unsafe_allow_html=True
)

# Image uploader
uploaded_file = st.file_uploader('Choose an Image!', type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('')
    st.write('Classifying...', style="font-weight: bold; font-size: 24px; text-align: center;")

    # Predict the class of the image
    image_array = np.array(image.resize((256, 256))) / 255
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)

    # Display results
    if prediction[0] >= 0.5:
        st.markdown("<p class='result-font' style='color:green;'>It's a Dog! 🐶</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='result-font' style='color:blue;'>It's a Cat! 🐱</p>", unsafe_allow_html=True)

# Add footer or other useful information
st.markdown('---')
st.write('Model trained using a deep learning CNN architecture.')