import streamlit as st
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Function to load and predict the image
def getPrediction(filename):
    # Load your model
    model = tf.keras.models.load_model("final_model_weights.hdf5")
    
    # Load the image
    img = load_img(filename, target_size=(180, 180))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    
    # Get the prediction
    probability = model.predict(img)
    category = np.argmax(probability, axis=1)
    
    # Return category and probability
    if category[0] == 1:
        answer = "Recycle"
        probability_result = probability[0][1]
    else:
        answer = "Organic"
        probability_result = probability[0][0]
    
    return answer, probability_result, filename

# Streamlit UI
st.title("Is this Recyclable?")
st.write("Upload your image to determine if the item is organic or recyclable.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Make prediction
    answer, probability, filename = getPrediction("temp_image.jpg")
    
    # Display the result
    st.write(f"Prediction: {answer}")
    st.write(f"Probability: {probability*100:.2f}%")

