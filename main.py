import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown

# Disease Information Dictionary (unchanged)
disease_info = {
      'Apple___Apple_scab': {
        'description': 'Apple Scab is a fungal disease that affects the leaves and fruit of apple trees.',
        'cause': 'It is caused by the fungus *Venturia inaequalis*.',
        'prevention': 'Prevent by removing fallen leaves, using fungicides, and selecting resistant varieties.'
    },
    'Apple___Black_rot': {
        'description': 'Black Rot affects apple trees, causing lesions on the fruit and branches.',
        'cause': 'Caused by the fungus *Glomerella cingulata*.',
        'prevention': 'Prune infected branches and use fungicides during the growing season.'
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Cedar-Apple Rust causes orange, rust-like spots on apple leaves.',
        'cause': 'Caused by the fungus *Gymnosporangium juniper-virginianae*.',
        'prevention': 'Remove cedar trees near apple trees and apply fungicides in spring.'
    },
    'Apple___healthy': {
        'description': 'The apple tree is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the tree.'
    },
    'Blueberry___healthy': {
        'description': 'The blueberry plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Maintain proper watering and soil conditions.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'description': 'Powdery mildew causes white fungal growth on leaves, stems, and buds of cherry trees.',
        'cause': 'Caused by fungi in the *Erysiphaceae* family.',
        'prevention': 'Use fungicides and prune the tree to improve air circulation.'
    },
    'Cherry_(including_sour)___healthy': {
        'description': 'The cherry plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the tree.'
    },
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot': {
        'description': 'Cercospora leaf spot causes gray to brown lesions on maize leaves.',
        'cause': 'Caused by the fungus *Cercospora zeae-maydis*.',
        'prevention': 'Use resistant varieties, crop rotation, and fungicides.'
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'Common rust causes reddish-brown pustules on maize leaves.',
        'cause': 'Caused by the fungus *Puccinia sorghi*.',
        'prevention': 'Use resistant varieties and apply fungicides.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Northern leaf blight causes long, gray lesions on maize leaves.',
        'cause': 'Caused by the fungus *Exserohilum turcicum*.',
        'prevention': 'Use resistant varieties and apply fungicides.'
    },
    'Corn_(maize)___healthy': {
        'description': 'The corn plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the plant.'
    },
    'Grape___Black_rot': {
        'description': 'Black rot causes dark lesions and shriveling of grape berries.',
        'cause': 'Caused by the fungus *Guignardia bidwellii*.',
        'prevention': 'Prune infected areas and use fungicides.'
    },
    'Grape___Esca_(Black_Measles)': {
        'description': 'Esca causes brown lesions on grape leaves and a decline in plant health.',
        'cause': 'Caused by fungal pathogens including *Phaeomoniella chlamydospora*.',
        'prevention': 'Prune infected vines and avoid wounding during pruning.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': 'Leaf blight causes dark lesions with yellow margins on grape leaves.',
        'cause': 'Caused by the fungus *Isariopsis caprea*.',
        'prevention': 'Use fungicides and remove infected leaves.'
    },
    'Grape___healthy': {
        'description': 'The grape plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the plant.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'description': 'Huanglongbing causes yellowing of leaves and stunted growth in citrus plants.',
        'cause': 'Caused by the bacterium *Candidatus Liberibacter*.',
        'prevention': 'Use disease-free stock and control citrus psyllids.'
    },
    'Peach___Bacterial_spot': {
        'description': 'Bacterial spot causes lesions on peach leaves and fruit.',
        'cause': 'Caused by the bacterium *Xanthomonas campestris*.',
        'prevention': 'Prune infected branches and apply copper-based fungicides.'
    },
    'Peach___healthy': {
        'description': 'The peach tree is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the tree.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'description': 'Bacterial spot causes dark lesions on pepper leaves and fruit.',
        'cause': 'Caused by the bacterium *Xanthomonas campestris*.',
        'prevention': 'Use resistant varieties and remove infected plants.'
    },
    'Pepper,_bell___healthy': {
        'description': 'The bell pepper plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the plant.'
    },
    'Potato___Early_blight': {
        'description': 'Early blight causes dark spots on potato leaves with concentric rings.',
        'cause': 'Caused by the fungus *Alternaria solani*.',
        'prevention': 'Use resistant varieties and rotate crops.'
    },
    'Potato___Late_blight': {
        'description': 'Late blight causes dark, water-soaked lesions on potato leaves.',
        'cause': 'Caused by the fungus *Phytophthora infestans*.',
        'prevention': 'Use fungicides and remove infected plants.'
    },
    'Potato___healthy': {
        'description': 'The potato plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the plant.'
    },
    'Raspberry___healthy': {
        'description': 'The raspberry plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Maintain proper watering and soil conditions.'
    },
    'Soybean___healthy': {
        'description': 'The soybean plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the plant.'
    },
    'Squash___Powdery_mildew': {
        'description': 'Powdery mildew causes white fungal growth on squash leaves.',
        'cause': 'Caused by fungi in the *Erysiphaceae* family.',
        'prevention': 'Use fungicides and improve air circulation around the plant.'
    },
    'Strawberry___Leaf_scorch': {
        'description': 'Leaf scorch causes browning of the edges of strawberry leaves.',
        'cause': 'Caused by various environmental stress factors, including high temperatures.',
        'prevention': 'Water regularly and ensure proper soil nutrition.'
    },
    'Strawberry___healthy': {
        'description': 'The strawberry plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the plant.'
    },
    'Tomato___Bacterial_spot': {
        'description': 'Bacterial spot causes dark lesions on tomato leaves and fruit.',
        'cause': 'Caused by the bacterium *Xanthomonas vesicatoria*.',
        'prevention': 'Use resistant varieties and remove infected plants.'
    },
    'Tomato___Early_blight': {
        'description': 'Early blight causes dark, concentric lesions on tomato leaves.',
        'cause': 'Caused by the fungus *Alternaria solani*.',
        'prevention': 'Use fungicides and remove infected plants.'
    },
    'Tomato___Late_blight': {
        'description': 'Late blight causes dark, water-soaked lesions on tomato leaves.',
        'cause': 'Caused by the fungus *Phytophthora infestans*.',
        'prevention': 'Use fungicides and remove infected plants.'
    },
    'Tomato___Leaf_Mold': {
        'description': 'Leaf mold causes yellowing and mold growth on the undersides of tomato leaves.',
        'cause': 'Caused by the fungus *Cladosporium fulvum*.',
        'prevention': 'Improve air circulation and use fungicides.'
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Septoria leaf spot causes dark, circular lesions on tomato leaves.',
        'cause': 'Caused by the fungus *Septoria lycopersici*.',
        'prevention': 'Use fungicides and rotate crops.'
    },
    'Tomato___healthy': {
        'description': 'The tomato plant is healthy with no signs of disease.',
        'cause': 'No disease detected.',
        'prevention': 'Ensure proper care and maintenance of the plant.'
    },


    # Add other diseases as needed...
    # Your disease_info dictionary here...
}

# Function to download the model from Google Drive
def download_model():
    model_url = "https://drive.google.com/uc?id=15IHjMsa-Qded8-fLEWI4l8VlnqUUQv9g"
    model_path = "trained_plant_disease_model.keras"
    
    if not os.path.exists(model_path):
        st.write("Downloading the model... This may take a few minutes.")
        gdown.download(model_url, model_path, quiet=False)
        st.write("Model downloaded successfully!")
    else:
        st.write("Model already exists locally.")

# TensorFlow Model Prediction
def model_prediction(test_image):
    # Ensure the model is downloaded
    download_model()
    
    # Load the model
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    
    # Resize image to the correct input size (224, 224)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    predictions = model.predict(input_arr)
    predicted_class_index = np.argmax(predictions)  # Get index of class with highest probability
    confidence = predictions[0][predicted_class_index]  # Get the confidence of the prediction
    return predicted_class_index, confidence

# Sidebar (unchanged)
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page (unchanged)
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "th.jpg"  # Replace with your image path
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    """)

# About Project (unchanged)
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset consists of images of healthy and diseased crop leaves categorized into 38 different classes.
                """)

# Prediction Page (unchanged)
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)
    
    # Predict button
    if st.button("Predict"):
        st.snow()  # Show a snowflake animation to indicate prediction is in progress
        st.write("Our Prediction")
        
        # Call model prediction
        result_index, confidence = model_prediction(test_image)
        
        # Class labels for the 38 classes
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        
        # Get disease name
        predicted_disease = class_name[result_index]
        
        # Display the predicted result with confidence
        st.success(f"Model predicts it's a **{predicted_disease}** with a confidence of **{confidence*100:.2f}%**")
        
        # Get disease description, cause, and prevention
        if predicted_disease in disease_info:
            disease_details = disease_info[predicted_disease]
            st.subheader(f"About {predicted_disease.replace('_', ' ').title()}")
            st.markdown(f"**Description:** {disease_details['description']}")
            st.markdown(f"**Cause:** {disease_details['cause']}")
            st.markdown(f"**Prevention:** {disease_details['prevention']}")
        else:
            st.write("No detailed information available for this disease.")
