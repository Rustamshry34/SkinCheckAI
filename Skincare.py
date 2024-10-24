import streamlit as st
import base64
from PIL import Image
import tensorflow as tf
import numpy as np
from pathlib import Path
import webbrowser
import streamlit as st
import streamlit.components.v1 as components
import io

st.set_page_config(layout="wide")


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)),
                              url(data:image/{"jpg"};base64,{encoded_string});
            background-size: cover;
            background-position: center;
            height: 100vh
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


add_bg_from_local('background3.avif')

# Load the HTML template
with open("index.html", "r") as f:
    html_template = f.read()
st.markdown(html_template, unsafe_allow_html=True)

# Add custom CSS styling
with open("style.css", "r") as f:
    css = f"<style>{f.read()}</style>"
st.markdown(css, unsafe_allow_html=True)



def main():
    # Hide Streamlit's default header/footer
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#@st.cache_resource(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('cancer_model.h5')  # Update with your model path
    return model

model = load_model()

   

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to your model's expected input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

st.markdown(
    """
    <style>
    p.prediction {
        color: white;
        background-color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
        """
        <style>
        .reportview-container {
            background-color: black;
        }
        .stFileUploader label {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
)

st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #7f0808; /* Background color of the app */
        }
        .stFileUploader {
            background-color: #d60606; /* Background color of the uploader */
            border: 2px dashed #7f0808; /* Border style */
            padding: 20px; /* Padding inside the uploader */
            border-radius: 10px; /* Rounded corners */
            width: 600px; /* Width of the uploader */
            margin: auto; /* Center the uploader */
        }
        .stFileUploader label {
            color: #7f0808; /* Text color of the label */
            font-size: 16px; /* Font size of the label */
        }
        .stFileUploader div {
            color: white; /* Text color for drag-and-drop text */
            font-size: 14px; /* Font size for drag-and-drop text */
        }
        </style>
        """,
        unsafe_allow_html=True
)
# Main function to run the app
def main():
        # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image file
        image = Image.open(uploaded_file)
        
        # Display the image
        #st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Predict using the model
        prediction = model.predict(preprocessed_image)
        
        # Show the prediction result
        if prediction[0][0] > 0.5:
            #st.write("The model predicts this as **Skin Cancer**.")
            st.markdown("<div style='text-align: center;'><h3 style='color: white;'>The model predicts this as <strong>Skin Cancer</strong>.</h1></div>", unsafe_allow_html=True)
        else:
            #st.write("The model predicts this as **Normal**.")
            st.markdown("<div style='text-align: center;'><h3 style='color: white;'>The model predicts this as <strong>Normal</strong>.</h1></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()