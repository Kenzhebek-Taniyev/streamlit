import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import pytesseract
import base64
import io
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR.exe'

# Function to convert image to base64
def img_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Function to download processed images in CSV format
def create_csv_download(images):
    csv_data = pd.DataFrame(images, columns=["Image"])
    csv_data['Image'] = csv_data['Image'].apply(lambda x: f'<img src="data:image/png;base64,{x}">')
    csv_data.to_csv('processed_images.csv', index=False)

st.title("CV Model with Streamlit")

uploaded_files = st.file_uploader("Upload Photos", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    filters = st.multiselect("Select Filters", ["Remove Background", "Standard Size", "Change Resolution/Format", "Remove Blur", "Apply Background", "Text/Watermark Detection"])
    
    if "Standard Size" in filters:
        standard_size = st.selectbox("Select Standard Size", ["640x480", "800x600", "1024x768"])

    if "Change Resolution/Format" in filters:
        resolution = st.text_input("Enter Resolution (e.g., 1920x1080)")
        img_format = st.selectbox("Select Format", ["JPEG", "PNG"])

    if "Apply Background" in filters:
        background_choice = st.selectbox("Select Background", ["Nature", "Minimalism", "Solid Color"])

    if st.button("Process Photos"):
        processed_images = []
        
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            img_array = np.array(img)

            if "Remove Background" in filters:
                img_array = remove(img_array)

            if "Standard Size" in filters:
                width, height = map(int, standard_size.split('x'))
                img_array = cv2.resize(img_array, (width, height))

            if "Change Resolution/Format" in filters:
                width, height = map(int, resolution.split('x'))
                img_array = cv2.resize(img_array, (width, height))
                img = Image.fromarray(img_array)
                img_format = img_format.lower()

            if "Remove Blur" in filters:
                img_array = cv2.GaussianBlur(img_array, (5, 5), 0)

            if "Apply Background" in filters:
                if background_choice == "Nature":
                    background = Image.open("backgrounds\\nature.jpg")
                elif background_choice == "Minimalism":
                    background = Image.open("backgrounds\\minimalism.jpg")
                else:
                    background = Image.new("RGB", img.size, "blue")
                
                background = background.resize(img.size)
                img = Image.alpha_composite(background.convert("RGBA"), img.convert("RGBA"))
                img_array = np.array(img)

            if "Text/Watermark Detection" in filters:
                text = pytesseract.image_to_string(img_array)
                st.write("Detected Text/Watermark:")
                st.write(text)
            
            img = Image.fromarray(img_array)
            processed_images.append(img_to_base64(img))

            st.image(img, caption=f"Processed {uploaded_file.name}")

        create_csv_download(processed_images)
        st.success("Photos processed and CSV file created successfully!")

        with open("processed_images.csv", "rb") as f:
            st.download_button(label="Download Processed Images CSV", data=f, file_name="processed_images.csv", mime="text/csv")
