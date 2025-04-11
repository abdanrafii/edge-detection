import streamlit as st
import cv2
from edge_detection import (
    read_image, apply_sobel, apply_prewitt, apply_roberts,
    apply_compass, apply_log
)
from PIL import Image
import numpy as np

st.set_page_config(page_title="Edge Detection App", layout="wide")
st.title("Edge Detection")

uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg', 'webp'])

algorithm = st.selectbox(
    "Choose Edge Detection Algorithm",
    ["Sobel", "Prewitt", "Roberts", "Compass", "LoG (Marr-Hildreth)"]
)

if uploaded_file is not None:
    image = read_image(uploaded_file)
    st.image(image, caption='Original Image')

    if st.button("Run Edge Detection"):
        if algorithm == "Sobel":
            result = apply_sobel(image)
        elif algorithm == "Prewitt":
            result = apply_prewitt(image)
        elif algorithm == "Roberts":
            result = apply_roberts(image)
        elif algorithm == "Compass":
            result = apply_compass(image)
        elif algorithm == "LoG (Marr-Hildreth)":
            result = apply_log(image)
        else:
            result = None

        if result is not None:
            st.image(result, caption=f"{algorithm} Result", channels="GRAY")
