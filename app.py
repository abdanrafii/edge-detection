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

algorithms = st.multiselect(
    "Choose Edge Detection Algorithms",
    ["Sobel", "Prewitt", "Roberts", "Compass", "LoG (Marr-Hildreth)"],
    default=["Sobel"]
)

if uploaded_file is not None:
    image = read_image(uploaded_file)
    st.image(image, caption='Original Image')

    if st.button("Run Edge Detection"):
        if not algorithms:
            st.warning("Please select at least one algorithm")
        else:
            algorithm_map = {
                "Sobel": apply_sobel,
                "Prewitt": apply_prewitt,
                "Roberts": apply_roberts,
                "Compass": apply_compass,
                "LoG (Marr-Hildreth)": apply_log
            }

            num_cols = len(algorithms)
            cols = st.columns(num_cols)
            
            for i, algo_name in enumerate(algorithms):
                result = algorithm_map[algo_name](image)
                with cols[i]:
                    st.image(result, 
                            caption=f"{algo_name} Result", 
                            channels="GRAY",
                            use_container_width =True)