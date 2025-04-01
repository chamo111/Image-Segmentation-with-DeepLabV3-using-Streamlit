import streamlit as st
import cv2
from PIL import Image
from run_model import run

st.set_page_config(layout = 'wide')

fl = st.file_uploader("upload an image")

if fl is not None:
    if "image" not in fl.type:
        st.warning("Only images only")
    else:
        img = Image.open(fl.name)
        st.image(img)
        with st.spinner("model running"):
            objects, segments = run(fl.name)
        cols = st.columns(len(objects))
        for i , col in enumerate(cols):
            col.header(objects[i])
            col.image(segments[i], channels = "BGR") # Show the segmented image for that object (OpenCV uses BGR)