"""Web demo (upload a photo or use the camera):   streamlit run app.py"""
import os
import numpy as np, pandas as pd, streamlit as st
from PIL import Image
import torch
from dl_utils import load_checkpoint, preprocess_array

st.set_page_config(page_title="Plant & Fruit Detector", layout="wide")
st.title("Plant Disease & Fruit Defect Detector")
st.caption("Models trained on laboratory images and evaluated on field images (lab-to-field gap project)")

@st.cache_resource
def get(path):
    return load_checkpoint(path) if os.path.exists(path) else None

models = {"Plant disease": get("models/plant.pt"), "Fruit defect": get("models/fruit.pt")}
avail = [k for k, v in models.items() if v]
if not avail:
    st.error("No models found in models/. Run train.py first."); st.stop()
task = st.radio("What are you checking?", avail, horizontal=True)
src = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
f = st.file_uploader("Upload a leaf/fruit photo", type=["jpg", "jpeg", "png"]) if src == "Upload" else st.camera_input("Take a photo")
if f:
    img = Image.open(f).convert("RGB"); m, classes, S = models[task]
    with torch.no_grad():
        p = m(preprocess_array(np.array(img), S)).float().softmax(1)[0].cpu().numpy()
    top = np.argsort(p)[::-1][:5]
    c1, c2 = st.columns(2); c1.image(img, use_container_width=True)
    c2.subheader(classes[top[0]].replace("_", " ")); c2.metric("Confidence", f"{p[top[0]]*100:.1f}%")
    if p[top[0]] < 0.6: c2.warning("Low confidence: retake the photo closer, in better light, on a single leaf/fruit.")
    c2.bar_chart(pd.DataFrame({"probability": p[top]}, index=[classes[i].replace("_", " ") for i in top]))
