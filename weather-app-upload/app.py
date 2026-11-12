import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG = 160  # ukuran input sama dengan saat training

st.set_page_config(page_title="Klasifikasi Cuaca - Kelompok 14", page_icon="🌦️")


@st.cache_resource
def load():
    model = tf.keras.models.load_model("best_model.keras")
    names = json.load(open("class_names.json"))
    return model, names


model, class_names = load()

st.title("Klasifikasi Cuaca (Custom CNN) - Kelompok 14")
st.write("Unggah foto langit/cuaca. Model mengenali: " + ", ".join(class_names) + ".")

file = st.file_uploader("Unggah foto cuaca", type=["jpg", "jpeg", "png", "bmp", "gif"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img)

    # preprocessing sama dengan notebook: RGB -> resize 160x160 (Rescaling ada di dalam model)
    arr = np.array(img)
    x = tf.image.resize(arr, (IMG, IMG)).numpy()[None]
    p = model.predict(x, verbose=0)[0]

    order = np.argsort(p)[::-1]
    st.subheader(f"Prediksi: {class_names[order[0]]} ({p[order[0]]:.1%})")
    for i in order:
        st.write(f"{class_names[i]}: {p[i]:.1%}")
        st.progress(float(p[i]))
