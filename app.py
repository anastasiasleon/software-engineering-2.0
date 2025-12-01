import streamlit as st
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import torch

# Загрузка предварительно обученной модели и процессора изображений
@st.cache_resource
def load_model():
    model_name = "google/vit-base-patch16-224"
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    return feature_extractor, model

feature_extractor, model = load_model()

def classify_image(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

st.title("Классификатор изображений")
st.write("Загрузите изображение для классификации")

uploaded_file = st.file_uploader("Выбрать изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    st.write("")
    st.write("Классификация...")

    label = classify_image(image)
    st.write(f"Предсказано: {label}")
