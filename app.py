import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Sheep Lung AI Classifier",
    page_icon="🔬",
    layout="centered"
)

categories = [
    "Undetermined_D",
    "Healthy_B",
    "Inflammation_A",
    "Neoplastic_C"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "sheep_lung_efficientnet.keras",
        compile=False
    )

st.title("🔬 Sheep Lung AI Classifier")
st.write(
    "Experimental AI-powered classification "
    "of macroscopic sheep lung photographs."
)

st.warning(
    "Research prototype trained on 40 images. "
    "Not validated for veterinary diagnosis."
)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a sheep lung photograph",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded lung photograph")

    if st.button("Analyze Image", type="primary"):

        resized = image.resize((224, 224))
        array = np.array(resized, dtype=np.float32)
        array = np.expand_dims(array, axis=0)

        predictions = model.predict(array, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))

        st.subheader("Classification Result")

        st.success(
            f"Predicted category: "
            f"{categories[predicted_index]}"
        )

        st.write("**Model prediction scores**")

        for category, probability in zip(categories, predictions):
            st.write(
                f"{category}: {probability * 100:.1f}%"
            )
            st.progress(float(probability))

        st.caption(
            "Scores are uncalibrated model outputs, "
            "not diagnostic probabilities."
        )
