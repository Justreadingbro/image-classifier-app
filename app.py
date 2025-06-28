import tensorflow as tf
import gradio as gr
import numpy as np
import urllib.parse

# Load models once to save time
models = {
    "MobileNetV2": tf.keras.applications.MobileNetV2(weights="imagenet"),
    "EfficientNetB0": tf.keras.applications.EfficientNetB0(weights="imagenet")
}

# Preprocess config per model
preprocess_map = {
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input
}

decode_map = {
    "MobileNetV2": tf.keras.applications.mobilenet_v2.decode_predictions,
    "EfficientNetB0": tf.keras.applications.efficientnet.decode_predictions
}


def classify_image(image, model_name):
    image = image.resize((224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    preprocess = preprocess_map[model_name]
    model = models[model_name]
    decode = decode_map[model_name]

    image_array = preprocess(image_array)
    predictions = model.predict(image_array)
    decoded = decode(predictions, top=3)[0]

    # Prepare label dictionary
    results = {label: float(score) for (_, label, score) in decoded}

    # Top prediction Wikipedia link
    top_label = decoded[0][1].replace("_", " ")
    wiki_link = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(top_label)}"

    return results, wiki_link


interface = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(["MobileNetV2", "EfficientNetB0"], value="EfficientNetB0", label="Select Model")
    ],
    outputs=[
        gr.Label(num_top_classes=3, label="Top Predictions"),
        gr.Textbox(label="Wikipedia Link for Top Class")
    ],
    title="Advanced AI Image Classifier",
    description="Upload or take a picture to classify using ImageNet-trained MobileNetV2 or EfficientNetB0.",
    theme="soft",
    examples=[
        ["examples/dog.jpg", "EfficientNetB0"],
        ["examples/cat.jpg", "MobileNetV2"]
    ]
)

if __name__ == "__main__":
    interface.launch()
