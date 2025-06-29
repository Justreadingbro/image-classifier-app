import gradio as gr
import tensorflow as tf
import numpy as np
import urllib.parse
from PIL import Image, ImageEnhance, ImageFilter
import logging
import time
import json
from typing import List, Tuple, Dict, Optional
import requests
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedImageClassifier:
    def __init__(self):
        """Initialize the classifier with multiple models and preprocessing options."""
        self.models = {}
        self.preprocess_map = {}
        self.decode_map = {}
        self.input_sizes = {}
        self.load_models()
        
    def load_models(self):
        """Load pre-trained models with error handling."""
        model_configs = {
            "MobileNetV2": {
                "model": tf.keras.applications.MobileNetV2,
                "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
                "decode": tf.keras.applications.mobilenet_v2.decode_predictions,
                "input_size": (224, 224)
            },
            "EfficientNetB0": {
                "model": tf.keras.applications.EfficientNetB0,
                "preprocess": tf.keras.applications.efficientnet.preprocess_input,
                "decode": tf.keras.applications.efficientnet.decode_predictions,
                "input_size": (224, 224)
            },
            "ResNet50": {
                "model": tf.keras.applications.ResNet50,
                "preprocess": tf.keras.applications.resnet50.preprocess_input,
                "decode": tf.keras.applications.resnet50.decode_predictions,
                "input_size": (224, 224)
            },
            "InceptionV3": {
                "model": tf.keras.applications.InceptionV3,
                "preprocess": tf.keras.applications.inception_v3.preprocess_input,
                "decode": tf.keras.applications.inception_v3.decode_predictions,
                "input_size": (299, 299)
            }
        }
        
        for name, config in model_configs.items():
            try:
                logger.info(f"Loading {name}...")
                self.models[name] = config["model"](weights="imagenet")
                self.preprocess_map[name] = config["preprocess"]
                self.decode_map[name] = config["decode"]
                self.input_sizes[name] = config["input_size"]
                logger.info(f"✓ {name} loaded successfully")
            except Exception as e:
                logger.error(f"✗ Failed to load {name}: {str(e)}")

    def enhance_image(self, image: Image.Image, enhancement_type: str) -> Image.Image:
        """Apply image enhancement techniques."""
        if enhancement_type == "None":
            return image
        elif enhancement_type == "Sharpen":
            return image.filter(ImageFilter.SHARPEN)
        elif enhancement_type == "Enhance Contrast":
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.5)
        elif enhancement_type == "Enhance Brightness":
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(1.2)
        elif enhancement_type == "Auto Enhance":
            enhanced = image.filter(ImageFilter.SHARPEN)
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.3)
            enhancer = ImageEnhance.Brightness(enhanced)
            return enhancer.enhance(1.1)
        return image

    def preprocess_image(self, image: Image.Image, model_name: str, enhancement: str) -> np.ndarray:
        """Preprocess image for model input with enhancements."""
        try:
            image = self.enhance_image(image, enhancement)

            input_size = self.input_sizes[model_name]
            image = image.resize(input_size, Image.Resampling.LANCZOS)

            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = self.preprocess_map[model_name](image_array)
            
            return image_array
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def classify_single(self, image: Image.Image, model_name: str, 
                       enhancement: str = "None", top_k: int = 5) -> Tuple[Dict, str, Dict]:
        """Classify a single image with detailed results."""
        try:
            start_time = time.time()
            
            image_array = self.preprocess_image(image, model_name, enhancement)
            
            predictions = self.models[model_name].predict(image_array, verbose=0)
            inference_time = time.time() - start_time
            
            decoded = self.decode_map[model_name](predictions, top=top_k)[0]

            results = {}
            confidence_scores = []
            for i, (class_id, label, score) in enumerate(decoded):
                clean_label = label.replace("_", " ").title()
                results[clean_label] = float(score)
                confidence_scores.append(score)

            top_label = decoded[0][1].replace("_", " ")
            wiki_link = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(top_label)}"
            wiki_html = f'<a href="{wiki_link}" target="_blank" style="color: #1976d2; text-decoration: none;">📖 {top_label.title()} on Wikipedia</a>'

            metadata = {
                "inference_time": f"{inference_time:.3f}s",
                "model_used": model_name,
                "enhancement_applied": enhancement,
                "top_confidence": f"{confidence_scores[0]:.1%}",
                "prediction_spread": f"{confidence_scores[0] - confidence_scores[-1]:.3f}"
            }
            
            return results, wiki_html, metadata
            
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            return {"Error": f"Classification failed: {str(e)}"}, "", {}

    def classify_batch(self, images: List[Image.Image], model_name: str, 
                      enhancement: str = "None", top_k: int = 3) -> Tuple[List, str, Dict]:
        """Classify multiple images efficiently."""
        try:
            start_time = time.time()
            results = []
            wiki_links = []
            all_confidences = []
            
            for i, img in enumerate(images):
                try:
                    pred_dict, wiki_link, _ = self.classify_single(img, model_name, enhancement, top_k)
                    
                    if "Error" not in pred_dict:
                        pred_str = " | ".join([f"{label}: {score:.1%}" for label, score in pred_dict.items()])
                        top_confidence = list(pred_dict.values())[0]
                        all_confidences.append(top_confidence)
                    else:
                        pred_str = pred_dict["Error"]
                        top_confidence = 0.0
                    
                    results.append([
                        f"Image {i+1}",
                        pred_str,
                        f"{top_confidence:.1%}" if top_confidence > 0 else "N/A"
                    ])
                    wiki_links.append(wiki_link)
                    
                except Exception as e:
                    logger.error(f"Error processing image {i+1}: {str(e)}")
                    results.append([f"Image {i+1}", f"Error: {str(e)}", "N/A"])
                    wiki_links.append("")
            
            valid_links = [link for link in wiki_links if link]
            if valid_links:
                combined_links_html = "<div style='max-height: 200px; overflow-y: auto;'>" + "<br>".join(valid_links) + "</div>"
            else:
                combined_links_html = "<p>No Wikipedia links available</p>"
            
            total_time = time.time() - start_time
            avg_confidence = np.mean(all_confidences) if all_confidences else 0
            metadata = {
                "total_images": len(images),
                "successful_predictions": len(all_confidences),
                "total_time": f"{total_time:.2f}s",
                "avg_time_per_image": f"{total_time/len(images):.3f}s",
                "average_confidence": f"{avg_confidence:.1%}",
                "model_used": model_name
            }
            
            return results, combined_links_html, metadata
            
        except Exception as e:
            logger.error(f"Error in batch classification: {str(e)}")
            return [["Error", str(e), "N/A"]], "", {}

    def get_model_info(self, model_name: str) -> str:
        """Get information about the selected model."""
        model_info = {
            "MobileNetV2": "🚀 **MobileNetV2**: Lightweight model optimized for mobile devices. Fast inference with good accuracy.",
            "EfficientNetB0": "⚡ **EfficientNetB0**: Excellent balance of accuracy and efficiency using compound scaling.",
            "ResNet50": "🎯 **ResNet50**: Deep residual network with 50 layers. Robust performance across diverse images.",
            "InceptionV3": "🔬 **InceptionV3**: Advanced architecture with factorized convolutions. High accuracy for complex scenes."
        }
        return model_info.get(model_name, "Model information not available.")

classifier = AdvancedImageClassifier()

css = """
.gradio-container {
    max-width: 1200px !important;
}
.metadata-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
.model-info {
    background: #f8f9fa;
    padding: 10px;
    border-radius: 8px;
    border-left: 4px solid #007bff;
}
"""

with gr.Blocks(css=css, title="Advanced AI Image Classifier") as demo:
    gr.HTML("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🤖 Advanced AI Image Classifier</h1>
        <p style='font-size: 18px; color: #666;'>
            State-of-the-art image classification with multiple models and enhancement options
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(classifier.models.keys()),
                value=list(classifier.models.keys())[0] if classifier.models else None,
                label="🔧 Choose Model",
                info="Select the neural network architecture"
            )
            
            model_info_display = gr.Markdown(
                value=classifier.get_model_info(list(classifier.models.keys())[0]) if classifier.models else "",
                elem_classes=["model-info"]
            )
            
            enhancement_dropdown = gr.Dropdown(
                choices=["None", "Sharpen", "Enhance Contrast", "Enhance Brightness", "Auto Enhance"],
                value="None",
                label="✨ Image Enhancement",
                info="Apply preprocessing to improve classification"
            )
            
            top_k_slider = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="📊 Number of Predictions",
                info="How many top predictions to show"
            )

        with gr.Column(scale=2):
            mode = gr.Radio(
                ["Single Image", "Batch Upload"], 
                value="Single Image", 
                label="📂 Classification Mode"
            )

            image_input = gr.Image(
                type="pil", 
                label="📷 Upload Image",
                height=300
            )
            
            batch_input = gr.File(
                file_types=["image"], 
                file_count="multiple", 
                label="📁 Upload Multiple Images",
                visible=False
            )

    predict_btn = gr.Button(
        "🔍 Classify Images", 
        variant="primary", 
        size="lg"
    )

    with gr.Row():
        with gr.Column():
            output_label = gr.Label(
                label="🎯 Top Predictions",
                num_top_classes=10
            )
            
            output_html = gr.HTML(label="🔗 Wikipedia Reference")
            
            batch_output = gr.Dataframe(
                headers=["Image", "Predictions", "Confidence"],
                label="📋 Batch Results",
                visible=False,
                wrap=True
            )
            
            batch_links_html = gr.HTML(
                label="🔗 Wikipedia Links",
                visible=False
            )

        with gr.Column():
            metadata_output = gr.JSON(
                label="📈 Classification Metadata",
                visible=True
            )

    def update_model_info(model_name):
        return classifier.get_model_info(model_name)

    def toggle_inputs(selected_mode):
        is_single = selected_mode == "Single Image"
        return (
            gr.update(visible=is_single),
            gr.update(visible=not is_single),
            gr.update(visible=is_single),
            gr.update(visible=is_single),
            gr.update(visible=not is_single),
            gr.update(visible=not is_single),
        )

    def predict(image, files, model_name, selected_mode, enhancement, top_k):
        if not model_name or model_name not in classifier.models:
            error_msg = "Please select a valid model"
            return None, error_msg, None, None, {"error": error_msg}
        
        try:
            if selected_mode == "Single Image" and image is not None:
                results, wiki_html, metadata = classifier.classify_single(
                    image, model_name, enhancement, int(top_k)
                )
                return results, wiki_html, None, None, metadata
                
            elif selected_mode == "Batch Upload" and files is not None and len(files) > 0:
                images = []
                for file in files:
                    img = Image.open(file).convert("RGB")
                    images.append(img)
                
                preds, combined_links_html, metadata = classifier.classify_batch(
                    images, model_name, enhancement, 3
                )
                return None, None, preds, combined_links_html, metadata
                
            else:
                error_msg = "Please upload an image or select files for batch processing"
                return None, error_msg, None, None, {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Classification error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg, None, None, {"error": error_msg}

    model_dropdown.change(
        fn=update_model_info,
        inputs=[model_dropdown],
        outputs=[model_info_display]
    )

    mode.change(
        fn=toggle_inputs,
        inputs=[mode],
        outputs=[image_input, batch_input, output_label, output_html, batch_output, batch_links_html]
    )

    predict_btn.click(
        fn=predict,
        inputs=[image_input, batch_input, model_dropdown, mode, enhancement_dropdown, top_k_slider],
        outputs=[output_label, output_html, batch_output, batch_links_html, metadata_output]
    )

    gr.Examples(
        examples=[
            ["Single Image", "EfficientNetB0", "Auto Enhance", 5],
            ["Batch Upload", "ResNet50", "Sharpen", 3],
            ["Single Image", "InceptionV3", "Enhance Contrast", 7],
        ],
        inputs=[mode, model_dropdown, enhancement_dropdown, top_k_slider],
        label="💡 Quick Start Examples"
    )

    gr.HTML("""
    <div style='text-align: center; padding: 20px; color: #666; border-top: 1px solid #eee; margin-top: 20px;'>
        <p>🚀 Powered by TensorFlow & Gradio | 🔬 Advanced Computer Vision Pipeline</p>
        <p><small>Supports: JPEG, PNG, WebP | Max file size: 10MB per image</small></p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )
