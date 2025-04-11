import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Human-Action-Recognition"  # Change to your updated model path
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# ID to Label mapping
id2label = {
    0: "calling",
    1: "clapping",
    2: "cycling",
    3: "dancing",
    4: "drinking",
    5: "eating",
    6: "fighting",
    7: "hugging",
    8: "laughing",
    9: "listening_to_music",
    10: "running",
    11: "sitting",
    12: "sleeping",
    13: "texting",
    14: "using_laptop"
}

def classify_action(image):
    """Predicts the human action in the image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {id2label[i]: round(probs[i], 3) for i in range(len(probs))}
    return predictions

# Gradio interface
iface = gr.Interface(
    fn=classify_action,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Action Prediction Scores"),
    title="Human Action Recognition",
    description="Upload an image to recognize the human action (e.g., dancing, calling, sitting, etc.)."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
