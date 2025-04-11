![zfdggzdrg.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/DPx-7s4BmG_XocnPQ4TR9.png)
# **Human-Action-Recognition**

> **Human-Action-Recognition** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for multi-class human action recognition. It uses the **SiglipForImageClassification** architecture to predict human activities from still images.

```py
Classification Report:
                    precision    recall  f1-score   support

           calling     0.8525    0.7571    0.8020       840
          clapping     0.8679    0.7119    0.7822       840
           cycling     0.9662    0.9857    0.9758       840
           dancing     0.8302    0.8381    0.8341       840
          drinking     0.9093    0.8714    0.8900       840
            eating     0.9377    0.9131    0.9252       840
          fighting     0.9034    0.7905    0.8432       840
           hugging     0.9065    0.9000    0.9032       840
          laughing     0.7854    0.8583    0.8203       840
listening_to_music     0.8494    0.7988    0.8233       840
           running     0.8888    0.9321    0.9099       840
           sitting     0.5945    0.7226    0.6523       840
          sleeping     0.8593    0.8214    0.8399       840
           texting     0.8195    0.6702    0.7374       840
      using_laptop     0.6610    0.9190    0.7689       840

          accuracy                         0.8327     12600
         macro avg     0.8421    0.8327    0.8339     12600
      weighted avg     0.8421    0.8327    0.8339     12600
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/O9ir2VwHirB-T75ABCP7m.png)

The model categorizes images into 15 action classes:

- **0:** calling  
- **1:** clapping  
- **2:** cycling  
- **3:** dancing  
- **4:** drinking  
- **5:** eating  
- **6:** fighting  
- **7:** hugging  
- **8:** laughing  
- **9:** listening_to_music  
- **10:** running  
- **11:** sitting  
- **12:** sleeping  
- **13:** texting  
- **14:** using_laptop  

---

# **Run with Transformers 🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
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
```

---

# **Intended Use**

The **Human-Action-Recognition** model is designed to detect and classify human actions from images. Example applications:

- **Surveillance & Monitoring:** Recognizing suspicious or specific activities in public spaces.  
- **Sports Analytics:** Identifying player activities or movements.  
- **Social Media Insights:** Understanding trends in user-posted visuals.  
- **Healthcare:** Monitoring elderly or patients for activity patterns.  
- **Robotics & Automation:** Enabling context-aware AI systems with visual understanding.
