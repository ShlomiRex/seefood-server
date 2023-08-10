# from flask import Flask
# from datasets import load_dataset_builder
# from datasets import load_dataset
# import os
# from IPython.display import Image, display
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")


def predict(image):
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()

    predicted_labels_str = model.config.id2label[predicted_label]

    predicted_labels = []
    for s in predicted_labels_str.split(","):
        predicted_labels.append(s.strip())

    print("Predicted: ")
    print(predicted_labels)

    return predicted_labels


def is_hotdog(labels):
    for lbl in labels:
        if lbl in ["hotdog", "hot dog"]:
            return True
    return False


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.post("/api/")
def api():
    print("Got API request")

    print("File:")
    file = request.files['file']
    image = Image.open(file)

    print(image)
    labels = predict(image)
    hotdog = is_hotdog(labels)
    if hotdog:
        print("Yes Hotdog")
    else:
        print("Not hotdog")
    return jsonify({
        "is_hotdog": hotdog
    })


if __name__ == "__main__":
    app.run()
