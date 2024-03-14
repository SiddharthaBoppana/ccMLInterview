# Version 2.0

import torch
import torchvision.models as models
from flask import Flask, request, jsonify
import pickle
import yaml
from time import time
from tqdm import tqdm
from dataloader import get_train_dataloader

app = Flask(__name__)

# Load the pre-trained model (initialize only once)
model = models.vit_l_32()
model.eval()

# Load configuration from cfg.yaml (if needed)
CFG_PATH = './cfg.yaml'
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# Cache for memoization
image_cache = {}

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        img_file = request.files["train_image"]
        img_bytes = img_file.read()

        # Check if image is already in cache
        if img_bytes in image_cache:
            output = image_cache[img_bytes]
        else:
            img = pickle.loads(img_bytes)

            # Process the image using the pre-loaded model
            with torch.no_grad():
                output = model(img)

            # Store result in cache
            image_cache[img_bytes] = output

        return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=True)
