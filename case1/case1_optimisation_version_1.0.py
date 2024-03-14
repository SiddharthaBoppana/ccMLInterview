# Version 1.0
import torch
import torchvision.models as models
from flask import Flask, request, jsonify
import pickle
import yaml

app = Flask(__name__)

# Load the pre-trained model (initialize only once)
model = models.vit_l_32()
model.eval()

# Load configuration from cfg.yaml (if needed)
CFG_PATH = './cfg.yaml'
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        img_file = request.files["train_image"]
        img_bytes = img_file.read()
        img = pickle.loads(img_bytes)

        # Process the image using the pre-loaded model
        with torch.no_grad():
            output = model(img)

        return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=True)