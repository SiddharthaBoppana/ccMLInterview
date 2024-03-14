import torch
import torchvision.models as models
from flask import Flask, request
import pickle
import yaml

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        img_file = request.files["train_image"]
        img_bytes = img_file.read()
        img = pickle.loads(img_bytes)

        model = models.vit_l_32()
        model.eval()

        with torch.no_grad():
            output = model(img)

        return {'success': True}

if __name__ == "__main__":

    CFG_PATH = './cfg.yaml'
    with open(CFG_PATH, "r") as f: # load cfg
        cfg = yaml.safe_load(f)

    app.run(
        debug=True
        )
