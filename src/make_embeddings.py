import os
import numpy as np 

from src.datasets.food101 import make_food101
from src.models import vision_transformer
from src.helper import load_encoder
from src.transforms import make_transforms

def main(args):
    model_name = args["model_name"]
    crop_size = args["crop_size"]
    patch_size = args["patch_size"]
    image_folder = args["image_folder"]
    checkpoint_path = args["checkpoint_path"]
    out_folder = args["out_folder"]

    encoder = vision_transformer.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)
    encoder = load_encoder(encoder, checkpoint_path)
    transforms = make_transforms(crop_size=crop_size)
    dataset, dataloader = make_food101(
        transforms, 
        1, 
        training=False,
        root_path="",
        image_folder=image_folder
    )

    embeddings = {cl: [] for cl in dataset.classes}
    it = iter(dataloader)

    for x, y in it:
        embedding = encoder(x)
        embeddings[y].append(embedding)

    for k, v in embeddings.items():
        fpath = os.path.join(out_folder, f"{k}.npy")
        with open(fpath, "wb") as file:
            v = np.array(v)
            np.save(file, v)
