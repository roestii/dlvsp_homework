import os
import numpy as np 
import torch

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
    batch_size = args["batch_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        transforms = make_transforms(crop_size=crop_size)

        for mode in ["val/", "test/"]:
            data_path = os.path.join(out_folder, mode)
            if not os.path.exists(data_path):
                os.mkdir(os.path.join(out_folder, mode))

            dataset, dataloader = make_food101(
                transforms, 
                batch_size, 
                mode=mode,
                root_path="",
                image_folder=image_folder
            )

            it = iter(dataloader)
            for i, (x, y) in enumerate(it):
                print(f"Iteration: {i}")
                x = x.to(device)
                embedding = encoder(x)
                emb_path = os.path.join(data_path, f"embedding_{i}.pth")
                y_path = os.path.join(data_path, f"class_{i}.pth")
                torch.save(embedding, emb_path)
                torch.save(y, y_path)
                del embedding
