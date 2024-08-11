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
    batch_size = args["batch_size"]

    encoder = vision_transformer.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)
    encoder = load_encoder(encoder, checkpoint_path)
    transforms = make_transforms(crop_size=crop_size)

    for mode in ["val/", "test/"]:
        os.mkdir(os.path.join(out_folder, mode))
        dataset, dataloader = make_food101(
            transforms, 
            batch_size, 
            mode="val/",
            root_path="",
            image_folder=image_folder
        )

        for cl in dataset.classes:
            path = os.path.join(out_folder, mode, cl)
            os.mkdir(path)

        it = iter(dataloader)
        for i, (xs, ys) in enumerate(it):
            print(f"iteration {i}")
            embeddings = encoder(xs)
            for k in range(len(embeddings)):
                cl = dataset.classes[ys[k]]
                path = os.path.join(out_folder, mode, cl, f"{i}_{k}.npy")
                with open(path, "wb") as file:
                    embedding = embeddings[i].detach().numpy()
                    np.save(file, embedding)
