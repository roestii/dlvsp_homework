from src.datasets.food101 import make_food101
from src.models import vision_transformer
from src.helper import load_encoder
from src.transforms import make_transforms

def main(args):
    model_name = args["model_name"]
    crop_size = args["crop_size"]
    patch_size = args["patch_size"]
    batch_size = args["batch_size"]
    image_folder = args["image_folder"]
    checkpoint_path = args["checkpoint_path"]

    encoder = vision_transformer.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size)
    encoder = load_encoder(encoder, checkpoint_path)
    transforms = make_transforms(crop_size=crop_size)
    dataset, dataloader = make_food101(
        transforms, 
        batch_size, 
        training=False,
        root_path="",
        image_folder=image_folder
    )

    it = iter(dataloader)
    for x, y in it:
        embedding = encoder(x)
        print(embedding)
        print(y)
        break
        
