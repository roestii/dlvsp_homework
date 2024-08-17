import torch
from torch import nn
from torchvision.models import resnet18

from src.datasets.food101 import make_food101
from src.transforms import make_transforms
from src.utils.logging import AverageMeter
from src.models.baseline import BaselineModel

def main(args):
    batch_size = args["batch_size"]
    root_path = args["root_path"]
    image_folder = args["image_folder"]
    pretrained = args["pretrained"]
    epochs = args["epochs"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transforms = make_transforms()
    dataset, data_loader = make_food101(
        transforms, 
        batch_size, 
        root_path=root_path,
        image_folder=image_folder
    )

    backbone = resnet18(weights=pretrained)
    # remove imagenet classification head
    backbone = nn.Sequential(*(list(backbone.children())[:-1]))

    base_learner = BaselineModel(backbone, len(dataset.classes))
    base_learner = base_learner.to(device)

    optimizer = torch.optim.Adam(base_learner.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        avg_loss =  AverageMeter()
        it = iter(data_loader)

        for x, y in it:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = base_learner(x)
            loss = criterion(output, y)
            loss.backward()
            avg_loss.update(loss)

        print(f"Average loss: {avg_loss.avg}")
        avg_loss.reset()
        torch.save(base_learner, f"checkpoints/base_learner_ep{epoch}.pth.tar")

    torch.save(base_learner, f"checkpoints/base_learner_final.pth.tar")
