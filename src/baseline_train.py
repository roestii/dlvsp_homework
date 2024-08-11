import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

from src.datasets.food101 import make_food101
from src.transforms import make_transforms
from src.utils.logging import AverageMeter

class Food101(Dataset):
    def __init__(self):
        pass

class BaseLearner(nn.Module):
    def __init__(self, backbone, n_classes):
        super(BaseLearner, self).__init__()
        self.backbone = backbone
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def main(args):
    batch_size = args["batch_size"]
    root_path = args["root_path"]
    image_folder = args["image_folder"]
    pretrained = args["pretrained"]
    epochs = args["epochs"]

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

    base_learner = BaseLearner(backbone, len(dataset.classes))
    optimizer = torch.optim.Adam(base_learner.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        avg_loss =  AverageMeter()
        it = iter(data_loader)

        for x, y in it:
            optimizer.zero_grad()
            output = base_learner(x)
            loss = criterion(output, y)
            loss.backward()
            avg_loss.update(loss)

        print(f"Average loss: {avg_loss}")
        avg_loss.reset()

    torch.save(base_learner, f"checkpoints/base_learner.pth.tar")

