from torch import nn

class BaselineModel(nn.Module):
    def __init__(self, backbone, n_classes):
        super(BaselineModel, self).__init__()
        self.backbone = backbone
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
