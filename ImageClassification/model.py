import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze=True):
        super(Model, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
        # Freeze baseline
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Replace classification head
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, img):
        out = self.resnet(img)

        return out


if __name__ == '__main__':
    model = Model(101)
    test = torch.randn(1, 3, 224, 224)
    out = model(test)
    print(out.shape)