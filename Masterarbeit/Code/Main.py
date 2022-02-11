import torch
import torch.nn.functional as F
from LightningDataloader import *
from Model import *


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = make_kinetics_resnet()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        y_hat = self.model(batch["video"])

        # Compute cross entropy loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        loss = F.cross_entropy(y_hat, batch["label"])

        # Log the train loss to Tensorboard
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """
      Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
      usually useful for training video models.
      """
        return torch.optim.Adam(self.parameters(), lr=1e-1)


def train():
    classification_module = VideoClassificationLightningModule()
    data_module = KineticsDataModule()
    trainer = pytorch_lightning.Trainer(gpus=[9])
    trainer.fit(classification_module, data_module)


if __name__ == '__main__':
    train()