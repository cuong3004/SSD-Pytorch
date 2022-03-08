
import torch
from torch import nn
import pytorch_lightning as pl

class VOCLightning(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()

        self.save_hyperparameters("lr")

        self.model = model

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.hparams.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        loss = self.model(x, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        loss = self.model(x, y)
        self.log('val_loss', loss)

if __name__ == "__main__":
    from data_module import VOCLightningModule
    voc_data_module = VOCLightningModule()
    voc_data_module.prepare_data()
    x, y = next(iter(voc_data_module.train_dataloader()))

    from model import get_model
    model = get_model()
    model_lit = VOCLightning(model)
    print(x, y)