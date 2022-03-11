
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from custom_data import CustomVoc
from torchvision import transforms as T
import pytorch_lightning as pl

class VOCLightningModule(pl.LightningDataModule):
    def __init__(self, data_dir='.', batch_size=32, num_workers=0, transforms=None):
        super().__init__(batch_size)
        self.batch_size = batch_size
        self.transforms = transforms
        self.data_dir = data_dir
        self.num_workers = num_workers

    def prepare_data(self):
        data_train_voc = VOCDetection(
                    root=self.data_dir, download="True", 
                    image_set="train", 
                    year='2007',
                    transform=T.ToTensor())
        self.data_train = CustomVoc(data_train_voc)
        
        data_val_voc = VOCDetection(
                    root=self.data_dir, download="True", 
                    image_set="val", 
                    year='2007',
                    transform=T.ToTensor())
        self.data_val = CustomVoc(data_val_voc)
        
        data_test_voc = VOCDetection(
                    root=self.data_dir, download="True", 
                    image_set="test", 
                    year='2007',
                    transform=T.ToTensor())
        self.data_test = CustomVoc(data_test_voc)

    def train_dataloader(self):
        dataset = self.data_train
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        dataset = self.data_val
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
        return loader
    
    def tesst_dataloader(self):
        dataset = self.data_train
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
        return loader


if __name__ == "__main__":
    voc_data_module = VOCLightningModule()
    voc_data_module.prepare_data()
    x, y = next(iter(voc_data_module.train_dataloader()))
    print(x, y)
        # break