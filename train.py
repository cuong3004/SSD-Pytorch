
from random import shuffle
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader

class MNISTClassifier(LightningMNISTClassifier):
    def __init__(self, data_dir=None, batch_size=32, num_workers=0, transforms=None):
        super().__init__(batch_size)
        self.batch_size = batch_size
        self.transforms = transforms
        self.data_dir = data_dir
        self.num_workers = num_workers

    def prepare_data(self):
        self.data_train = VOCDetection(
                    root=self.data_dir, download="True", 
                    image_set="train", 
                    year='2007'
                    transforms=transforms)
        
        self.data_val = VOCDetection(
                    root=self.data_dir, download="True", 
                    image_set="val", 
                    year='2007')
        
        self.data_test = VOCDetection(
                    root=self.data_dir, download="True", 
                    image_set="test", 
                    year='2007')

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