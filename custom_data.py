import numpy as np
import torch

class CustomVoc:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform 

        with open("pascalvoc.txt", "r") as f:
            labels_list = f.readlines()
            self.labels_list = [i.replace("\n","") for i in labels_list]
        
        self.labels_dir = {k: v for v,k in enumerate(self.labels_list, 1)}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, ann = self.dataset[idx]
        
        objs = ann['annotation']['object']
        boxes = []
        labels = []
        for obj in objs:
            xmin = int(obj["bndbox"]["xmin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymin = int(obj["bndbox"]["ymin"])
            ymax = int(obj["bndbox"]["ymax"])
            label = obj["name"]

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.labels_dir[label])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        # if self.transforms:
        #     img = self.transforms("image")

            # trans = 
        return img, target


        
