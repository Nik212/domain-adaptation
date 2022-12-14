from dataset import Dataset
import pandas as pd
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
import torchvision.transforms as T
import torch



class WeatherDatamodule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.root_train = config.root_train
        self.root_val = config.root_val
        self.batch_size = config.batch_size
        self.climate = config.climate
                

    def setup(self, stage: str):
        self.train = Dataset(self.root_train, self.climate)
        self.val = Dataset(self.root_val, self.climate)
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False,  drop_last=True)