from torch.utils.data import Dataset
import glob
import numpy as np

class trainDatasets(Dataset):

    def __init__(self,datasets,three_d,label,transforms=None):
        self.datasets = np.load(datasets)
        self.three_d = np.load(three_d)
        self.label = np.load(label)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self,idx):
        traindata = self.datasets[idx]
        label = self.label[idx]
        three_d = self.three_d[idx]
        data = (traindata,label,three_d)
        return data

class testDatasets(Dataset):

    def __init__(self,datasets,three_d,transforms=None):
        self.datasets = np.load(datasets)
        self.transforms = transforms
        self.three_d = np.load(three_d)
        
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self,idx):
        testdata = self.datasets[idx]
        three_d = self.three_d[idx]
        data = (testdata,three_d)
        return data
