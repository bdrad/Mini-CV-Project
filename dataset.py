import numpy as np
import os
from torch.utils.data import Dataset

"""Breast density dataset"""
class CBISDataset(Dataset):
    #Uses train dataset csv by default. Can createa test dataset by passing in path.
    def __init__(self, data_type="Training", transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                                         at retrieval time
        """
        self.data_type = data_type
        self.load_data()
        self.transform = transform
        self.index = 0
    
    #Method for finding the dataset files and loading them in memory
    def load_data(self):
        data_list = []
        path = "/Users/alan/Documents/bdrad/images/CBIS-DDSM"
        for subdir, dirs, files in os.walk(path):
            for f in files:
                filepath = subdir + os.sep + f
                if filepath.endswith(".npz") and self.data_type in filepath:
                    data = np.load(filepath)
                    data_list.append(data)
        self.data_list = data_list


    #Returns the ith row o dataframe
    def __getitem__(self, idx):
        x, y= self.data_list[self.index]["x"], self.data_list[self.index]["y"]
        #Perform transforma/normalization to fit what resnet pretrained
        #model was trained on
        if self.transform is not None:
            img1 = self.transform(img1)
        
        y_label = [[0,0]]
        if y[0] == 1:
            y_label[0][0] = 1
        else:
            y_label[0][1] = 1
        y = np.array(y_label)
        y = y.astype(np.float32)
        x = x.astype(np.float32)
        x = np.reshape(x, [1,x.shape[0], x.shape[1]])
        self.index += 1
        return (x, y)

    def __len__(self):
        #must return the length of this dataset
        return len(self.data_list)
