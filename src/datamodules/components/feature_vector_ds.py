import os
import numpy as np

import torch
from torch.utils.data import Dataset
from random import sample, seed
seed(1234)

class FeatureVectorDS(Dataset):

    def __init__(self,
                 root_dir,
                 data_subset):

        ds_path = os.path.join(root_dir,data_subset)
        self.fv_list =[os.path.join(ds_path,fv_file) for fv_file in os.listdir(ds_path)]

    def __len__(self):
        return len(self.fv_list)

    def __getitem__(self, item):
        item_path = self.fv_list[item]
        fv,label =  np.load(item_path, allow_pickle=True)
        return [fv,label]


class FeatureVectorDS_FS(Dataset):

    def __init__(self,
                 ds):
        self.fv = ds[0]
        self.label = ds[1]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        fv = self.fv[item]
        label = self.label[item]
        return [torch.tensor(fv, dtype=torch.float32), torch.tensor(label, dtype=torch.long)]
