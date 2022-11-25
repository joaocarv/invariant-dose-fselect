from src.datamodules.components.feature_vector_ds import FeatureVectorDS, FeatureVectorDS_FS
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

class PL_FeatureVector(LightningDataModule):
    def __init__(self,
                 data_dir,
                 batch_size,
                 num_workers,):
        super().__init__()


        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir

    def prepare_data(self):
        pass

    def setup(self,stage=None):

        if stage =='fit' or stage is None:
            self.train_ds = FeatureVectorDS(root_dir=self.data_dir,
                                     data_subset='train')

            self.val_ds = FeatureVectorDS(root_dir=self.data_dir,
                                        data_subset='val')

        if stage =='test' or stage is None:
            self.test_ds = FeatureVectorDS(root_dir=self.data_dir,
                                           data_subset='test')

    def train_dataloader(self):
        train_datalodar =DataLoader(self.train_ds,
                                    batch_size = self.batch_size,
                                    pin_memory=False,
                                    shuffle=True,
                                    num_workers=self.num_workers)

        return train_datalodar

    def val_dataloader(self):
        val_datalodar = DataLoader(self.val_ds,
                                     batch_size=self.batch_size,
                                     pin_memory=False,
                                     shuffle=True,
                                     num_workers=self.num_workers)

        return val_datalodar

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_ds,
                                     batch_size=self.batch_size,
                                     pin_memory=False,
                                     shuffle=True,
                                     num_workers=self.num_workers)

        return test_dataloader


class PL_FeatureVectorDS_FS(LightningDataModule):
    def __init__(self,
                 batch_size,
                 num_workers,
                 ds_train,
                 ds_val):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ds_train = ds_train
        self.ds_val = ds_val

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            self.train_ds = FeatureVectorDS_FS(ds=self.ds_train)

            self.val_ds = FeatureVectorDS_FS(ds=self.ds_val)

        if stage == 'test' or stage is None:
            self.test_ds = FeatureVectorDS_FS(ds=self.ds_val)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_ds,
                                      batch_size=self.batch_size,
                                      pin_memory=False,
                                      shuffle=True,
                                      num_workers=self.num_workers)

        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_ds,
                                    batch_size=self.batch_size,
                                    pin_memory=False,
                                    shuffle=True,
                                    num_workers=self.num_workers)

        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.val_ds,
                                     batch_size=self.batch_size,
                                     pin_memory=False,
                                     shuffle=True,
                                     num_workers=self.num_workers)

        return test_dataloader