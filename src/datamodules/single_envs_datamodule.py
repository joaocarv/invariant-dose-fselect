from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning import LightningDataModule
from torchvision import transforms as transform_lib
import torchvision.transforms.functional as F
from torch.nn.functional import interpolate
from torchvision import transforms, datasets
from src.datamodules.components.single_envs import SingleEnv,SingleEnvCV
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PL_SingleEnv(LightningDataModule):

    def __init__(self, data_dir,
                 batch_size,
                 num_workers,
                 resize,
                 pin_memory,
                 extract_roi,
                 masked,
                 train_env,
                 val_env,
                 test_env,
                 val_split=0.1,
                 test_split=0.1
                 ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.pin_memory = pin_memory

        self.val_split = val_split
        self.test_split = test_split

        self.extract_roi = extract_roi
        self.resize = resize
        self.masked = masked

        self.train_env = train_env
        self.val_env = val_env
        self.test_env = test_env




    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # self.transform_train = A.Compose(
        #     [
        #         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        #         ToTensorV2
        #     ]
        # )
        # self.transform_val = A.Compose(
        #     [
        #         ToTensorV2
        #     ]
        # )
        self.transform_train = None
        self.transform_val = None
        # called on every GPU
        if stage == "fit" or stage is None:
            train_env_ds = SingleEnv(root_dir=self.data_dir,
                                     env=self.train_env,
                                     resize=self.resize,
                                     extract_roi=self.extract_roi,
                                     masked=self.masked)

            val_env_ds = SingleEnv(root_dir=self.data_dir,
                                   env=self.val_env,
                                   resize=self.resize,
                                   extract_roi=self.extract_roi,
                                   masked=self.masked)

            n_val = int(len(train_env_ds)*self.val_split)
            n_test = int(len(train_env_ds)*self.test_split)
            idx_train = list(range(len(train_env_ds)))
            idx_val = list(range(len(train_env_ds)))

            train_ds = Subset(train_env_ds, idx_train[n_val:-n_test])
            val_ds = Subset(val_env_ds, idx_val[:n_val])

            self.train_ds = DataSubSet(train_ds, transform=self.transform_val)
            self.val_ds = DataSubSet(val_ds, transform=self.transform_val)

        if stage  == "test" or stage is None:
            test_env_ds = SingleEnv(root_dir=self.data_dir,
                                     env=self.test_env,
                                     resize=self.resize,
                                     extract_roi=self.extract_roi,
                                     masked=self.masked)

            n_test = int(len(test_env_ds)*self.test_split)
            idx_test = list(range(len(test_env_ds)))

            test_ds = Subset(test_env_ds,idx_test[-n_test:])
            self.test_ds = DataSubSet(test_ds,transform=self.transform_val)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size= self.batch_size,
            pin_memory = False,
            shuffle = True,
            num_workers = self.num_workers
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            pin_memory=False,
            shuffle=False,
            num_workers=self.num_workers
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            pin_memory=False,
            shuffle=False,
            num_workers=self.num_workers
        )
        return test_loader

class PL_SingleEnvCV(LightningDataModule):

    def __init__(self, data_dir,
                 batch_size,
                 num_workers,
                 resize,
                 pin_memory,
                 extract_roi,
                 masked,
                 cv_fold,
                 train_env,
                 val_env,
                 test_env

                 ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.pin_memory = pin_memory

        self.extract_roi = extract_roi
        self.resize = resize
        self.masked = masked

        self.train_env = train_env
        self.val_env = val_env
        self.test_env = test_env

        self.cv_fold = cv_fold




    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # self.transform_train = A.Compose(
        #     [
        #         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        #         ToTensorV2
        #     ]
        # )
        # self.transform_val = A.Compose(
        #     [
        #         ToTensorV2
        #     ]
        # )
        self.transform_train = None
        self.transform_val = None
        # called on every GPU
        if stage == "fit" or stage is None:
            self.train_ds = SingleEnvCV(root_dir=self.data_dir,
                                       env=self.train_env,
                                       resize=self.resize,
                                       extract_roi=self.extract_roi,
                                       masked=self.masked,
                                       cv_fold = self.cv_fold,
                                       data_subset = 'train',
                                       transform = self.transform_train
                                       )

            self.val_ds = SingleEnvCV(root_dir=self.data_dir,
                                       env=self.val_env,
                                       resize=self.resize,
                                       extract_roi=self.extract_roi,
                                       masked=self.masked,
                                       cv_fold = self.cv_fold,
                                       data_subset = 'val',
                                       transform = self.transform_val
                                       )


        if stage  == "test" or stage is None:
            self.test_ds = SingleEnvCV(root_dir=self.data_dir,
                                      env=self.test_env,
                                      resize=self.resize,
                                      extract_roi=self.extract_roi,
                                      masked=self.masked,
                                      cv_fold=self.cv_fold,
                                      data_subset='test',
                                      transform=self.transform_val
                                      )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size= self.batch_size,
            pin_memory = False,
            shuffle = True,
            num_workers = self.num_workers
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            pin_memory=False,
            shuffle=False,
            num_workers=self.num_workers
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            pin_memory=False,
            shuffle=False,
            num_workers=self.num_workers
        )
        return test_loader


class Interpolate(object):
    """Reduces size of image"""

    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, image):
        image = interpolate(image.unsqueeze(0), size=(self.window_size, self.window_size))
        image = image.squeeze(0)

        return image


class DataSubSet(Dataset):
    '''
    Dataset wrapper to apply transforms separately to subsets
    '''
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        img, y = self.subset[i]
        if self.transform:
            img = self.transform(image=img)
        return img, y
