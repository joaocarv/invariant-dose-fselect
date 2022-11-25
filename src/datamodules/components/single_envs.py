import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from scipy.ndimage.measurements import center_of_mass
from random import sample, seed
seed(1234)

class SingleEnv(Dataset):

    def __init__(self,
                 root_dir,
                 env,
                 resize,
                 extract_roi,
                 transform=None,
                 masked=True,
                 shiftvar=False): # considered_class 1,2 or 3 (see annotations.csv)

        self.rootdir = root_dir
        self.transform = transform
        self.env = env
        self.resize = resize
        self.extract_roi = extract_roi
        self.masked = masked
        self.env_path = os.path.join(root_dir,env)
        self.seg_path = os.path.join(root_dir,'Seg')
        self.img_list = os.listdir(self.env_path)
        self.img_list = sorted(self.img_list)
        self.img_list = [file_name.split('.')[0] for file_name in self.img_list]
        self.annotations = pd.read_excel(os.path.join(root_dir,'annotations_final.xlsx'))

        if shiftvar:
            patients_selected = [int(slice_path.split('_' + env)[0][8:]) for slice_path in self.img_list]
            patients_selected = sorted(set(patients_selected))
            labels = [self.annotations[self.annotations["New ID"] == pid]['ILD_20'].values[0] for pid in
                         patients_selected]
            patients_0 = [patients_selected[i] for i, label in enumerate(labels) if label == 0]
            patients_1 = [patients_selected[i] for i, label in enumerate(labels) if label == 1]
            patients_2 = [patients_selected[i] for i, label in enumerate(labels) if label == 2]
            patients_per_label = [len(patients_0), len(patients_2), len(patients_2)]
            samp0, samp1, samp2 = shiftvar2values(shiftvar, patients_per_label)
            final_patients_selected = sample(patients_0, samp0) + sample(patients_1, samp1) + sample(patients_2, samp2)
            self.img_list = [file for file in self.img_list if
                             int(file.split('_' + env)[0][8:]) in final_patients_selected]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_name = self.img_list[item]
        img_path = os.path.join(self.env_path, img_name + '.nii.gz')
        msk_path = os.path.join(self.seg_path, img_name.replace(self.env, 'Seg') + '.nii.gz')
        patient_id = int(img_name.split('_')[1])

        # Read Image as np
        msk = sitk.ReadImage(msk_path)
        msk_np = sitk.GetArrayFromImage(msk)
        msk_np = np.squeeze(msk_np)
        img = sitk.ReadImage(img_path)
        img_np = sitk.GetArrayFromImage(img).astype('float32')
        img_np_norm_clip = np.squeeze(img_np)

        if self.extract_roi:
            img_np_norm_clip,msk_np = extract_ROI(img_np_norm_clip, msk_np)

        if self.resize < 128:
            msk_np = resize(msk_np, (self.resize, self.resize), anti_aliasing=True)
            msk_np[msk_np == 1] = 0.5
            msk_np[msk_np != 0] = 1
            img_np_norm_clip = resize(img_np_norm_clip, (self.resize, self.resize), anti_aliasing=True)


        # img normalization and clipping
        img_np_norm_clip[img_np_norm_clip < -1000] = -1000
        img_np_norm_clip[img_np_norm_clip > 1000] = 1000
        img_np_norm_clip = (img_np_norm_clip + 1000) / 2000.

        # Apply mask
        if self.masked is True:
            img_np_norm_clip = img_np_norm_clip*msk_np

        #Transform into multi-channel
        img_np_norm_clip = np.array([img_np_norm_clip,
                            img_np_norm_clip,
                            img_np_norm_clip])

        if self.transform is not None:
            image = self.transform(img_np_norm_clip)
        else:
            image = img_np_norm_clip


        # Labels
        label = self.annotations[self.annotations["New ID"] == patient_id]['ILD_20'].values[0]

        sample = [image, label, patient_id]
        return sample


class SingleEnvCV(Dataset):

    def __init__(self,
                 root_dir,
                 env,
                 resize,
                 extract_roi,
                 cv_fold,
                 data_subset,
                 transform=None,
                 masked=True,
                 shiftvar=None): # considered_class 1,2 or 3 (see annotations.csv)

        self.rootdir = root_dir
        self.transform = transform
        self.env = env
        self.resize = resize
        self.extract_roi = extract_roi
        self.masked = masked

        self.env_path = os.path.join(root_dir,env)
        self.seg_path = os.path.join(root_dir,'Seg')

        df_files = pd.read_csv(root_dir+'/cv/'+str(cv_fold) +'/'+data_subset+'.csv')
        self.img_list = [file.replace('DOSE', env) for file in df_files['filename'].tolist()]
        self.img_list = [file.split('.')[0] for file in self.img_list]
        self.annotations = pd.read_excel(os.path.join(root_dir,'annotations_final.xlsx'))

        if shiftvar:
            patients_selected_cv = [int(slice_path.split('_'+env)[0][8:]) for slice_path in self.img_list]
            patients_selected_cv = sorted(set(patients_selected_cv))
            labels_cv = [self.annotations[self.annotations["New ID"] == pid]['ILD_20'].values[0] for pid in patients_selected_cv]
            patients_0 = [patients_selected_cv[i] for i, label in enumerate(labels_cv) if label == 0]
            patients_1 = [patients_selected_cv[i] for i, label in enumerate(labels_cv) if label == 1]
            patients_2 = [patients_selected_cv[i] for i, label in enumerate(labels_cv) if label == 2]
            patients_per_label = [len(patients_0),len(patients_2),len(patients_2)]

            final_patients_selected = sample(patients_0,samp0)+sample(patients_1,samp1)+sample(patients_2,samp2)
            self.img_list = [file for file in self.img_list if int(file.split('_'+env)[0][8:]) in final_patients_selected]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_name = self.img_list[item]
        img_path = os.path.join(self.env_path, img_name + '.nii.gz')
        msk_path = os.path.join(self.seg_path, img_name.replace(self.env, 'Seg') + '.nii.gz')
        patient_id = int(img_name.split('_')[1])

        # Read Image as np
        msk = sitk.ReadImage(msk_path)
        msk_np = sitk.GetArrayFromImage(msk)
        msk_np = np.squeeze(msk_np)

        img = sitk.ReadImage(img_path)
        img_np = sitk.GetArrayFromImage(img).astype('float32')
        img_np_norm_clip = np.squeeze(img_np)

        if self.extract_roi:
            img_np_norm_clip,msk_np = extract_ROI(img_np_norm_clip, msk_np)

        if self.resize < 128:
            msk_np = resize(msk_np, (self.resize, self.resize), anti_aliasing=True)
            msk_np[msk_np == 1] = 0.5
            msk_np[msk_np != 0] = 1
            img_np_norm_clip = resize(img_np_norm_clip, (self.resize, self.resize), anti_aliasing=True)


        # img normalization and clipping
        img_np_norm_clip[img_np_norm_clip < -1000] = -1000
        img_np_norm_clip[img_np_norm_clip > 1000] = 1000
        img_np_norm_clip = (img_np_norm_clip + 1000) / 2000.

        # Apply mask
        if self.masked is True:
            img_np_norm_clip = img_np_norm_clip*msk_np

        #Transform into multi-channel
        img_np_norm_clip = np.array([img_np_norm_clip,
                            img_np_norm_clip,
                            img_np_norm_clip])

        if self.transform is not None:
            image = self.transform(img_np_norm_clip)
        else:
            image = img_np_norm_clip


        # Labels
        label = self.annotations[self.annotations["New ID"] == patient_id]['ILD_20'].values[0]

        sample = [image, label]
        return sample


def extract_ROI(image, mask, window_size=128):

    # Get mask centroid
    c1, c2 = center_of_mass(mask)
    c1, c2 = int(c1), int(c2)

    # get bbox location
    bbox_dim = return_window_size(c1,
                                  c2,
                                  window_size,
                                  mask.shape[0])

    # Reshape image according to window_size
    mask = mask[c1 - int(bbox_dim[0]):c1 + int(bbox_dim[1]),
            c2 - int(bbox_dim[2]):c2 + int(bbox_dim[3])]
    image = image[c1 - int(bbox_dim[0]):c1 + int(bbox_dim[1]),
            c2 - int(bbox_dim[2]):c2 + int(bbox_dim[3])]

    return image,mask


def return_window_size(c1, c2, window_size, img_size):
    # For dimension x
    cond_1 = (img_size - 1) - c1 < window_size / 2
    cond_2 = c1 - window_size / 2 < 0
    if cond_1 or cond_2:
        if cond_1:
            wsr1 = img_size - 1 - c1
            wsl1 = window_size - wsr1
        elif cond_2:
            wsl1 = c1
            wsr1 = window_size - wsl1
    else:
        wsr1 = int(window_size / 2)
        wsl1 = int(window_size / 2)

    # For dimension y
    cond_1 = (img_size - 1) - c2 < window_size / 2
    cond_2 = c2 - window_size / 2 < 0
    if cond_1 or cond_2:
        if cond_1:
            wsd2 = img_size - 1 - c2
            wsu2 = window_size - wsd2
        elif cond_2:
            wsu2 = c2
            wsd2 = window_size - wsu2
    else:
        wsd2 = int(window_size / 2)
        wsu2 = int(window_size / 2)

    return [wsl1, wsr1, wsu2, wsd2]


def shiftvar2values(shiftvar,patients_per_label):
    # Original number of samples per class 75(0),60(1),58(2)
    if shiftvar == 1:
        # [3/4,1/4,0]
        samples_per_class = [patients_per_label[0],
                             int(patients_per_label[0]/3),
                             0]
    elif shiftvar == 2:
        # [1/2,1/3,1/6]
        samples_per_class = [patients_per_label[0],
                             int(patients_per_label[0]*2/3),
                             int(patients_per_label[0]*2/6)]
    elif shiftvar == 3:
        # [1/3,1/3,1/3]
        samples_per_class = [min(patients_per_label),
                             min(patients_per_label),
                             min(patients_per_label)]
    elif shiftvar == 4:
        # [1/6,1/3,1/2]
        samples_per_class = [int(patients_per_label[2]*2/6),
                             int(patients_per_label[2]*2/3),
                             patients_per_label[2]]
    elif shiftvar == 5:
        # [0,1/4,3/4]
        samples_per_class = [0,int(patients_per_label[2]/3),patients_per_label[2]]
    else:
        raise(Exception("shiftvar undefined"))
    return samples_per_class
