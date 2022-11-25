import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from scipy.ndimage.measurements import center_of_mass
from random import sample, seed
seed(1234)

class MultiEnv(Dataset):

    def __init__(self,
                 root_dir,
                 resize,
                 env1,
                 env2,
                 extract_roi,
                 transform=None,
                 masked=True,
                 shiftvar=None): # considered_class 1,2 or 3 (see annotations.csv)

        self.rootdir = root_dir
        self.transform = transform
        self.env1 = env1
        self.env2 = env2
        self.resize = resize
        self.extract_roi = extract_roi
        self.masked = masked

        self.env_path_1 = os.path.join(root_dir,env1)
        self.env_path_2 = os.path.join(root_dir,env2)
        self.seg_path = os.path.join(root_dir,'Seg')

        self.img_list_1 = os.listdir(self.env_path_1)
        self.img_list_1 = sorted(self.img_list_1)
        self.img_list_2 = os.listdir(self.env_path_1)
        self.img_list_2 = sorted(self.img_list_2)
        self.img_list_1 = [file_name.split('.')[0] for file_name in self.img_list_1]
        self.img_list_2 = [file_name.split('.')[0] for file_name in self.img_list_2]
        self.annotations = pd.read_excel(os.path.join(root_dir,'annotations_final.xlsx'))

        if shiftvar:
            patients_selected = [int(slice_path.split('_' + env1)[0][8:]) for slice_path in self.img_list_1]
            patients_selected = sorted(set(patients_selected))
            labels = [self.annotations[self.annotations["New ID"] == pid]['ILD_20'].values[0] for pid in
                         patients_selected]
            patients_0 = [patients_selected[i] for i, label in enumerate(labels) if label == 0]
            patients_1 = [patients_selected[i] for i, label in enumerate(labels) if label == 1]
            patients_2 = [patients_selected[i] for i, label in enumerate(labels) if label == 2]
            patients_per_label = [len(patients_0), len(patients_2), len(patients_2)]
            samp0, samp1, samp2 = shiftvar2values(shiftvar, patients_per_label)
            # print(len(patients_0),len(patients_2),len(patients_2))
            # print(samp0, samp1, samp2)
            final_patients_selected = sample(patients_0, samp0) + sample(patients_1, samp1) + sample(patients_2, samp2)
            self.img_list_1 = [file for file in self.img_list_1 if
                             int(file.split('_' + env1)[0][8:]) in final_patients_selected]
            self.img_list_2 = [file for file in self.img_list_2 if
                               int(file.split('_' + env1)[0][8:]) in final_patients_selected]

    def __len__(self):
        return len(self.img_list_1)

    def __getitem__(self, item):
        img_name_1 = self.img_list_1[item]
        img_name_2 = self.img_list_2[item]
        img_path_1 = os.path.join(self.env_path_1, img_name_1 + '.nii.gz')
        img_path_2 = os.path.join(self.env_path_2, img_name_2 + '.nii.gz')
        msk_path = os.path.join(self.seg_path, img_name_1.replace(self.env1, 'Seg') + '.nii.gz')
        patient_id = int(img_name_1.split('_')[1])

        # Read Image as np
        msk = sitk.ReadImage(msk_path)
        msk_np = sitk.GetArrayFromImage(msk)
        msk_np = np.squeeze(msk_np)
        img_1 = sitk.ReadImage(img_path_1)
        img_2 = sitk.ReadImage(img_path_2)
        img_np_1 = sitk.GetArrayFromImage(img_1).astype('float32')
        img_np_norm_clip_1 = np.squeeze(img_np_1)
        img_np_2 = sitk.GetArrayFromImage(img_2).astype('float32')
        img_np_norm_clip_2 = np.squeeze(img_np_2)

        if self.extract_roi:
            img_np_norm_clip_1,_ = extract_ROI(img_np_norm_clip_1, msk_np)
            img_np_norm_clip_2,msk_np = extract_ROI(img_np_norm_clip_2, msk_np)

        if self.resize < 128:
            msk_np = resize(msk_np, (self.resize, self.resize), anti_aliasing=True)
            msk_np[msk_np == 1] = 0.5
            msk_np[msk_np != 0] = 1
            img_np_norm_clip_1 = resize(img_np_norm_clip_1, (self.resize, self.resize), anti_aliasing=True)
            img_np_norm_clip_2 = resize(img_np_norm_clip_2, (self.resize, self.resize), anti_aliasing=True)


        # img normalization and clipping
        img_np_norm_clip_1[img_np_norm_clip_1 < -1000] = -1000
        img_np_norm_clip_1[img_np_norm_clip_1 > 1000] = 1000
        img_np_norm_clip_1 = (img_np_norm_clip_1 + 1000) / 2000.
        img_np_norm_clip_2[img_np_norm_clip_2 < -1000] = -1000
        img_np_norm_clip_2[img_np_norm_clip_2 > 1000] = 1000
        img_np_norm_clip_2 = (img_np_norm_clip_2 + 1000) / 2000.

        # Apply mask
        if self.masked is True:
            img_np_norm_clip_1 = img_np_norm_clip_1*msk_np
            img_np_norm_clip_2 = img_np_norm_clip_2*msk_np

        #Transform into multi-channel
        img_np_norm_clip_1 = np.array([img_np_norm_clip_1,
                            img_np_norm_clip_1,
                            img_np_norm_clip_1])
        img_np_norm_clip_2 = np.array([img_np_norm_clip_2,
                            img_np_norm_clip_2,
                            img_np_norm_clip_2])

        if self.transform is not None:
            image_1 = self.transform(img_np_norm_clip_1)
            image_2 = self.transform(img_np_norm_clip_2)
        else:
            image_1 = img_np_norm_clip_1
            image_2 = img_np_norm_clip_2


        # Labels
        label = self.annotations[self.annotations["New ID"] == patient_id]['ILD_20'].values[0]

        sample = [image_1,image_2,label, patient_id]
        return sample


class MultiEnvCV(Dataset):

    def __init__(self,
                 root_dir,
                 env1,
                 env2,
                 resize,
                 extract_roi,
                 cv_fold,
                 data_subset,
                 transform=None,
                 masked=True,
                 shiftvar=None): # considered_class 1,2 or 3 (see annotations.csv)

        self.rootdir = root_dir
        self.transform = transform
        self.env1 = env1
        self.env2 = env2
        self.resize = resize
        self.extract_roi = extract_roi
        self.masked = masked

        self.env_path_1 = os.path.join(root_dir, env1)
        self.env_path_2 = os.path.join(root_dir, env2)
        self.seg_path = os.path.join(root_dir,'Seg')

        df_files = pd.read_csv(root_dir+'/cv/'+str(cv_fold) +'/'+data_subset+'.csv')
        self.img_list_1 = [file.replace('DOSE', env1) for file in df_files['filename'].tolist()]
        self.img_list_1 = [file.split('.')[0] for file in self.img_list_1]
        self.img_list_2 = [file.replace('DOSE', env2) for file in df_files['filename'].tolist()]
        self.img_list_2 = [file.split('.')[0] for file in self.img_list_2]
        self.annotations = pd.read_excel(os.path.join(root_dir,'annotations_final.xlsx'))

        if shiftvar:
            patients_selected_cv = [int(slice_path.split('_'+env1)[0][8:]) for slice_path in self.img_list_1]
            patients_selected_cv = sorted(set(patients_selected_cv))
            labels_cv = [self.annotations[self.annotations["New ID"] == pid]['ILD_20'].values[0] for pid in patients_selected_cv]
            patients_0 = [patients_selected_cv[i] for i, label in enumerate(labels_cv) if label == 0]
            patients_1 = [patients_selected_cv[i] for i, label in enumerate(labels_cv) if label == 1]
            patients_2 = [patients_selected_cv[i] for i, label in enumerate(labels_cv) if label == 2]
            patients_per_label = [len(patients_0),len(patients_2),len(patients_2)]
            samp0,samp1,samp2 = shiftvar2values(shiftvar, patients_per_label)
            # print(len(patients_0),len(patients_2),len(patients_2))
            # print(samp0, samp1, samp2)
            final_patients_selected = sample(patients_0,samp0)+sample(patients_1,samp1)+sample(patients_2,samp2)
            self.img_list_1 = [file for file in self.img_list_1 if
                               int(file.split('_'+env1)[0][8:]) in final_patients_selected]
            self.img_list_2 = [file for file in self.img_list_2 if
                               int(file.split('_' + env1)[0][8:]) in final_patients_selected]


    def __len__(self):
        return len(self.img_list_1)

    def __getitem__(self, item):
        img_name_1 = self.img_list_1[item]
        img_name_2 = self.img_list_2[item]
        img_path_1 = os.path.join(self.env_path_1, img_name_1 + '.nii.gz')
        img_path_2 = os.path.join(self.env_path_2, img_name_2 + '.nii.gz')
        msk_path = os.path.join(self.seg_path, img_name_1.replace(self.env1, 'Seg') + '.nii.gz')
        patient_id = int(img_name_1.split('_')[1])

        # Read Image as np
        msk = sitk.ReadImage(msk_path)
        msk_np = sitk.GetArrayFromImage(msk)
        msk_np = np.squeeze(msk_np)
        img_1 = sitk.ReadImage(img_path_1)
        img_2 = sitk.ReadImage(img_path_2)
        img_np_1 = sitk.GetArrayFromImage(img_1).astype('float32')
        img_np_norm_clip_1 = np.squeeze(img_np_1)
        img_np_2 = sitk.GetArrayFromImage(img_2).astype('float32')
        img_np_norm_clip_2 = np.squeeze(img_np_2)

        if self.extract_roi:
            img_np_norm_clip_1, _ = extract_ROI(img_np_norm_clip_1, msk_np)
            img_np_norm_clip_2, msk_np = extract_ROI(img_np_norm_clip_2, msk_np)
        if self.resize < 128:
            msk_np = resize(msk_np, (self.resize, self.resize), anti_aliasing=True)
            msk_np[msk_np == 1] = 0.5
            msk_np[msk_np != 0] = 1

            img_np_norm_clip_1 = resize(img_np_norm_clip_1, (self.resize, self.resize), anti_aliasing=True)
            img_np_norm_clip_2 = resize(img_np_norm_clip_2, (self.resize, self.resize), anti_aliasing=True)

        # img normalization and clipping
        img_np_norm_clip_1[img_np_norm_clip_1 < -1000] = -1000
        img_np_norm_clip_1[img_np_norm_clip_1 > 1000] = 1000
        img_np_norm_clip_1 = (img_np_norm_clip_1 + 1000) / 2000.

        img_np_norm_clip_2[img_np_norm_clip_2 < -1000] = -1000
        img_np_norm_clip_2[img_np_norm_clip_2 > 1000] = 1000
        img_np_norm_clip_2 = (img_np_norm_clip_2 + 1000) / 2000.

        # Apply mask
        if self.masked is True:
            img_np_norm_clip_1 = img_np_norm_clip_1 * msk_np
            img_np_norm_clip_2 = img_np_norm_clip_2 * msk_np

        # Transform into multi-channel
        img_np_norm_clip_1 = np.array([img_np_norm_clip_1,
                                       img_np_norm_clip_1,
                                       img_np_norm_clip_1])
        img_np_norm_clip_2 = np.array([img_np_norm_clip_2,
                                       img_np_norm_clip_2,
                                       img_np_norm_clip_2])

        if self.transform is not None:
            image_1 = self.transform(img_np_norm_clip_1)
            image_2 = self.transform(img_np_norm_clip_2)
        else:
            image_1 = img_np_norm_clip_1
            image_2 = img_np_norm_clip_2

        # Labels
        label = self.annotations[self.annotations["New ID"] == patient_id]['ILD_20'].values[0]

        sample = [image_1, image_2, label, patient_id]
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
