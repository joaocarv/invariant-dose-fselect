from src.models.erm import ERM
from src.datamodules.components.multi_envs import  MultiEnvCV
from src.datamodules.components.single_envs import  SingleEnvCV
from src.feature_selection.methods import FeatureSelection

import os
import torch
import pandas as pd
import numpy as np
import itertools

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule

@hydra.main(config_path="configs_fs/", config_name="feature_selection.yaml")
def main(config: DictConfig):

    print('---------> Feature selection started')

    # Init Lightning Classifier Datamodule
    dm_classifier: LightningDataModule = hydra.utils.instantiate(config.datamodule_class)

    # Init Dataset pairs
    ds_multi_env_train: MultiEnvCV = hydra.utils.instantiate(config.datamodule_pairs.dataset_train)
    ds_multi_env_val: MultiEnvCV = hydra.utils.instantiate(config.datamodule_pairs.dataset_val)

    print('Loading Model')
    # Setup names
    pd_ckpt = pd.read_csv(config.ckpt_csv_path)
    experiment_name = config.name

    # Load model
    ckpt_path = pd_ckpt[pd_ckpt['experiment_name'] == experiment_name]['ckpt_path'].iloc[0]
    print(ckpt_path)
    lm_model = ERM.load_from_checkpoint(ckpt_path)
    lm_model.eval()

    # Aggregate features extracted for the full dataset
    print('Setup dataset for feature selection')
    ds_train_features_1 = np.empty((0, 2048))
    ds_train_features_2 = np.empty((0, 2048))

    ds_train_pid = np.empty((0))
    ds_train_label = np.empty((0))
    for i, b in enumerate(ds_multi_env_train):
        # print(b)
        # print(torch.tensor(b[0])[None,:].shape)
        f1 = lm_model.encoder(torch.tensor(b[0])[None,:]).detach().numpy()
        f2 = lm_model.encoder(torch.tensor(b[1])[None,:]).detach().numpy()

        ds_train_features_1 = np.concatenate((ds_train_features_1, f1))
        ds_train_features_2 = np.concatenate((ds_train_features_2, f2))
        ds_train_pid = np.concatenate((ds_train_pid,np.array(b[3])[None]))
        ds_train_label = np.concatenate((ds_train_label, np.array(b[2])[None]))
        # np.save(os.path.join(path_ds,'p'+pat_id+'_'+str(1)+'_'+str(i)+'.npy'), img1)
        # np.save(os.path.join(path_ds,'p'+pat_id+'_'+str(2)+'_'+str(i)+'.npy'), img2)

    ds_val_features_1 = np.empty((0, 2048))
    ds_val_features_2 = np.empty((0, 2048))
    ds_val_pid = np.empty((0))
    ds_val_label = np.empty((0))

    for i, b in enumerate(ds_multi_env_val):
        f1 = lm_model.encoder(torch.tensor(b[0])[None,:]).detach().numpy()
        f2 = lm_model.encoder(torch.tensor(b[1])[None,:]).detach().numpy()
        ds_val_features_1 = np.concatenate((ds_val_features_1, f1))
        ds_val_features_2 = np.concatenate((ds_val_features_2, f2))
        ds_val_pid = np.concatenate((ds_val_pid,np.array(b[3])[None]))
        ds_val_label = np.concatenate((ds_val_label, np.array(b[2])[None]))
        # np.save(os.path.join(path_ds,'p'+pat_id+'_'+str(1)+'_'+str(i)+'.npy'), img1)
        # np.save(os.path.join(path_ds,'p'+pat_id+'_'+str(2)+'_'+str(i)+'.npy'), img2)

    # Select Features
    print('Select features')

    algos = ['random','pc','ks','kl','epa']

    path_ds = '/cluster/scratch/jcarvalho/multi-dose-fselect/logs/features/' + experiment_name
    os.makedirs(path_ds, exist_ok=True)
    feature_selection = FeatureSelection(
        ds_train_features_1=ds_train_features_1,
        ds_train_features_2=ds_train_features_2,
        ds_train_label=ds_train_label,
        ds_val_features_1=ds_val_features_1,
        ds_val_features_2=ds_val_features_2,
        ds_val_label=ds_val_label,
        nb_of_features_increment=50,
        max_features = lm_model.encoder.nb_features
    )

    pairs_features_paths =[]
    for sim_func in algos:
        selected_features = feature_selection._select_features(
            sim_func=sim_func
        )
        sf_dir = os.path.join(path_ds, sim_func)
        os.makedirs(sf_dir, exist_ok=True)

        sf_path = os.path.join(sf_dir, 'selected_features.npy')
        np.save(sf_path,
                np.array(selected_features))

        np.save(sf_path,
                np.array(selected_features))

        pairs_features_paths.append([sf_dir,selected_features])


    # for fs_algo,nb_features in itertools.product(algos,nb_features_l):
    #
    #     selected_features = feature_selection.selected_features(fs_type=fs_algo,
    #                                         nb_features=nb_features)
    #     sf_dir = os.path.join(path_ds, fs_algo+'_'+str(nb_features))
    #     os.makedirs(sf_dir,exist_ok=True)
    #
    #     sf_path = os.path.join(sf_dir, 'selected_features.npy')
    #     np.save(sf_path,
    #             np.array(selected_features))
    #
    #     pairs_features_paths.append([sf_dir,selected_features])

    # Prepare Classifier Datasets
    print('Setup selected features dataset')
    dm_classifier.prepare_data()
    dm_classifier.setup()
    #
    # path_ds = '/cluster/scratch/jcarvalho/multi-dose-fselect/logs/features/'+experiment_name+'/train/'
    # os.makedirs(path_ds,exist_ok=True)
    for i, b in enumerate(dm_classifier.train_dataloader()):
        # Extract feature vector and target
        features = lm_model.encoder(b[0])
        target = b[1].detach().numpy()[0]

        for path_ds,selected_features in pairs_features_paths:
            path_ds = os.path.join(path_ds,'train')
            os.makedirs(path_ds,exist_ok=True)
            np.save(os.path.join(path_ds, str(i)+'.npy'),
                    np.array((features[:,selected_features].detach().numpy(),target),dtype=object))



    # path_ds = '/cluster/scratch/jcarvalho/multi-dose-fselect/logs/features/'+experiment_name+'/val'
    # os.makedirs(path_ds,exist_ok=True)
    for i, b in enumerate(dm_classifier.val_dataloader()):
        # Extract feature vector and target
        features = lm_model.encoder(b[0])
        target = b[1].detach().numpy()[0]

        for path_ds, selected_features in pairs_features_paths:
            path_ds = os.path.join(path_ds, 'val')
            os.makedirs(path_ds, exist_ok=True)
            np.save(os.path.join(path_ds, str(i) + '.npy'),
                    np.array((features[:, selected_features].detach().numpy(), target), dtype=object))

    # path_ds = '/cluster/scratch/jcarvalho/multi-dose-fselect/logs/features/'+experiment_name+'/test'
    # os.makedirs(path_ds,exist_ok=True)
    for i, b in enumerate(dm_classifier.test_dataloader()):
        # Extract feature vector and target
        features = lm_model.encoder(b[0])
        target = b[1].detach().numpy()[0]

        for path_ds, selected_features in pairs_features_paths:
            path_ds = os.path.join(path_ds, 'test')
            os.makedirs(path_ds, exist_ok=True)
            np.save(os.path.join(path_ds, str(i) + '.npy'),
                    np.array((features[:, selected_features].detach().numpy(), target), dtype=object))

    print('---------> Feature selection finished')


if __name__ == "__main__":
    main()


