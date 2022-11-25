from pytorch_lightning import Trainer
from src.datamodules.feature_vector_multi_envs_dm import PL_FeatureVectorDS_FS
from src.models.classfier_feature_selection import ClassiferFeatureSubset_Simple
import numpy as np

def fit_classifier_fs(ds_train_features_it,
                        ds_train_label_it,
                        ds_val_features_it,
                        ds_val_label_it,
                        nb_of_features):
    dm_it = PL_FeatureVectorDS_FS(batch_size=64,
                                  num_workers=0,
                                  ds_train=[ds_train_features_it, ds_train_label_it],
                                  ds_val=[ds_val_features_it, ds_val_label_it])
    lm_it = ClassiferFeatureSubset_Simple(lr=0.0001,
                                          weight_decay=0.000001,
                                          z_size=nb_of_features)

    trainer = Trainer(max_epochs=50,
                      gpus=0,
                      weights_summary=None,
                      enable_progress_bar=True)

    trainer.fit(datamodule=dm_it, model=lm_it)

    test = trainer.test(datamodule=dm_it)[0]
    eval_metric = list(test.items())[0][-1]

    return eval_metric

class Datasets_Setup:

    def __init__(self,
                 ds_train_features_1,
                 ds_train_features_2,
                 ds_val_features_1,
                 ds_val_features_2,
                 feature_rank):
        self.ds_train_features_1 = ds_train_features_1
        self.ds_train_features_2 = ds_train_features_2
        self.ds_val_features_1 = ds_val_features_1
        self.ds_val_features_2 = ds_val_features_2
        self.feature_rank = feature_rank

    def new_datasets(self, nb_of_features):
        ds_train_features_1_it = self.ds_train_features_1[:, self.feature_rank[:nb_of_features]]
        ds_train_features_2_it = self.ds_train_features_2[:, self.feature_rank[:nb_of_features]]
        ds_train_features_it = np.concatenate([ds_train_features_1_it, ds_train_features_2_it])
        ds_val_features_1_it = self.ds_val_features_1[:, self.feature_rank[:nb_of_features]]
        ds_val_features_2_it = self.ds_val_features_2[:, self.feature_rank[:nb_of_features]]
        ds_val_features_it = np.concatenate([ds_val_features_1_it, ds_val_features_2_it], axis=0)

        return ds_train_features_it, ds_val_features_it