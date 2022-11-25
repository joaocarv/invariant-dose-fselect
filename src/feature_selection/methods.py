from scipy.stats import pearsonr, kstest
from scipy.special import rel_entr
import numpy as np
import heapq
from random import randint,sample
from scipy.stats import gaussian_kde

from src.feature_selection.helpers import  Datasets_Setup, fit_classifier_fs


def pc(f_env1,f_env2):
    return np.array([pearsonr(v1,v2)[0] for v1,v2 in zip(f_env1,f_env2)])

def ks(f_env1,f_env2):
    return np.array([kstest(v1,v2)[0] for v1,v2 in zip(f_env1,f_env2)])

def kl(f_env1,f_env2):
    return np.array([rel_entr(v1,v2)[0] for v1,v2 in zip(f_env1,f_env2)])

def epa(f_env1,f_env2):
    return np.array([gaussian_kde(v1, bw_method='silverman').evaluate(v2).sum() for v1,v2 in zip(f_env1,f_env2)])

def rank_simfunc_index(similarity_vector):
    return heapq.nlargest(len(similarity_vector), range(len(similarity_vector)), similarity_vector.__getitem__)

def compute_feature_weights(sim_func,
                            ds_features_1,
                            ds_features_2,
                            ds_label
                            ):
    feature_rank = []
    for i in np.unique(ds_label):

        f_env1 = ds_features_1[ds_label == np.unique(ds_label)[int(i)]].transpose()
        f_env2 = ds_features_2[ds_label == np.unique(ds_label)[int(i)]].transpose()

        if sim_func == 'epa':
            weights = epa(f_env1, f_env2)
        elif sim_func == 'pc':
            weights = pc(f_env1, f_env2)
        elif sim_func == 'ks':
            weights = ks(f_env1, f_env2)
        elif sim_func == 'kl':
            weights = kl(f_env1, f_env2)
        feature_rank.append(rank_simfunc_index(weights))

    feature_rank = np.array(feature_rank).mean(axis=0) / len(np.unique(ds_label))

    feature_rank = rank_simfunc_index(feature_rank)
    return feature_rank

class FeatureSelection:

    def __init__(self,
                 ds_train_features_1,
                 ds_train_features_2,
                 ds_train_label,
                 ds_val_features_1,
                 ds_val_features_2,
                 ds_val_label,
                 nb_of_features_increment,
                 max_features):

        self.ds_train_features_1 = ds_train_features_1
        self.ds_train_features_2 = ds_train_features_2
        self.ds_train_label = np.concatenate([ds_train_label,ds_train_label])
        self.ds_val_features_1 = ds_val_features_1
        self.ds_val_features_2 = ds_val_features_2
        self.ds_val_label = np.concatenate([ds_val_label,ds_val_label])

        self.nb_of_features_increment = nb_of_features_increment
        self.max_features = max_features


    def _select_features(self,
                         sim_func):

        if sim_func == 'random':
            selected_features = sample(range(0,2048), randint(0,2048)  )
            selected_features.sort()
            return selected_features


        feature_rank = compute_feature_weights(sim_func=sim_func,
                                               ds_features_1=self.ds_train_features_1,
                                               ds_features_2=self.ds_train_features_2,
                                               ds_label=self.ds_train_label)

        nb_of_features = 500 # initial number of features
        dataset_setup = Datasets_Setup(ds_train_features_1=self.ds_train_features_1,
                                       ds_train_features_2=self.ds_train_features_2,
                                       ds_val_features_1=self.ds_val_features_1,
                                       ds_val_features_2=self.ds_val_features_2,
                                       feature_rank= feature_rank)

        eval_metric_increase = True
        eval_metric = -1
        while nb_of_features <self.max_features and eval_metric_increase:
            ds_train_features_it, ds_val_features_it = dataset_setup.new_datasets(nb_of_features)

            eval_metric_it = fit_classifier_fs(
                ds_train_features_it = ds_train_features_it,
                ds_train_label_it=self.ds_train_label,
                ds_val_features_it = ds_val_features_it,
                ds_val_label_it = self.ds_val_label,
                nb_of_features=nb_of_features
            )

            nb_of_features += self.nb_of_features_increment
            eval_metric_increase = eval_metric <= eval_metric_it

        selected_features = feature_rank[ :(nb_of_features-self.nb_of_features_increment)]
        selected_features.sort()

        return selected_features

