import os
import glob
import time
import h5py
import torch
import random
import pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn import preprocessing
from sklearn.decomposition import IncrementalPCA
from pyfiglet import figlet_format
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

import joblib


V2V_DIM = 128
I2V_DIM = 300
NUM_SB = 1
INP_DIM = 128
OUT_DIM =100
NUM_SB=1
SEED=1004
# NUM_WALKS = 100
NUM_WALKS = 1
NUM_NEIGHBORS=100

random.seed(SEED)
np.random.seed(SEED)
    

def preprocess_CSV_dataset(csv_filepath, pca_model_path):
    d = pd.read_csv(csv_filepath)
    # d.drop(['index'], axis=1, inplace=True)
    print('Shape of loaded DF: ', d.shape)
    
    d['embed_O_v2v'] = d['embed_O_v2v'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    d['embed_T_v2v'] = d['embed_T_v2v'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    d['embed_A_v2v'] = d['embed_A_v2v'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    d['strembed_v2v'] = d['strembed_v2v'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    d['libemb_v2v'] = d['libemb_v2v'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
   
    d['embed_O_i2v'] = d['embed_O_i2v'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    d['embed_T_i2v'] = d['embed_T_i2v'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    d['embed_A_i2v'] = d['embed_A_i2v'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))


    # reducing i2v embed dim from 300 to 128
    data_stacked = np.vstack(d['embed_O_i2v'].values) # Stack all arrays in the 'embed_i2v' column vertically to create a 2D array
    pca = PCA(n_components=INP_DIM)
    pca.fit(data_stacked)  # Fit PCA on the training data

    # Store the fitted PCA object
    joblib.dump(pca, pca_model_path)

    # Load the stored PCA object
    pca_loaded = joblib.load(pca_model_path)

    # Transform data using the loaded PCA object
    d['embed_O_i2v'] = list(pca_loaded.transform(data_stacked))

    data_stacked = np.vstack(d['embed_T_i2v'].values)
    d['embed_T_i2v'] = list(pca_loaded.transform(data_stacked))

    data_stacked = np.vstack(d['embed_A_i2v'].values)
    d['embed_A_i2v'] = list(pca_loaded.transform(data_stacked))


    # reduced_data = pca.fit_transform(data_stacked) # Apply PCA
    # d['embed_i2v'] = list(reduced_data) # Update the 'embed_i2v' column with the reduced data

    #converting space separated strings to numpy array
    d['strembed_i2v'] = d['strembed_i2v'].apply(lambda x: np.array(x.split(), dtype=float))
    d['libemb_i2v'] = d['libemb_i2v'].apply(lambda x: np.array(x.split(), dtype=float))
    


    #before generating can do a train,test,validation split...do it
    
    pos_pairs, neg_pairs = [], []
    
    for idx in range(len(d)):
        # row = train_list[idx]
        row = d[idx:idx+1]
        
        row_label = row.loc[row.index[0], 'label'] # accessing value at 0th row and 'label' column
        
        # value = data.loc[1, "Age"] #use of loc
        if row_label == 1: #similar
            #strembed_v2v,libemb_v2v,embed_v2v,strembed_i2v,libemb_i2v,embed_i2v,label
            pos_pairs.append((torch.from_numpy(row.loc[row.index[0], 'strembed_v2v']), torch.from_numpy(row.loc[row.index[0], 'libemb_v2v']), torch.from_numpy(row.loc[row.index[0], 'embed_O_v2v']), torch.from_numpy(row.loc[row.index[0], 'embed_T_v2v']), torch.from_numpy(row.loc[row.index[0],'embed_A_v2v']), 
            torch.from_numpy(row.loc[row.index[0],'strembed_i2v']), torch.from_numpy(row.loc[row.index[0], 'libemb_i2v']), torch.from_numpy(row.loc[row.index[0], 'embed_O_i2v']), torch.from_numpy(row.loc[row.index[0], 'embed_T_i2v']), torch.from_numpy(row.loc[row.index[0],'embed_A_i2v']), torch.from_numpy(np.array(1))))
            
        else:  #dissimilar
            #strembed_v2v,libemb_v2v,embed_v2v,strembed_i2v,libemb_i2v,embed_i2v,label
            neg_pairs.append((torch.from_numpy(row.loc[row.index[0], 'strembed_v2v']), torch.from_numpy(row.loc[row.index[0], 'libemb_v2v']), torch.from_numpy(row.loc[row.index[0], 'embed_O_v2v']), torch.from_numpy(row.loc[row.index[0], 'embed_T_v2v']), torch.from_numpy(row.loc[row.index[0],'embed_A_v2v']), 
            torch.from_numpy(row.loc[row.index[0],'strembed_i2v']), torch.from_numpy(row.loc[row.index[0],'libemb_i2v']), torch.from_numpy(row.loc[row.index[0], 'embed_O_i2v']), torch.from_numpy(row.loc[row.index[0],'embed_T_i2v']), torch.from_numpy(row.loc[row.index[0],'embed_A_i2v']), torch.from_numpy(np.array(0))))
            
    
    train_data = pos_pairs + neg_pairs
    return train_data , pos_pairs, neg_pairs
            

class kdtree_neigh:
    def __init__(self, tp_data):
        #kdt = KDTree(tp_data, leaf_size=30, metric='euclidean')
        kdtree_fit_start = time.time()
        self.kdt = cKDTree(tp_data, leafsize=100)
        kdtree_fit_end = time.time()
        # print('KDTree fitting time: ', kdtree_fit_end - kdtree_fit_start)

    def get_duplicate_dists(self, dist_arr):
        stride = 0
        while True:
            if NUM_NEIGHBORS + stride > dist_arr.shape[0]:
                stride -= (NUM_NEIGHBORS - unq_cnt)
                break
            r = np.array(dist_arr[:int(NUM_NEIGHBORS+stride)])
            r1 = np.unique(r)
            unq_cnt = r1.shape[0]
            if unq_cnt < NUM_NEIGHBORS:
                stride += (NUM_NEIGHBORS - unq_cnt)
            else:
                break
        return dist_arr[:int(NUM_NEIGHBORS+stride)]

    def get_topK_neigh_and_dist(self, tp_query, k):
        #vexir2vec_dist, vexir2vec_index = kdt.query(tp_query, k=len(tp_data), return_distance=True)
        kdtree_query_start = time.time()
        vexir2vec_dist, vexir2vec_index = self.kdt.query(tp_query, k=k, p=2, workers=-1)
        kdtree_query_end = time.time()
        # print('KDTree query time: ', kdtree_query_end - kdtree_query_start)

        duplicate_neigh_start = time.time()
        duplicate_noise_start = time.time()
        stride = np.zeros(vexir2vec_dist.shape[0])

        ### Remove when we have fixed model
        # vexir2vec_dist = np.array(vexir2vec_dist)
        # print(vexir2vec_dist.shape)
        # noise = np.random.normal(size=vexir2vec_dist.shape)
        # print(noise.shape)
        # vexir2vec_dist = np.add(vexir2vec_dist, noise)
        # duplicate_noise_end = time.time()

        # print('Duplicate noise time: ', duplicate_noise_end - duplicate_noise_start)
        
        unq_cnt = 0
        duplicate_apply_start=time.time()
        for i in range(0,vexir2vec_dist.shape[0]):
            while True:
                if NUM_NEIGHBORS + stride[i] > vexir2vec_dist.shape[1]:
                    stride[i] -= (NUM_NEIGHBORS - unq_cnt)
                    break
                r = np.array(vexir2vec_dist[i][:int(NUM_NEIGHBORS+stride[i])])
                r1 = np.unique(r)
                unq_cnt = r1.shape[0]
                if unq_cnt < NUM_NEIGHBORS:
                    stride[i] += (NUM_NEIGHBORS - unq_cnt)
                else:
                    break
        vexir2vec_dist1 = []
        for i in range(0,vexir2vec_dist.shape[0]):
            vexir2vec_dist1.append(vexir2vec_dist[i][:int(NUM_NEIGHBORS+stride[i])])
        vexir2vec_dist1 = np.array(vexir2vec_dist1)
        # vexir2vec_dist1 = np.apply_along_axis(get_duplicate_dists, 1, vexir2vec_dist)
        duplicate_apply_end=time.time()
        # print('Duplicate apply time: ', duplicate_apply_end - duplicate_apply_start)
        duplicate_index_start=time.time()
        vexir2vec_index1 = []

        for i in range(0,vexir2vec_index.shape[0]):
            vexir2vec_index1.append(vexir2vec_index[i][:int(vexir2vec_dist1[i].shape[0])])
        vexir2vec_index1 = np.array(vexir2vec_index1)
        duplicate_index_end=time.time()
        # print('Duplicate index time: ', duplicate_index_end - duplicate_index_start)
        duplicate_neigh_end = time.time()
        # print('Duplicate neigh time: ', duplicate_neigh_end - duplicate_neigh_start)
        return vexir2vec_index1, vexir2vec_dist1