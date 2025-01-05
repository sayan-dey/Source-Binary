#For inferencing the FC model


#Eg cmd: 
# python siamese_inference.py 
# -bmp /Pramana/VexIR2Vec/Source_Binary/models/COFO-model-with-8-pass-seqs-train-data-str-lib-noRelu/cont_adam_lr0.0001_b32_e500_m0.04_128D_b0.9.model 
# -dp na -csv_filepath /Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_data_infer_str_lib_only_test.csv -out_dir abc 
# -res_dir /Pramana/VexIR2Vec/Source_Binary/inference/ > /Pramana/VexIR2Vec/Source_Binary/inference/results_k25_w_str_lib.txt

import os
import time
import torch
import random
import json
import numpy as np
import pandas as pd
# from utils import pad_and_reshape_array, merge_emb, get_extlibEmb, get_strEmb, print_vex2vec, ft, NUM_SB, INP_DIM, SEED, NUM_NEIGHBORS
from utils import kdtree_neigh, NUM_SB, INP_DIM, NUM_WALKS, SEED, NUM_NEIGHBORS
from scipy.stats import rankdata
from scipy.spatial import cKDTree
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from argparse import ArgumentParser
# from torch2trt import torch2trt

import joblib

import warnings

warnings.filterwarnings("ignore")

random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)

device = torch.device('cpu')
le=LabelEncoder()
pdist=torch.nn.PairwiseDistance()
scaler=preprocessing.StandardScaler()

# Function for matching Binary to Source
def match_bin_src (data_v2v, data_i2v):

    print("\n\n-----------Binary to Source Matching---------\n\n")
    
    
    total_match_prob = 0.0
    comp_cnt=0
    
    for v2v_ind in range(len(data_v2v)):
        matchcnt=0
        total_comp_cnt=0
        for i2v_ind in range(len(data_i2v)):
            if data_v2v[v2v_ind] == data_i2v[i2v_ind]: #Keys matched
                matchcnt+=1
            total_comp_cnt+=1
        
        total_match_prob += (matchcnt/total_comp_cnt)
        comp_cnt += 1
        
    print(f"Overall Match probability: {total_match_prob/comp_cnt}")
    
    return
            
        

# Function for matching Source to Binary
def match_src_bin(data_i2v, data_v2v):

    print("\n\n-----------Source to Binary Matching---------\n\n")
    
    total_match_prob = 0.0
    comp_cnt=0
    
    for i2v_ind in range(len(data_i2v)):
        matchcnt=0
        total_comp_cnt=0
        for v2v_ind in range(len(data_v2v)):
            if data_i2v[i2v_ind] == data_v2v[v2v_ind]: #Keys matched
                matchcnt+=1
            total_comp_cnt+=1
        
        total_match_prob += (matchcnt/total_comp_cnt)
        comp_cnt += 1
        
    print(f"Overall Match probability: {total_match_prob/comp_cnt}")
    
    return
    
    

def get_topK_scores(args, csv_filepath):
    

    # model = torch.load(args.best_model_path, map_location=device)
    # model.eval()
    
    d = pd.read_csv(csv_filepath)
    
    # filter the dataframe and take only ones with label=1
    #....otherwise repetition of same datapoints will happen
    # d= d[d['label']==1]
   
    
    data_v2v = []
    data_i2v = []
    for index,row in d.iterrows():
        data_v2v.append(row['key_v2v'])
        data_i2v.append(row['key_i2v'])
        
    random.shuffle(data_v2v)
    random.shuffle(data_i2v)
        
    
    # -------------------Matching Binary to Source--------------------
    match_bin_src (data_v2v, data_i2v)

    # -------------------Matching Source to Binary--------------------
    match_src_bin(data_i2v, data_v2v)
    
    
if __name__ == "__main__":
    # print_vex2vec()
    parser = ArgumentParser(description='VexIR2Vec framework for binary similarity.')
    parser.add_argument('-bmp', '--best_model_path', required=True, help='Path to the best model')
    parser.add_argument('-dp', '--data_path', help='Directory containing data files of all the projects.')
    parser.add_argument('-csv_filepath', '--csv_filepath', required=True, help='Path to the test data csv')
    parser.add_argument('-res_dir', '--res_dir', required=True, help='Path to output directory to store results (prec, recall, f1)')
    parser.add_argument('-out_dir', '--out_dir', required=True, help='Path to output directory of roc json files')
    # parser.add_argument('-cfg', '--use_cfg', type = bool, default=False, help='Use CFG data for inferencing if set')

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    args.device = device
    get_topK_scores(args, args.csv_filepath)
