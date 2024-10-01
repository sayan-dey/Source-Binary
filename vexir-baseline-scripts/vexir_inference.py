# Need to modify...

#For inferencing the FC model

# Usage: python src/vex2vec_inference.py
#        -bmp ../models/_onl_trp_adam_lr0.001_b512_e200_m0.7_128D_b0.7.model
#        -dp /Pramana/VexIR2Vec/Datafiles/x86-data-all/findutils
#        -test_csv=/Pramana/VexIR2Vec/new-Ground-Truth/findutils-ground-truth-inline-sub-fixed/exp1-arm-sb/findutils-full-arm-clang-12-O0-arm-clang-12-O2.csv
#        -out_dir /home/cs20mtech01004/vexIR/vexNet/test-inference/roc-json/
#        -res_dir /home/cs20mtech01004/vexIR/vexNet/test-inference/results/

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

def get_embed_from_data(row):

    embedding = rw_str_to_arr(row['embedding'])
    embedding = pad_and_reshape_array(embedding, NUM_WALKS, INP_DIM)
    # embedding = merge_emb(embedding)
    # print("final type: ", type(embedding)) #<class 'numpy.ndarray'>
    # print("final shape: ", embedding.shape) #(128,)
    # exit()

    embedding = torch.from_numpy(embedding)
    embedding = embedding.view(NUM_WALKS, -1, INP_DIM).float().to(device)
    row['embedding'] = embedding

    # strEmbed = torch.from_numpy(get_strEmb(row['strRefs']))
    strEmbed = np.fromstring(row['strRefs'].replace('[','').replace(']',''), sep=' ')
    row['strRefs'] = strEmbed

    strEmbed = torch.from_numpy(row['strRefs'])
    strEmbed = strEmbed.view(1, -1).float().to(device)
    row['strEmbed'] = strEmbed
    # print('strEmbed.shape: ', strEmbed.shape)
    # embedding = torch.from_numpy(np.array(embedding).sum(0))

    # libEmbed = torch.from_numpy(get_extlibEmb(row['extlibs']))
    libEmbed = np.fromstring(row['extlibs'].replace('[','').replace(']',''), sep=' ')
    row['extlibs'] = libEmbed

    libEmbed = torch.from_numpy(row['extlibs'])
    libEmbed = libEmbed.view(1, -1).float().to(device)
    row['libEmbed'] = libEmbed

    if 'cfg' in row:
        # print('CFG data exists')
        cfg = str_to_muldim_np_arr(row['cfg'])
        row['cfg'] = cfg

    return row

def get_addr_embed_list(unstrip_file, strip_file, cfg = False):
    unstripped_df = pd.read_csv(unstrip_file, sep='\t', usecols=[0, 1, 2], header=None, names=['addr', 'key', 'name'])
    if cfg == True:
        stripped_df = pd.read_csv(strip_file, sep='\t', usecols=[0, 1, 2, 3, 4, 5, 6], header=None, names=['addr', 'key', 'name', 'strRefs', 'extlibs', 'embedding', 'cfg'])
    else:
        stripped_df = pd.read_csv(strip_file, sep='\t', usecols=[0, 1, 2, 3, 4, 5], header=None, names=['addr', 'key', 'name', 'strRefs', 'extlibs', 'embedding'])
    stripped_df.drop(['key'], axis=1, inplace=True)
    # Drop unstripped function name
    #unstripped_df.drop(['name'], axis=1, inplace=True)
    #data = pd.merge(unstripped_df, stripped_df, on='addr')
    data = pd.merge(unstripped_df, stripped_df, on='addr', suffixes=('_x', '_y'))
    # print(data.columns)
    data = data[data['name_x'].str.startswith("sub_") == False].reset_index().drop(['index'], axis=1).astype({'addr': str})
    data = data[data['key'].str.contains("NO_FILE_SRC") == False].reset_index().drop(['index'], axis=1).astype({'addr': str})
    data = data[data['name_y'].str.startswith("sub_") | data['name_y'].str.fullmatch("main")].reset_index().drop(['index'], axis=1).astype({'addr': str})

    data = data.apply(get_embed_from_data, axis=1)
    addr_list = [int(func_addr) for func_addr in data['addr'].tolist()]
    emb_list = data['embedding'].tolist()
    strEmb_list = data['strEmbed'].tolist()
    libEmb_list = data['libEmbed'].tolist()

    if cfg == True:
        cfg_list = data['cfg'].tolist()
        return list(zip(addr_list, emb_list, strEmb_list, libEmb_list, cfg_list))
    else:
        return list(zip(addr_list, emb_list, strEmb_list, libEmb_list))

def valid_addr_list(data_path_x, pri_bin, model, device):
    # t=time.time()
    valid_addr_list = []
    with open(os.path.join(data_path_x, pri_bin)+'.data', 'r') as file:
        for line in file:
            # Avoiding new line chars at the end using the check below
            # if len(line.split('\t')) == 5:
            if len(line.split('\t')) == 6:
                [func_addr, key, func_name, strRefs, extlibs, embedding] = line.split('\t')
                if func_name.startswith('sub_') or func_name == 'main':
                    try:
                        
                        # embedding = np.vstack([np.fromstring(i.replace('[', '')
                        #                     .replace(']', '')
                        #                     .replace(' ', '  '), sep=' ') for i in eval(embedding)])
                        
                        embedding = pad_and_reshape_array(embedding, NUM_WALKS, INP_DIM)
                        # embedding = merge_emb(embedding)
                        # print(embedding.shape)
                        embedding = torch.from_numpy(embedding)
                        embedding = embedding.view(NUM_WALKS, -1, INP_DIM).float().to(device)
                        # strEmbed = torch.from_numpy(get_strEmb(strRefs))
                        strEmbed = torch.from_numpy(strRefs)
                        strEmbed = strEmbed.view(1, -1).float().to(device)

                        # libEmbed = torch.from_numpy(get_extlibEmb(extlibs))
                        libEmbed = torch.from_numpy(extlibs)
                        libEmbed = libEmbed.view(1, -1).float().to(device)

                        # print('strEmbed.shape: ', strEmbed.shape)
                        # embedding = torch.from_numpy(np.array(embedding).sum(0))

                        with torch.no_grad():
                            output = model(embedding, strEmbed, libEmbed, test=True)
                            # print('output.shape: ', output.shape)
                        valid_addr_list.append((func_addr, output))

                    except Exception as err:
                        print(err)

    # print('Time taken in creating pri_addr_list: {:.2}s'.format(time.time()-t))
    # print('pri_addr_list created.')
    # print(len(pri_addr_list))
    return valid_addr_list


def get_labels_and_sim_scores(vexir2vec_func_list, vexir2vec_dist_list, test_func_list, labels, sim_scores):
    tp_count = 0
    fp_count = 0
    for tuples, dists in zip(vexir2vec_func_list, vexir2vec_dist_list):
        assert len(tuples) == len(dists), "tuples len: {}, dists len: {}".format(len(tuples), len(dists))
        counter = 0
        for idx in range(len(tuples)):
            # True positive
            if tuples[idx] in test_func_list:
                labels.append(1)
                tp_count += 1
                dist = dists[idx]
                sim_scores.append(1/(1+dist))
                break
            else:
                counter += 1
        # False positive
        if counter == len(tuples):
            fp_count += 1
            labels.append(0)
            dist = sum(dists)/len(dists)
            sim_scores.append(1/(1+dist))

    assert fp_count + tp_count == len(vexir2vec_func_list), "fp_count: {}, tp_count {}, len(vexir2vec_func_list): {}".format(fp_count, tp_count, len(vexir2vec_func_list))
    return labels, sim_scores

def get_topK_neigh_and_dist_old(tp_query, tp_data):
    kdt = KDTree(tp_data, leaf_size=30, metric='euclidean')
    vexir2vec_dist, vexir2vec_index = kdt.query(
        tp_query, k=NUM_NEIGHBORS, return_distance=True)
    vexir2vec_dist = vexir2vec_dist.tolist()
    vexir2vec_index = vexir2vec_index.tolist()
    return vexir2vec_index, vexir2vec_dist

def get_topK_neigh_and_dist(tp_query, tp_data):
    kdt = KDTree(tp_data, leaf_size=30, metric='euclidean')
    # vexir2vec_dist, vexir2vec_index = kdt.query(
        # tp_query, k=NUM_NEIGHBORS, return_distance=True)
    vexir2vec_dist, vexir2vec_index = kdt.query(tp_query, k=len(tp_data), return_distance=True)

    # vexir2vec_dist = vexir2vec_dist.tolist()
    # vexir2vec_index = vexir2vec_index.tolist()

    # print('vexir2vec_dist size: ', vexir2vec_dist.size)
    # print('vexir2vec_index size: ',vexir2vec_index.size)
    # print('vexir2vec_dist: ', type(vexir2vec_dist))
    # print('vexir2vec_index: ', vexir2vec_index)

    stride = np.zeros(vexir2vec_dist.shape[0])


    for i in range(0,vexir2vec_dist.shape[0]):
        while True:
            if NUM_NEIGHBORS + stride[i] > vexir2vec_dist.shape[1]:
                stride[i] -= (NUM_NEIGHBORS - unq_cnt)
                break
            r = np.array(vexir2vec_dist[i][:int(NUM_NEIGHBORS+stride[i])])
            r1, counts = np.unique(r,return_counts=True) #counts not req
            unq_cnt = r1.shape[0]
            # dupl_cnt = r.shape[0]-r1.shape[0]
            if unq_cnt < NUM_NEIGHBORS:
                stride[i] = stride[i] + (NUM_NEIGHBORS - unq_cnt)
            else:
                break

    vexir2vec_dist1 = []
    for i in range(0,vexir2vec_dist.shape[0]):
        vexir2vec_dist1.append(vexir2vec_dist[i][:int(NUM_NEIGHBORS+stride[i])])
    vexir2vec_dist1 = np.array(vexir2vec_dist1)

    vexir2vec_index1 = []
    for i in range(0,vexir2vec_index.shape[0]):
        vexir2vec_index1.append(vexir2vec_index[i][:int(NUM_NEIGHBORS+stride[i])])
    vexir2vec_index1 = np.array(vexir2vec_index1)

    # print("vexir2vec_index1[0]: ", vexir2vec_index1[0])
    return vexir2vec_index1, vexir2vec_dist1


# Function for matching Binary to Source
def match_bin_src(v2v_key_embed_list, i2v_key_embed_list):

    print("\n\n-----------Binary to Source Matching---------\n\n")

    #from here i can segregate metric calculation code for a particular class
    
    # print(i2v_key_embed_list)
    # exit()
    
    # search = np.array([tup[1] for tup in i2v_key_embed_list]).astype(np.float32)
    # search_key_list = [i[0] for i in i2v_key_embed_list]
    # search_key_list = np.array(search_key_list)
    
    print(f"number of i2v embeds before duplicate filtering: {len(i2v_key_embed_list)}")
    
    # Use a dictionary to keep unique embedings and their keys
    unique_dict = {}
    
    for key, embedding in i2v_key_embed_list:
        # Convert the numpy array to string before using it as a dictionary key
        embedding_str = str(embedding)
        if embedding_str not in unique_dict:
            unique_dict[embedding_str] = key

    # Extract the keys and embeddings from the dictionary
    unique_keys = list(unique_dict.values())
    unique_embeddings = list(unique_dict.keys())
    
    for i in range(0, len(unique_embeddings)):
        unique_embeddings[i] = np.fromstring(unique_embeddings[i].replace('[','').replace(']',''), sep=' ')
        
    print(f"number of i2v embeds after duplicate filtering: {len(unique_embeddings)}")
    # print(f"number of i2v embeds after duplicate filtering: {len(unique_keys)}")

    # Convert to numpy arrays
    search = np.array(unique_embeddings).astype(np.float32)
    search_key_list = np.array(unique_keys)
    
    kdt = kdtree_neigh(search)

    # query_key_embed_list = query_key_embed_list_full[start_chunk_index:end_chunk_index]
    query = np.array([tup[1] for tup in v2v_key_embed_list]).astype(np.float32)
    #print(query.shape)
    query_key_list = [i[0] for i in v2v_key_embed_list]
    
    ir2vec_func_index, ir2vec_func_dist = kdt.get_topK_neigh_and_dist(query, len(search))
    
    # print(search[:10])
    # print(len(search))#1552
    # print(len(query))#1552
    # exit()

    reciprocal_rank_sum = 0
    ir2vec_func_index = np.array(ir2vec_func_index)
    ir2vec_func_dist = np.array(ir2vec_func_dist)
    
    
    #checking matches manually ...

    cnt=0
    matchFalse=0
    matchTrue=0
    top_k_cnt=0
    for index in range(len(ir2vec_func_index)):
        arr = ir2vec_func_index[index]
        top_k_cnt+=len(arr) #count in top k
        match = False
        for ind in arr:
            if search_key_list[ind] == query_key_list[index]:
                match=True
                break
            
        if match == True:
            matchTrue+=1
        else:
            matchFalse+=1
            
        cnt+=1
    
    print(f"Comparisons count: {cnt}")
    print(f"Match count: {matchTrue}")
    print(f"No Match count: {matchFalse}")
    print(f"Average number of embeds in top k: {top_k_cnt/cnt}")

    return
    
        
    matches = 0
    total_gt = 0
    ap_list = []

    for index in range(len(ir2vec_func_index)):
        matches_new = 0
        avg_prec = 0
        prev_dist = -1
        rank = 0
        last_rank = rank
        first_match = 0
        arr = ir2vec_func_index[index]
        print("search_key_list[arr]: ",search_key_list[arr])
        print("query_key_list[index]: ", query_key_list[index])
        print()
        # exit()
        keymatch_arr = (search_key_list[arr] == query_key_list[index])
        # print(keymatch_arr)
        # exit()
        for label_index, label in enumerate(keymatch_arr):
            if prev_dist != ir2vec_func_dist[index][label_index]:
                prev_dist = ir2vec_func_dist[index][label_index]
                rank += 1
            if label and last_rank != rank:
                matches_new += 1
                avg_prec += matches_new/rank
                last_rank = rank
                if first_match == 0:
                    reciprocal_rank_sum += 1/rank
                    matches += 1
                    first_match = 1
                    
        if matches_new:
            avg_prec /= matches_new
        total_gt += 1
        ap_list.append(avg_prec)
    

    mean_ap = sum(ap_list)/len(ap_list)
    acc = (matches/total_gt)*100
    mrr = reciprocal_rank_sum/total_gt

    # with open(res_file, 'a') as f:
    #     f.write('Config: ' + base_config + '\n')
    #     f.write('MAP: ' + str(mean_ap) + '\n')
    #     f.write('Acc: ' + str(acc) + '\n')
    #     f.write('MRR: ' + str(mrr) + '\n')
    print('MAP: ', mean_ap)
    print('Acc: ', acc)
    print('MRR: ', mrr)
    
    return



# Function for matching Source to Binary
def match_src_bin(i2v_key_embed_list, v2v_key_embed_list):

    print("\n\n-----------Source to Binary Matching---------\n\n")
    
    print(f"number of v2v embeds before duplicate filtering: {len(v2v_key_embed_list)}")
    
    # Use a dictionary to keep unique embedings and their keys
    unique_dict = {}
    
    for key, embedding in v2v_key_embed_list:
        # Convert the numpy array to string before using it as a dictionary key
        embedding_str = str(embedding)
        if embedding_str not in unique_dict:
            unique_dict[embedding_str] = key

    # Extract the keys and embeddings from the dictionary
    unique_keys = list(unique_dict.values())
    unique_embeddings = list(unique_dict.keys())
    
    for i in range(0, len(unique_embeddings)):
        unique_embeddings[i] = np.fromstring(unique_embeddings[i].replace('[','').replace(']',''), sep=' ')
        
    print(f"number of v2v embeds after duplicate filtering: {len(unique_embeddings)}")

    # Convert to numpy arrays
    search = np.array(unique_embeddings).astype(np.float32)
    search_key_list = np.array(unique_keys)
    
    kdt = kdtree_neigh(search)

    # query_key_embed_list = query_key_embed_list_full[start_chunk_index:end_chunk_index]
    query = np.array([tup[1] for tup in i2v_key_embed_list]).astype(np.float32)
    #print(query.shape)
    query_key_list = [i[0] for i in i2v_key_embed_list]
    
    vexir2vec_func_index, vexir2vec_func_dist = kdt.get_topK_neigh_and_dist(query, len(search))
    
    # print(search[:10])
    # print(len(search))#1552
    # print(len(query))#1552
    # exit()

    reciprocal_rank_sum = 0
    vexir2vec_func_index = np.array(vexir2vec_func_index)
    vexir2vec_func_dist = np.array(vexir2vec_func_dist)
    
    
    #checking matches manually ...
    cnt=0
    matchFalse=0
    matchTrue=0
    top_k_cnt=0
    for index in range(len(vexir2vec_func_index)):
        arr = vexir2vec_func_index[index]
        top_k_cnt+=len(arr) #count in top k
        match = False
        for ind in arr:
            if search_key_list[ind] == query_key_list[index]:
                match=True
                break
            
        if match == True:
            matchTrue+=1
        else:
            matchFalse+=1
            
        cnt+=1
    
    print(f"Comparisons count: {cnt}")
    print(f"Match count: {matchTrue}")
    print(f"No Match count: {matchFalse}")
    print(f"Average number of embeds in top k: {top_k_cnt/cnt}")

    return


def get_topK_scores(args, csv_filepath):
    
    model = torch.load(args.best_model_path, map_location=device)
    model.eval()
    
    d = pd.read_csv(csv_filepath)
    
    # filter the dataframe and take only ones with label=1
    #....otherwise repetition of same datapoints will happen
    # d= d[d['label']==1]
    
    d['embed_O_bin'] = d['embed_O_bin'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    d['embed_T_bin'] = d['embed_T_bin'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    d['embed_A_bin'] = d['embed_A_bin'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    d['strembed_bin'] = d['strembed_bin'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    d['libemb_bin'] = d['libemb_bin'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))

    d['embed_O_src'] = d['embed_O_src'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    d['embed_T_src'] = d['embed_T_src'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    d['embed_A_src'] = d['embed_A_src'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    d['strembed_src'] = d['strembed_src'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    d['libemb_src'] = d['libemb_src'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))

    '''
    d['embed_i2v'] = d['embed_i2v'].apply(lambda x: np.fromstring(x.replace('[','').replace(']',''), sep=' '))
    
    # reducing i2v embed dim from 300 to 128
    data_stacked = np.vstack(d['embed_i2v'].values) # Stack all arrays in the 'embed_i2v' column vertically to create a 2D array
    # pca = PCA(n_components=INP_DIM)
    # reduced_data = pca.fit_transform(data_stacked) # Apply PCA
    # d['embed_i2v'] = list(reduced_data) # Update the 'embed_i2v' column with the reduced data

    # Load the stored PCA object
    pca_model_path = os.path.join(os.path.dirname(args.best_model_path), 'pca-model.pkl')
    pca_loaded = joblib.load(pca_model_path)

    # Transform data using the loaded PCA object
    d['embed_i2v'] = list(pca_loaded.transform(data_stacked))

    #converting space separated strings to numpy array
    d['strembed_i2v'] = d['strembed_i2v'].apply(lambda x: np.array(x.split(), dtype=float))
    d['libemb_i2v'] = d['libemb_i2v'].apply(lambda x: np.array(x.split(), dtype=float))
    '''
    
    data_v2v = []
    data_i2v = []
    for index,row in d.iterrows():
        data_v2v.append((row['key_bin'], torch.from_numpy(row['embed_O_bin']), torch.from_numpy(row['embed_T_bin']), 
                         torch.from_numpy(row['embed_A_bin']), torch.from_numpy(row['strembed_bin']), torch.from_numpy(row['libemb_bin'])))
        data_i2v.append((row['key_src'], torch.from_numpy(row['embed_O_src']), torch.from_numpy(row['embed_T_src']), 
                         torch.from_numpy(row['embed_A_src']), torch.from_numpy(row['strembed_src']), torch.from_numpy(row['libemb_src'])))
        

    v2v_dataloader = DataLoader(data_v2v, batch_size=256, shuffle=False, num_workers=10) #, num_workers=10
    i2v_dataloader = DataLoader(data_i2v, batch_size=256, shuffle=False, num_workers=10) #, num_workers=10
    print('len(v2v_dataloader): ', len(v2v_dataloader))
    print('len(i2v_dataloader): ', len(i2v_dataloader))
    
    v2v_key_embed_list = []
    v2v_key_list = []

    for _, data in enumerate(v2v_dataloader):
        
        with torch.no_grad():
            output, _ = model(data[1].float().to(device), data[2].float().to(device), data[3].float().to(device), data[4].float().to(device), data[5].float().to(device))
            # output = model(data[1].float().to(device), test=True)
        v2v_key_embed_list.extend([np.array(embed).astype(np.float32) for embed in output.tolist()])
        v2v_key_list.extend(data[0])
        
    v2v_key_embed_list = list(zip(v2v_key_list, v2v_key_embed_list))
    
    i2v_key_embed_list = []
    i2v_key_list = []
    
    for _, data in enumerate(i2v_dataloader):
        with torch.no_grad():
            output, _ = model(data[1].float().to(device), data[2].float().to(device), data[3].float().to(device), data[4].float().to(device), data[5].float().to(device))
            # output = model(data[1].float().to(device), test=True)
            
        i2v_key_embed_list.extend([np.array(embed).astype(np.float32) for embed in output.tolist()])
        i2v_key_list.extend(data[0])
        
    i2v_key_embed_list = list(zip(i2v_key_list, i2v_key_embed_list))
    # print(i2v_key_embed_list[:10])
    # exit()

    # -------------------Matching Binary to Source--------------------
    match_bin_src (v2v_key_embed_list, i2v_key_embed_list)

    # -------------------Matching Source to Binary--------------------
    match_src_bin(i2v_key_embed_list, v2v_key_embed_list)
    
    


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
