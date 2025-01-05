# For training the FC model 

#Eg cmd:
# python siamese_training.py 
# -td /Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_train_data_with_8_pass_seqs.csv 
# -bs 32 -inpd 128 -l cont -e 500 -m 0.08 -opt adam -lr 0.0001 
# -bmp /Pramana/VexIR2Vec/Source_Binary/models/COFO-model-with-8-pass-seqs-train-data-noRelu 
# > /Pramana/VexIR2Vec/Source_Binary/models/COFO-model-with-8-pass-seqs-train-data-noRelu/training_log.txt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, LinearLR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import argparse
import os

from losses import *
from modes import trainSiamese
# from models import FloodFCwithAttn
from model_OTA import FCNNWithAttention
import math
from utils import preprocess_CSV_dataset, SEED, INP_DIM

random.seed(SEED)
np.random.seed(SEED)

def main(args):
    
    device = args.device
    train_data, pos_pairs, neg_pairs = preprocess_CSV_dataset(args.training_data, args.pca_model_path)
    # train_list = list(data.itertuples(index=False, name=None))
    
    # print(train_data[0][0].shape) #embed_v2v:  torch.Size([128])
    # print(train_data[0][1].shape) #embed_i2v:  torch.Size([128])
    # print(train_data[0][2].shape) #label:  torch.Size([])

    # Offline triplet loss
    # if args.loss == 'trp':
    #     train_data = gen_train_data_triplets(train_list)
    #     print('Size of Training Data: ', len(train_data))

    # Contrastive loss
    # train_data, pos_pairs, neg_pairs = gen_train_data_pairs_new(train_list)
    print('Size of Training Data: ', len(train_data))
    print('#Pos pairs: ', len(pos_pairs))
    print('#Neg pairs: ', len(neg_pairs))
    
    random.shuffle(train_data)
    train_size = int(0.8 * len(train_data))
    test_size = len(train_data) - train_size
    print(train_size, test_size)
    train_data, val_data = random_split(train_data, [train_size, test_size])
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=10) #, num_workers=10
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=10) #, num_workers=10
    print('len(train_dataloader): ', len(train_dataloader))
    print('len(val_dataloader): ', len(val_dataloader))
    
    # exit()

    #For /Pramana/VexIR2Vec/Source_Binary/models/COFO-model-v2v-bestconfig-finetuned
    config =  {
            "activation": "silu",
            "batch_size": 64,
            "beta": 0.7266643338917541,
            "concat_layer": -2,
            "drop_units": [
                0.028316946776863926
            ],
            "gamma": 0.8171417255946217,
            "hidden": [
                180
            ],
            "lr": 0.005,
            "margin": 0.08,
            "num_O_layers": 1,
            "num_layers": 1,
            "opt": "adam",
            "sched": "Linear_lr",
            "thresh_max": 150,
            "thresh_min": 50
        }
    
    
    if args.pretrained_model == '':
        # model = FloodFCwithAttn(args.inp_dim, args.batch_size).to(device)
        model = FCNNWithAttention(INP_DIM, config=config).to(device)

    else:
        model = torch.load(args.pretrained_model)
        model.to(device)

    
    if args.loss == 'trp':
        criterion = TripletLoss()
    elif args.loss == 'cont':
        criterion = ContrastiveLoss(margin=0.3)
    elif args.loss == 'cosine':
        criterion = torch.nn.CosineEmbeddingLoss()
    else:
        criterion = BCELoss()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), betas=(args.beta, 0.999), lr=args.lr, weight_decay=0.01)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), args.lr, weight_decay=0.01)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=0.01)
        
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    scheduler = LinearLR(optimizer)

    print('\nTraining the Siamese Net...')
    trainSiamese(args, model, train_dataloader, val_dataloader, optimizer, criterion, scheduler)

    # print('Training the classfier ...')
    # trainClassifier(args, train_dataloader)
    print('Training finished.')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-td', '--training_data', dest='training_data',
                        help='training data csv path', default=None)   
    parser.add_argument('-bs', '--batch_size', type=int, required=True) 
    parser.add_argument('-output', '--output_path', dest='output_path',
                        help='output csv file path', default=None)
    parser.add_argument('-ptm', '--pretrained_model', help='Load a pretrained model and train further', default='')
    parser.add_argument('-bmp', '--best_model_path', required=True, help='Path to the best model')
    parser.add_argument('-inpd', '--inp_dim', type=int, required=True)
    parser.add_argument('-l', '--loss', required=True, help='Loss to be used.')
    parser.add_argument('-e', '--epochs', type=int, required=True)
    parser.add_argument('-m', '--margin', type=float, required=True)
    parser.add_argument('-b', '--beta', type=float, default=0.9, help='beta1 to be used in Adam.')
    parser.add_argument('-opt', '--optimizer', required=True, help='Optimizer to be used.')  # 'adam', 'sgd'
    parser.add_argument('-lr', '--lr', required=True, type=float, help='Learning rate to be used.')
    
    args = parser.parse_args()

    args.model = args.loss.lower()
    args.optimizer = args.optimizer.lower()
    # args.dataset_type = args.dataset_type.upper()
    # args.classifier = args.classifier.lower()
    args.pca_model_path = os.path.join(args.best_model_path, 'pca-model.pkl')
    
    # For new saved models, this can work.
    # To use older saved models, args.more needs to be put at the end of list.
    args.config_name = '_'.join([args.loss, args.optimizer, 'lr'+str(args.lr),'b'+str(args.batch_size),
                                'e'+str(args.epochs), 'm'+str(args.margin), str(args.inp_dim)+'D'])
    if args.beta is not None:
        args.config_name += ('_b'+str(args.beta))
    
    if not os.path.exists(args.best_model_path):
        os.mkdir(args.best_model_path)
    args.best_model_path = os.path.abspath(os.path.join(args.best_model_path, args.config_name+'.model'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    args.device = device
    
    print('Available Device: ', device)
    main(args)
    