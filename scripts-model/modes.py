import os
import sys
import time
import h5py
import torch
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import INP_DIM, OUT_DIM, NUM_SB, SEED
from online_triplet_loss.losses import *
# from models import FloodFCwithAttn, SiameseNetwork
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# from ray import tune
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import BCELoss

from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning import distances, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.regularizers import LpRegularizer

# from ray.air.checkpoint import Checkpoint
le = LabelEncoder()
pdist = torch.nn.PairwiseDistance(p=2)
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
np.set_printoptions(threshold=sys.maxsize)
# from torch.cuda.amp import GradScaler, autocast
# scaler = GradScaler()
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Note: Use_classifier_head is not used in current implementation


def contrastiveForward(data, device, model, lossFn, criterion):
    
    # embed_v2v, embed_i2v, label = data
    # strembed_v2v,libembed_v2v,embed_v2v,strembed_i2v,libembed_i2v,embed_i2v,label = data

    strembed_v2v, libembed_v2v, embed_O_v2v, embed_T_v2v, embed_A_v2v, strembed_i2v, libembed_i2v, embed_O_i2v, embed_T_i2v, embed_A_i2v, label = data
    
    # print(f"device: {device}")
    strembed_v2v, strembed_i2v, label = strembed_v2v.float().to(device), strembed_i2v.float().to(device), label.float().to(device)
    libembed_v2v, libembed_i2v = libembed_v2v.float().to(device), libembed_i2v.float().to(device)

    embed_O_v2v = embed_O_v2v.float().view(-1, INP_DIM).to(device)
    embed_T_v2v = embed_T_v2v.float().view(-1, INP_DIM).to(device)
    embed_A_v2v = embed_A_v2v.float().view(-1, INP_DIM).to(device)
    embed_O_i2v = embed_O_i2v.float().view(-1, INP_DIM).to(device)
    embed_T_i2v = embed_T_i2v.float().view(-1, INP_DIM).to(device)
    embed_A_i2v = embed_A_i2v.float().view(-1, INP_DIM).to(device)

    pred1, attn_weights = model(embed_O_v2v, embed_T_v2v, embed_A_v2v, strembed_v2v, libembed_v2v)
    pred1 = pred1.view(-1, OUT_DIM)
    pred2, attn_weights = model(embed_O_i2v, embed_T_i2v, embed_A_i2v, strembed_i2v, libembed_i2v)
    pred2 = pred2.view(-1, OUT_DIM)

    # print("pred1: ",pred1)
    # print("pred2: ",pred2)

    
    '''
    pred1 = model(embed_v2v)
    # print(f"pred1.shape: {pred1.shape}")
    pred1 = pred1.view(-1, OUT_DIM)
    # print(f"pred1.shape: {pred1.shape}")
    pred2 = model(embed_i2v)
    # print(f"pred2.shape: {pred2.shape}")
    pred2 = pred2.view(-1, OUT_DIM)
    # print(f"pred2.shape: {pred2.shape}")
    '''
    

    if lossFn == 'cosine':
        label[label==1] = -1  
        loss = criterion(pred1, pred2, label)
    else:
        loss = criterion(pdist(pred1, pred2), label.to(device))
    return loss

def offlineTripletForward(data, device, model, criterion):
    
    fnEmbed_x, fnEmbed_y, fnEmbed_z, poslabel, negLabel = data
    # print(fnEmbed_x.shape, fnEmbed_y.shape, fnEmbed_z.shape)
    fnEmbed_x, fnEmbed_y, fnEmbed_z = fnEmbed_x.float().to(device), fnEmbed_y.float().to(device), fnEmbed_z.float().to(device)
    fnEmbed_x, fnEmbed_y, fnEmbed_z = fnEmbed_x.view(
        -1, 1, NUM_SB, INP_DIM), fnEmbed_y.view(-1, 1, NUM_SB, INP_DIM), fnEmbed_z.view(-1, 1, NUM_SB, INP_DIM)

    output1 = model(fnEmbed_x)
    output2 = model(fnEmbed_y)
    output3 = model(fnEmbed_z)
    loss = criterion(output1, output2, output3)
    return loss

def onlineTripletForward(data, device, model, margin):
    labels, embeddings, strEmbed, libEmbed = data
    # print(type(labels))
    # unique, counts = np.unique(labels, return_counts=True)
    # print(np.asarray((unique, counts)).T)
    # exit(1)
    # labels = torch.from_numpy(le.transform(labels)).to(device)
    # if args.dataset_type == 'CSV':
    #     labels = torch.from_numpy(le.transform(labels)).to(device)
    # print('embeddings shape before: ', embeddings.shape)
    # print("labels: ",labels.shape,"\nembeddings: ",embeddings.shape,"\nstrEmbed: ",strEmbed.shape)   #([512]) ([512, 128]) ([512, 1, 100])
    embeddings = embeddings.float().view(-1, 1, NUM_SB, INP_DIM).to(device)
    # embeddings = embeddings.float().view(-1, NUM_SB, INP_DIM).to(device) #replaced the above line with this
    strEmbed = strEmbed.float().to(device)
    
    libEmbed = libEmbed.float().to(device)
    
    # embeddings = embeddings.float().view(-1, INP_DIM).to(device)
    # print('embeddings shape: ', embeddings.shape)
    # print('strEmbed shape: ', strEmbed.shape)
    outputs = model(embeddings, strEmbed, libEmbed) #=>calling forward() of models.py
    outputs = outputs.view(-1, OUT_DIM)
    # hard_pairs = miners.BatchEasyHardMiner()(outputs, labels)
    hard_pairs = miners.TripletMarginMiner(margin)(outputs, labels)
    loss = pml_losses.TripletMarginLoss(margin=margin, embedding_regularizer = LpRegularizer())(outputs, labels, hard_pairs)
    
    # loss = batch_hard_triplet_loss(labels, outputs, margin=args.margin)
    # loss = (torch.ones([args.batch_size, 1]) * loss).to(device)
    # loss, frac = batch_all_triplet_loss(labels, outputs, margin=args.margin)
    # loss = batch_hard_triplet_loss(labels, outputs, margin=config["margin"])
    return loss



def trainSiamese(args, model, train_dataloader, val_dataloader, optimizer, criterion, scheduler, use_classifier_head=False):
    # -------------------> Uncomment to use with ray tune <--------------------------------
# def trainSiamese(config):
#     args = config["args"]
#     device = args.device
#     # model = config["model"]
#     train_dataloader = config["train_dataloader"]
#     # optimizer, criterion, scheduler = config["optimizer"], config["criterion"], config["scheduler"]
#     model = FloodFCwithAttn(args.inp_dim, args.batch_size).to(device)
#     # optimizer = optim.Adam(model.parameters(), betas=(config["beta1"], 0.999), lr=0.001, weight_decay=0.01)
#     optimizer = optim.SGD(model.parameters(), lr=config["lr"], weight_decay=0.01)
#     scheduler = ExponentialLR(optimizer, gamma=0.9)
#     criterion = BCELoss()
#     args.config_name = '_'.join([args.more, args.loss, 'adam', 'b'+str(args.batch_size),
#                             'e'+str(args.epochs), 'm'+str(config["margin"]), str(args.inp_dim)+'D'] )
#     args.best_model_path = os.path.abspath(os.path.join(
#         args.best_model_path, args.config_name+'.model'))
    # -----------------------------------------------------------------------------------------

    device = args.device
    c = 0
    counter = []
    loss_history = []
    avg_losses = []
    avg_val_losses = []
    best_val = 1000000
    best_epoch = 1
    if not os.path.exists('./lossplots'):
        os.mkdir('./lossplots')
    filename = '{}_lossplot.png'.format(args.config_name)
    print('Model on GPU? ', next(model.parameters()).is_cuda)
    loss_func = pml_losses.TripletMarginLoss()

    epoch_nums=[]
    avg_training_losses=[]
    training_time = 0

    for epoch in range(1, args.epochs+1):
        losses = []
        val_losses = []
        start = time.time()
        
        model.train()
        for _, data in enumerate(train_dataloader): 
            optimizer.zero_grad()
            
            # Contrastive loss
            if args.loss == 'cont':
                loss = contrastiveForward(data, device, model, args.loss, criterion)
 
            # Offline Triplet loss
            elif len(data) == 5:
                loss = offlineTripletForward(data, device, model, criterion)
                
            # Online Triplet loss
            else :
                labels, embeddings, strEmbed, libEmbed, cfg = data
                # labels = torch.from_numpy(le.transform(labels)).to(device)
                # if args.dataset_type == 'CSV':
                #     labels = torch.from_numpy(le.transform(labels)).to(device)
                # print('embeddings shape before: ', embeddings.shape)
                # print("labels: ",labels.shape,"\nembeddings: ",embeddings.shape,"\nstrEmbed: ",strEmbed.shape)   #([512]) ([512, 128]) ([512, 1, 100])
                # embeddings = embeddings.float().view(-1, 1, NUM_SB, INP_DIM).to(device)
                embeddings = embeddings.float().to(device)
                # embeddings = embeddings.float().view(-1, NUM_SB, INP_DIM).to(device) #replaced the above line with this
                strEmbed = strEmbed.float().to(device)
                
                libEmbed = libEmbed.float().to(device)

                cfg = cfg.float().to(device)
                
                # embeddings = embeddings.float().view(-1, INP_DIM).to(device)
                # print('embeddings shape: ', embeddings.shape)
                # print('strEmbed shape: ', strEmbed.shape)
                outputs = model(embeddings, strEmbed, libEmbed) #=>calling forward() of models.py
                outputs = outputs.view(-1, args.out_dim)
                # hard_pairs = miners.BatchEasyHardMiner()(outputs, labels)
                hard_pairs = miners.TripletMarginMiner(args.margin)(outputs, labels)
                loss = pml_losses.TripletMarginLoss(margin=args.margin, embedding_regularizer = LpRegularizer())(outputs, labels, hard_pairs)
                
                # loss = batch_hard_triplet_loss(labels, outputs, margin=args.margin)
                # loss = (torch.ones([args.batch_size, 1]) * loss).to(device)
                # loss, frac = batch_all_triplet_loss(labels, outputs, margin=args.margin)
                # loss = batch_hard_triplet_loss(labels, outputs, margin=config["margin"])
        
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            losses.append(loss.item())
            # counter.append(c)
            # c += 1
        counter.append(epoch-1)
        avg_loss = round(sum(losses)/len(losses), 5)
        avg_losses.append(avg_loss)
        # if avg_loss < best_val:
        #     best_val = avg_loss
        #     best_epoch = epoch
        #     torch.save(model, args.best_model_path)
        
        model.eval()
        for _, data in enumerate(val_dataloader):
            # Contrastive loss
            if args.loss == 'cont':
                loss = contrastiveForward(data, device, model, args.loss, criterion)
 
            # Offline Triplet loss
            elif len(data) == 5:
                loss = offlineTripletForward(data, device, model, criterion)
                
            # Online Triplet loss
            else:
                loss = onlineTripletForward(data, device, model, args.margin)
            val_losses.append(loss.item())
        avg_val_loss = round(sum(val_losses)/len(val_losses), 5)
        avg_val_losses.append(avg_val_loss)
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_epoch = epoch
            torch.save(model, args.best_model_path)
            # Maybe required in checkpoint with ray tune
            # checkpoint = Checkpoint.from_directory(args.best_model_path)
        scheduler.step()

        duration = round(time.time()-start, 2)
        
        epoch_nums.append(epoch)
        avg_training_losses.append(avg_loss)
        training_time += duration

        print("Epoch: {}\tLoss: {}\t Val Loss: {}\tTime taken: {}s".format(epoch, avg_loss, avg_val_loss, duration))
        # if epoch-(avg_losses.index(best_val)) >= PATIENCE:
        #     print("Early Stopping ...")
        #     print('Best Loss Value is {} at Epoch {}.'.format(best_val, best_epoch))
        #     # tune.report(loss=best_val)
        #     break

    # save_plot(counter, loss_history, filename)
    # save_plot(counter, avg_losses, filename)

    print(f'\nTotal Training time: {training_time}\n')
    # Plotting the epoch vs loss
    plt.figure(figsize=(20, 14))
    plt.plot(epoch_nums, avg_training_losses, marker='o', linestyle='-')
    plt.yscale('log')  # Set the y-axis to a logarithmic scale
    plt.xlabel('Epoch', fontsize=28)
    plt.ylabel('Loss', fontsize=28)
    plt.title('Epoch vs Loss (in log scale)', fontsize=34)
    plt.grid(True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    # Save the figure as a PDF file
    parent = os.path.dirname(args.best_model_path)
    figfile = os.path.join(parent,'epoch_vs_loss_plot.png')
    plt.savefig(figfile)

    plt.show()

    print(f'Plot stored in {figfile}')

    return


def trainClassifier(args, train_dataloader):
    # siameseNet = SiameseNetwork(args.inp_dim, args.out_dim)
    # siameseNet.load_state_dict(torch.load(args.best_model_path))
    device = args.device
    siameseNet = torch.load(args.best_model_path)
    siameseNet.to(device)
    X_list, y_list = [], []
    siameseNet.eval()
    with torch.no_grad():
        if args.loss == 'trp':
            for _, data in enumerate(train_dataloader, 0):
                embedX, embedY, embedZ, posLabel, negLabel = data
                embedX, embedY, embedZ = embedX.view(
                    -1, 1, NUM_SB, 100), embedY.view(-1, 1, NUM_SB, 100), embedZ.view(-1, 1, NUM_SB, 100)
                out1, out2, out3 = siameseNet(embedX.float().to(
                    device), embedY.float().to(device), embedZ.float().to(device))
                # print('Outputs shapes: ', out1.shape, out2.shape, out3.shape)
                # print(pdist(out1, out2).cpu().numpy().shape)
                X_list.append(pdist(out1, out2).cpu().numpy())
                y_list.append(posLabel.numpy())

                # Uncomment to use L1 distance between the function embeddings
                # print(torch.abs(torch.sub(out1, out2)).cpu().numpy().shape)
                # X_list.append(torch.abs(torch.sub(out1, out2)).cpu().numpy())
                # y_list.append(posLabel.numpy())

                X_list.append(pdist(out1, out3).cpu().numpy())
                y_list.append(negLabel.numpy())
                # X_list.append(torch.abs(torch.sub(out1, out3)).cpu().numpy())
                # y_list.append(negLabel.numpy())

        elif args.loss == 'cont':
            for _, data in enumerate(train_dataloader, 0):
                embedX, embedY, label = data
                out1, out2 = siameseNet(embedX.float().to(
                    device), embedY.float().to(device))
                X_list.append(pdist(out1, out2).cpu().numpy())
                y_list.append(label.numpy())

        else:
            # keys, embeddings = data
            # labels = torch.from_numpy(le.transform(keys)).to(device)
            # embeddings = embeddings.float().to(device).view(-1, 1, NUM_SB, 100)
            # outputs = siameseNet(embeddings)
            # outputs = outputs.view(args.batch_size, -1)
            for _, data in enumerate(train_dataloader, 0):
                # print('len(data): ', len(data))
                posLabel, negLabel, embedX, embedY, embedZ = data
                # print(embedX.shape)
                embedX, embedY, embedZ = embedX.view(-1, 1, NUM_SB, INP_DIM), embedY.view(-1, 1, NUM_SB, INP_DIM), embedZ.view(-1, 1, NUM_SB, INP_DIM)
                out1 = siameseNet(embedX.float().to(device))
                out2 = siameseNet(embedY.float().to(device))
                out3 = siameseNet(embedZ.float().to(device))
                # print('Outputs shapes: ', out1.shape, out2.shape, out3.shape)
                # X_list.append(pdist(out1, out2).cpu().numpy())
                # print(pdist(out1, out2).cpu().numpy().shape)
                # print(torch.abs(torch.sub(out1, out2)).cpu().numpy().shape)
                X_list.append(torch.abs(torch.sub(out1, out2)).cpu().numpy())
                y_list.append(posLabel.numpy())
                # X_list.append(pdist(out1, out3).cpu().numpy())
                X_list.append(torch.abs(torch.sub(out1, out3)).cpu().numpy())
                y_list.append(negLabel.numpy())

    X = np.concatenate(X_list, axis=0)
    # X_scaled = scaler.fit_transform(X)
    y = np.concatenate(y_list, axis=0)
    print(X.shape, y.shape)
    classifier = LogisticRegression(random_state=SEED, max_iter=200)
    # if args.classifier == 'rfc':
    #     classifier = RandomForestClassifier(random_state=SEED)
    # else:
    #     classifier = LogisticRegression(random_state=SEED, max_iter=200)

    # Uncomment when pdist on the output of the model is used
    # classifier.fit(X.reshape((-1, 1)), y)

    # Uncomment when L1 distance on the output of the model is used
    classifier.fit(X, y)

    classifier_path = os.path.dirname(
        args.best_model_path)+'/{}-L1.classifier'.format(args.config_name)
    with open(classifier_path, 'wb') as f:
        pickle.dump(classifier, f)
    return

