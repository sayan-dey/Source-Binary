import time
import torch
import random
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from utils import INP_DIM, OUT_DIM, SEED, NUM_SB

random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)

#let's start by modifying below model...        

class FloodFCwithAttn(nn.Module):
    def __init__(self, EMBED_DIM, batch_size):
        super(FloodFCwithAttn, self).__init__()
        self.batch_size = batch_size    #Comment-out while testing
        # Consuming all super blocks
        self.layer0 = nn.ModuleDict({f"fc_{i}": nn.Sequential(nn.Linear(EMBED_DIM, 450),
                                                              nn.ReLU()) for i in range(NUM_SB)})
        # self.attn_layer = nn.Linear(200, 1)

        # nn.linear(x,y) => input: x dimensional, output: y dimensional
        # Consuming concatenated layer0 outputs
        # self.layer0 = nn.Sequential(nn.Linear(500, 600), 
        #                             nn.ReLU())

        # self.layer1 = nn.Sequential(nn.Linear(600, 600), 
        #                             nn.Tanh(),
        #                             nn.Dropout(0.2),
        #                             nn.BatchNorm1d(600))
        # self.layer11 = nn.Sequential(nn.Linear(600, 600), 
        #                             nn.Tanh(),
        #                             nn.Dropout(0.2),
        #                             nn.BatchNorm1d(600))
        # self.layer12 = nn.Sequential(nn.Linear(600, 600), 
        #                             nn.Tanh(),
        #                             nn.Dropout(0.2),
        #                             nn.BatchNorm1d(600))
        # self.layer13 = nn.Sequential(nn.Linear(600, 600), 
        #                             nn.Tanh(),
        #                             nn.Dropout(0.2),
        #                             nn.BatchNorm1d(600))                                                                                                            

        # self.layer2 = nn.Sequential(nn.Linear(600, 500),
        #                             nn.Tanh(),
        #                             nn.Dropout(0.2),
        #                             nn.BatchNorm1d(500))
        
        # self.layer3 = nn.Sequential(nn.Linear(500, 300),
        #                             nn.Tanh(),
        #                             nn.Dropout(0.2),
        #                             nn.BatchNorm1d(300))

        # self.layer4 = nn.Sequential(nn.Linear(300, 200),
        #                             nn.Tanh(),
        #                             nn.Dropout(0.2),
        #                             nn.BatchNorm1d(200))                      
        # Emitting final function level embeddings
        # self.layerOut = nn.Linear(200, 200)
        
        self.layer1 = nn.Sequential(nn.Linear(450, 400), 
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.BatchNorm1d(400))

        self.layer2 = nn.Sequential(nn.Linear(400, 300),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.BatchNorm1d(300))

        self.layer3 = nn.Sequential(nn.Linear(300, OUT_DIM),
                                    nn.ReLU(),
                                    nn.Dropout(0.25),
                                    nn.BatchNorm1d(OUT_DIM))
        
        # self.layer4 = nn.Sequential(nn.Linear(300, 200),
        #                             nn.ReLU(),
        #                             nn.Dropout(0.2),
        #                             nn.BatchNorm1d(200))
        
        # Emitting final function level embeddings
        self.layerOut = nn.Sequential(nn.Linear(OUT_DIM, OUT_DIM))


    def sub_forward(self, input1, test):
        if test:
            self.batch_size = 1
        # print('input1 shape: ', input1.shape)
        # print('input2 shape: ', input2.shape)
            
        layer0_outs = torch.stack([self.layer0[f"fc_{i}"](
            input) for i, input in enumerate(input1.view(NUM_SB, -1, INP_DIM))], axis=0)
        # print('layer0_outs.shape: ', layer0_outs.shape)

        layer0_outs = layer0_outs.view(-1, NUM_SB, 450)
        
        # # print('layer0_outs.shape after reshaping: ', layer0_outs.shape)
        # attn_wts = torch.nn.functional.softmax(self.attn_layer(layer0_outs), dim=1)
        # # print('attn_wts: ', attn_wts.shape)
        # # print('sum in 0th batch: ', torch.sum(attn_wts[0,:,:]))
        # attended_layer0_outs = torch.bmm(attn_wts.view(-1, 1, NUM_SB), layer0_outs)  # 100D vector
        # # print('attended_layer0_outs: ', attended_layer0_outs.shape)
        # # print('attended_layer0_outs: ', attended_layer0_outs)

        # print(f"test: {test}")
        if not test:
            # print(f"\n\n-------------------{layer0_outs.shape}")
            layer1_out = self.layer1(layer0_outs.squeeze())
            # print('layer1_out shape: ',layer1_out.shape,'\n input2 shape: ',input2.shape)
            # layer2_out = self.layer2(torch.cat((layer1_out, input2.squeeze(), input3.squeeze()), dim=1))
            

        else:
            
            layer1_out = self.layer1(layer0_outs.view(1, -1))
            # layer2_out = self.layer2(torch.cat((layer1_out, input2, input3), dim=1))
            
            # layer3_out = self.layer3(torch.cat((layer2_out, input3), dim=1))
        
        # print('layer1_out.shape:', layer1_out.shape)
        # print('input2.shape: ', input2.shape)

        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        # output = self.layerOut(layer2_out)      # For single FC layer
        output = self.layerOut(layer3_out) #after adding lib func embs 
        # layer2_out = self.layer2(layer1_out)  # For two Fc layers
        # output = self.layerOut(layer2_out)

        # print('output: ', output.shape)
        return output

    def sub_forward_new(self, input1, test):
        if test:
            self.batch_size = 1
       
        # if(input1.ndim != input2.ndim):  # when input1 shape:  torch.Size([1, 1, 128])
        #     input1=input1.squeeze(0)
            
        # ip = torch.cat((input1, input2, input3), 1)
        
        # ip = ip.float()
        # with open("/Pramana/VexIR2Vec/vexIR-repo-sayan/test/model_ip.txt", "ab") as f:
        #     f.write(b"\n")
        #     np.savetxt(f, ip.cpu().detach().numpy())
        
        
        layer0_outs = self.layer0(ip)
        layer1_outs = self.layer1(layer0_outs)
        layer1_outs = self.layer11(layer1_outs)
        layer1_outs = self.layer12(layer1_outs)
        layer1_outs = self.layer13(layer1_outs)
        layer2_outs = self.layer2(layer1_outs)
        layer3_outs = self.layer3(layer2_outs)
        layer4_out = self.layer4(layer3_outs)
        output = self.layerOut(layer4_out) 

        # with open('/Pramana/VexIR2Vec/vexIR-repo-sayan/test/model_op.txt',"ab") as f:
        #     f.write(b"\n")
        #     np.savetxt(f, output.cpu().detach().numpy())
        # print(output)
        
        # if not test:
        #     layer1_out = self.layer1(layer0_outs.squeeze())
        #     # print('layer1_out shape: ',layer1_out.shape,'\n input2 shape: ',input2.shape)
        #     layer2_out = self.layer2(torch.cat((layer1_out, input2.squeeze()), dim=1))
            
        #     layer3_out = self.layer3(torch.cat((layer2_out, input3.squeeze()), dim=1))
        # else:
            
        #     layer1_out = self.layer1(attended_layer0_outs.view(1, -1))
        #     layer2_out = self.layer2(torch.cat((layer1_out, input2), dim=1))
            
        #     layer3_out = self.layer3(torch.cat((layer2_out, input3), dim=1))
        
        # print('layer1_out.shape:', layer1_out.shape)
        # print('input2.shape: ', input2.shape)

        # output = self.layerOut(layer2_out)      # For single FC layer

        # layer2_out = self.layer2(layer1_out)  # For two Fc layers
        # output = self.layerOut(layer2_out)

        # print('output: ', output)
        return output
    
    #adding input3 for libEmbed coming from line: outputs = model(embeddings, strEmbed, libEmbed) of modes.py
    def forward(self, input1, test=False):
        return self.sub_forward(input1, test)

    # Uncomment to use offline triplet loss
    # def forward(self, input1, input2, input3=None):
    #     output1 = self.sub_forward(input1)
    #     output2 = self.sub_forward(input2)
    #     if input3 is not None:
    #         output3 = self.sub_forward(input3)
    #         # triplet loss
    #         return output1, output2, output3
    #     # contrastive loss
    #     return output1, output2


class FloodFC(nn.Module):
    def __init__(self, EMBED_DIM, batch_size):
        super(FloodFC, self).__init__()
        # self.batch_size = batch_size    #Comment-out while testing
        # Consuming all super blocks
        # self.layer0 = nn.ModuleDict({f"fc_{i}": nn.Sequential(nn.Linear(EMBED_DIM, 10),
        #                                                       nn.Tanh()) for i in range(69)})

        # self.dropout0 = nn.Dropout(0.2)
        # self.bn0 = nn.BatchNorm1d(690)
        # Consuming concatenated layer0 outputs
        self.layer1 = nn.Sequential(nn.Linear(INP_DIM, 300),
                                    nn.ReLU(),
                                    # nn.Dropout(0.2),
                                    nn.BatchNorm1d(300))

        # Emitting final funtion level embeddings
        self.layerOut = nn.Linear(200, 100)

        self.layer2 = nn.Sequential(nn.Linear(300, 300),
                                    nn.ReLU(),
                                    # nn.Dropout(0.2),
                                    nn.BatchNorm1d(300))
        
        self.layer3 = nn.Sequential(nn.Linear(300, 200),
                                    nn.Sigmoid(),
                                    # nn.Dropout(0.1),
                                    nn.BatchNorm1d(200))


    def sub_forward(self, inputs):
        self.batch_size = 1   #Uncomment while testing
        layer1_out  = self.layer1(inputs)
        layer2_out  = self.layer2(layer1_out)
        # layer2_out  = self.layer2(layer2_out)
        layer3_out  = self.layer3(layer2_out)
        output = self.layerOut(layer3_out) # For single FC layer

        # layer0_outs = [self.layer0[f"fc_{i}"](input) for i, input in enumerate(inputs.view(-1, self.batch_size, 100))]
        # layer0_outs = torch.cat(layer0_outs, axis=1).view(self.batch_size, -1)
        # # print('layer0_outs before: ', layer0_outs.shape)
        # # print('layer0_outs after: ', layer0_outs.shape)

        # # layer1_inp = self.bn0(layer0_outs)
        # layer1_inp = self.bn0(self.dropout0(layer0_outs))
        # layer1_out = self.layer1(layer1_inp)

        # output = self.layerOut(layer1_out) # For single FC layer
        
        # layer2_out = self.layer2(layer1_out) # For two GFC layers
        # output = self.layerOut(layer2_out)

        # print('output: ', output.shape)
        return output

    def forward(self, input):
        return self.sub_forward(input)

        # output1 = self.sub_forward(input1)
        # output2 = self.sub_forward(input2)
        # if input3 is not None:
        #     output3 = self.sub_forward(input3)
        #     # triplet loss
        #     return output1, output2, output3
        # #contrastive loss
        # return output1, output2


class VanillaSiameseNetwork(nn.Module):
    def __init__(self, INP_DIM, OUT_DIM):
        super(VanillaSiameseNetwork, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(INP_DIM, 256),
                                 nn.ReLU(inplace=True),
                                 #  nn.Dropout(0.2),
                                 #  nn.BatchNorm1d(256),

                                 nn.Linear(256, 128),
                                 nn.ReLU(inplace=True),
                                 #  nn.Dropout(0.2),
                                 #  nn.BatchNorm1d(128),

                                 nn.Linear(128, OUT_DIM))
        self.fc2 = nn.Linear(OUT_DIM, OUT_DIM)
        self.olayer = nn.Sequential(nn.Linear(OUT_DIM, 1),
                                    nn.Sigmoid())

    def forward(self, input1, input2):
        input1 = self.fc1(input1)
        input2 = self.fc1(input2)
        distVec = torch.abs(torch.sub(input1, input2))
        output = self.fc2(distVec)
        preds = self.olayer(output)
        return preds


