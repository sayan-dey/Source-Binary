import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import INP_DIM, NUM_SB, SEED, OUT_DIM, CNN_INP_DIM, NUM_WALKS

random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, output_size, activation='relu', use_batchnorm=False):
        super(FullyConnectedLayer, self).__init__()
        
        self.fc = nn.Linear(input_size, output_size)
        self.activation = self.get_activation(activation)
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(output_size)

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'silu':
            return nn.SiLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise NotImplementedError("Activation function '{}' not supported".format(activation))

    def forward(self, x):
        x = self.fc(x)
        if self.use_batchnorm:
            x = self.batchnorm(x)
        x = self.activation(x)
        return x
    
class GlobalAttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(GlobalAttentionLayer, self).__init__()
        
        self.attention_fc = nn.Linear(input_size, 1)

    def forward(self, x):
        attention_logits = self.attention_fc(x)
        return attention_logits

class FCNNWithAttention(nn.Module):
    def __init__(self, EMBED_DIM, config):
        super(FCNNWithAttention, self).__init__()
        self.embed_dim = EMBED_DIM
        self.batch_size = config['batch_size']
        num_layers = config['num_layers']
        hidden_units = config['hidden']
        self.drop_units = config['drop_units']
        activation = config['activation']
        
        num_O_layers = config['num_O_layers']
        num_T_layers = num_O_layers # config['num_T_layers']
        num_A_layers = num_O_layers # config['num_A_layers']
        num_str_layers = num_O_layers # config['num_A_layers']
        num_lib_layers = num_O_layers # config['num_A_layers']
        
        
        self.opc_fc_layers = self.create_fc_layers(EMBED_DIM, num_O_layers, activation, use_batchnorm=True)
        self.ty_fc_layers = self.create_fc_layers(EMBED_DIM, num_T_layers, activation, use_batchnorm=True)
        self.arg_fc_layers = self.create_fc_layers(EMBED_DIM, num_A_layers, activation, use_batchnorm=True)
        self.str_fc_layers = self.create_fc_layers(100, num_str_layers, activation, use_batchnorm=True)
        self.lib_fc_layers = self.create_fc_layers(100, num_lib_layers, activation, use_batchnorm=True)

        self.global_attention = GlobalAttentionLayer(EMBED_DIM)
        
        self.concat_layer = config['concat_layer']
        # self.additional_fc_layers = self.create_fc_layers(EMBED_DIM + 200, num_layers, activation, use_batchnorm=True, hidden_units=hidden_units)
        self.additional_fc_layers = self.create_fc_layers(EMBED_DIM, num_layers, activation, use_batchnorm=True, hidden_units=hidden_units, concat_layer=self.concat_layer)
        
        self.output_layer = nn.Linear(hidden_units[-1], OUT_DIM)

    def create_fc_layers(self, input_size, num_layers, activation, use_batchnorm, hidden_units=None, concat_layer=-1):
        layers = []
        for i in range(num_layers):
            if hidden_units is not None:
                assert i < len(hidden_units)
                output_size = hidden_units[i]
            else:
                output_size = self.embed_dim
            if i == concat_layer:
                input_size += 200
            # output_size = hidden_units[i] if hidden_units and i < len(hidden_units) else 128
            layers.append(FullyConnectedLayer(input_size, output_size, activation, use_batchnorm))
            input_size = output_size  # Update input size for next layer
        return nn.ModuleList(layers)

    def forward(self, opc, ty, arg, strEmbed, libEmbed, test=False):
        if test:
            self.batch_size = 1
            
        opc = torch.nn.functional.normalize(opc, p=2)
        ty = torch.nn.functional.normalize(ty, p=2)
        arg = torch.nn.functional.normalize(arg, p=2)
        strEmbed = torch.nn.functional.normalize(strEmbed, p=2)
        libEmbed = torch.nn.functional.normalize(libEmbed, p=2)

        opc_out = self.process_input(opc, self.opc_fc_layers)
        ty_out = self.process_input(ty, self.ty_fc_layers)
        arg_out = self.process_input(arg, self.arg_fc_layers)
        str_out = self.process_input(strEmbed, self.str_fc_layers)
        lib_out = self.process_input(libEmbed, self.lib_fc_layers)
        
        # Compute attention logits for each context vector
        attention_logits_opc = self.global_attention(opc_out)
        attention_logits_ty = self.global_attention(ty_out)
        attention_logits_arg = self.global_attention(arg_out)
        attention_logits_str = self.global_attention(str_out)
        attention_logits_lib = self.global_attention(lib_out)
        
        # Compute attention-weighted representations
        attention_weights = F.softmax(torch.stack([attention_logits_opc, attention_logits_ty, attention_logits_arg, attention_logits_str, attention_logits_lib], dim=0), dim=0)

        # Aggregate attended representations
        aggregated_vector = (attention_weights[0] * opc_out) + (attention_weights[1] * ty_out) + (attention_weights[2] * arg_out) + (attention_weights[3] * str_out) + (attention_weights[4] * lib_out)

        output = aggregated_vector
        
        # Check if the current layer is the concatenation layer and concatenate strEmbed, libEmbed with previous layer's output
        for i, layer in enumerate(self.additional_fc_layers):
            if i == self.concat_layer:
                output = torch.cat((output, strEmbed, libEmbed), dim=1)
            output = layer(output)
            output = F.dropout(output, p=self.drop_units[i], training=self.training)
            
        output = self.output_layer(output)
        return output, attention_weights

    def process_input(self, x, fc_layers):
        for layer in fc_layers: 
            x = layer(x)
        return x
