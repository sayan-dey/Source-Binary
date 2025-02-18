import os
import gensim
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
import random
import re
import json

from utils import SEED, ADJMAT_SIZE, determine_vocab_size

random.seed(SEED)
np.random.seed(SEED)

# Step 1: Extract basic blocks from .ll files
def extract_basic_blocks_from_ll(ll_path):
    with open(ll_path, "r") as file:
        lines = file.readlines()
    
    basic_blocks = []
    current_block = []
    for line in lines:
        stripped_line = line.strip()

        # print(stripped_line)
        # exit()
        if stripped_line.startswith(";") or not stripped_line:
            continue
        if stripped_line.endswith(":"):  # Block label
            if current_block:
                basic_blocks.append(current_block)
                current_block = []
        current_block.append(stripped_line)
    
    if current_block:  # Add the last block
        basic_blocks.append(current_block)
    
    return basic_blocks

# Step 2: Train SentencePiece tokenizer
def train_sentencepiece(basic_blocks_list, model_prefix="sp_model", vocab_size=5000):
    with open("basic_blocks.txt", "w") as f:
        for blocks in basic_blocks_list:
            for block in blocks:
                f.write(" ".join(block) + "\n")
    
    spm.SentencePieceTrainer.train(
        input="basic_blocks.txt",
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe"
    )
    return f"{model_prefix}.model"

# Step 3: Tokenize using SentencePiece
def tokenize_basic_blocks(basic_blocks, sp_model_path):
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    tokenized_blocks = []
    for block in basic_blocks:
        tokens = sp.encode(" ".join(block), out_type=str)
        tokenized_blocks.append(tokens)
    return tokenized_blocks

# Step 4: Extract adjacency matrix for the main function
def extract_basic_block_adjacency_from_ll(ll_path):
    with open(ll_path, "r") as file:
        lines = file.readlines()

    block_labels = {}
    adjacency_matrix = np.zeros((ADJMAT_SIZE, ADJMAT_SIZE), dtype=np.int32)
    current_block = None

    for idx, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.endswith(":"):
            current_block = stripped_line[:-1]
            block_labels[current_block] = len(block_labels)

        if "br" in stripped_line or "jmp" in stripped_line:  # Check for branches
            targets = re.findall(r"%[a-zA-Z0-9_]+", stripped_line)
            for target in targets:
                if target in block_labels:
                    src_id = block_labels.get(current_block, -1)
                    dest_id = block_labels.get(target, -1)
                    if src_id < ADJMAT_SIZE and dest_id < ADJMAT_SIZE:
                        adjacency_matrix[src_id, dest_id] = 1

    return adjacency_matrix

# Compute basic block embeddings
def compute_basic_block_embeddings(skipgram_model, tokenized_blocks):
    embeddings = []
    for block in tokenized_blocks:
        token_embeddings = [skipgram_model.wv[token] for token in block if token in skipgram_model.wv]
        if token_embeddings:
            block_embedding = np.mean(token_embeddings, axis=0)
            embeddings.append(block_embedding)
        else:
            embeddings.append(np.zeros(skipgram_model.vector_size))
    return embeddings

# GNN model definition
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Process all LLVM IR files and train the GNN
def process_ll_files(directory_path, vocab_size=5000, embedding_dim=128, hidden_dim=64, output_dim=32, epochs=50):
    
    all_basic_blocks = []
    ll_paths = []

    json_file_path = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/train_test_split_classes.json'
    # Read the JSON file and load its contents as a dictionary
    with open(json_file_path, 'r') as json_file:
        data_dict = json.load(json_file)
    
    with open('/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_masterdict_with_8_pass_seqs.json', 'r') as json_file:
        master_dict = json.load(json_file)

    
    for dirpath, _, filenames in os.walk(directory_path):

        if dirpath.endswith('submissions'):
            c_directory = os.path.join(dirpath, 'C')
            if os.path.exists(c_directory) and c_directory in data_dict["classes_train"]:
                class_dict = master_dict[c_directory]

                for _ in range(0,100): # Adjust the range as needed
                    i2v_config = random.choice(list(class_dict['i2v'].keys()))   
                    # print(i2v_config)
                    filename = random.choice(class_dict['i2v'][i2v_config])
                    filename_without_extension = os.path.splitext(filename)[0]
                    ll_filename= filename_without_extension + '.ll'
                    ll_filepath = os.path.join(c_directory, i2v_config, 'll-files', ll_filename)
                    if os.path.exists(ll_filepath):
                        ll_paths.append(ll_filepath)


    ll_to_blocks = {}
    for ll_path in ll_paths:
        try:
            basic_blocks = extract_basic_blocks_from_ll(ll_path)
            # print(basic_blocks)
            # exit()
            ll_to_blocks[ll_path] = basic_blocks
            all_basic_blocks.extend(basic_blocks)
        except Exception as e:
            print(f"Error processing .ll file {ll_path}: {e}")

    vocab_size = determine_vocab_size(all_basic_blocks, default_vocab_size=5000)
    sp_model_path = train_sentencepiece(all_basic_blocks, vocab_size=vocab_size)

    all_tokenized_blocks = []
    ll_to_tokenized_blocks = {}
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)

    for ll_path, basic_blocks in ll_to_blocks.items():
        tokenized_blocks = tokenize_basic_blocks(basic_blocks, sp_model_path)
        ll_to_tokenized_blocks[ll_path] = tokenized_blocks
        all_tokenized_blocks.extend(tokenized_blocks)

    skipgram_model = gensim.models.Word2Vec(
        [token for block in all_tokenized_blocks for token in block],
        vector_size=embedding_dim,
        window=5,
        min_count=1,
        sg=1,
        workers=4
    )

    data_list = []
    for ll_path, tokenized_blocks in ll_to_tokenized_blocks.items():
        adj_matrix = extract_basic_block_adjacency_from_ll(ll_path)
        block_embeddings = compute_basic_block_embeddings(skipgram_model, tokenized_blocks)

        block_embeddings_tensor = torch.tensor(block_embeddings, dtype=torch.float)
        edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
        data = Data(x=block_embeddings_tensor, edge_index=edge_index)
        data_list.append(data)

    gnn_model = GNNModel(input_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
    loader = DataLoader(data_list, batch_size=1, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            out = gnn_model(data)
            loss = torch.nn.functional.mse_loss(out, torch.zeros_like(out))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# Root directory path
directory_path = "/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset"
process_ll_files(directory_path)
