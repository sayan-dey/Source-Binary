import os, sys
import angr
import gensim
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
import random
import json

# sys.path.append("/Pramana/VexIR2Vec/Source_Binary/scripts/scripts-model")

from utils import SEED, ADJMAT_SIZE, determine_vocab_size

random.seed(SEED)
np.random.seed(SEED)

# Step 1: Extract basic blocks
def extract_basic_blocks(binary_path):
    project = angr.Project(binary_path, load_options={"auto_load_libs": False})
    cfg = project.analyses.CFGFast()
    basic_blocks = []

    for func in cfg.kb.functions.values():
        if func.name == "main":
            for block in func.blocks:
                instructions = []
                for stmt in block.vex.statements:
                    stmt_str = str(stmt).lower()
                    instructions.append(stmt_str)
                basic_blocks.append(instructions)
    

    # print(basic_blocks)
    # exit()
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
        '''
        Ensure individual tokens are processed
        sp.encode method processes an entire block as a single string
        ensuring individual tokens are returned
        '''
        tokens = sp.encode(" ".join(block), out_type=str)
        tokenized_blocks.append(tokens)

    return tokenized_blocks


# Step 4: Extract adjacency matrix for the main function
def extract_basic_block_adjacency(binary_path):
    project = angr.Project(binary_path, load_options={"auto_load_libs": False})
    cfg = project.analyses.CFGFast()
    
    main_func = cfg.kb.functions.function(name="main")
    if main_func is None:
        raise ValueError("Main function not found!")
    
    basic_blocks = list(main_func.blocks)
    block_to_id = {block.addr: idx for idx, block in enumerate(basic_blocks)}
    num_blocks = len(basic_blocks)
    adjacency_matrix = np.zeros((ADJMAT_SIZE, ADJMAT_SIZE), dtype=np.int32)
    
    for block in basic_blocks:
        # Skip blocks not in the graph
        if block not in cfg.graph:
            continue
        
        src_id = block_to_id[block.addr]
        for successor in cfg.graph.successors(block):
            # Ensure successor is part of the adjacency matrix
            if successor.addr in block_to_id:
                dest_id = block_to_id[successor.addr]
                if src_id < ADJMAT_SIZE and dest_id < ADJMAT_SIZE:
                    adjacency_matrix[src_id, dest_id] = 1

    return adjacency_matrix

# Step 5: Train Skip-gram model
def train_skipgram(tokenized_blocks, embedding_dim=128):
    sentences = [token for block in tokenized_blocks for token in block]
    skipgram_model = gensim.models.Word2Vec(
        sentences,
        vector_size=embedding_dim,
        window=5,
        min_count=1,
        sg=1,
        workers=4
    )
    return skipgram_model

# Compute basic block embeddings
def compute_basic_block_embeddings(skipgram_model, tokenized_blocks):
    embeddings = []
    for block in tokenized_blocks:
        # Flatten block (a list of lists) into a single list of tokens
        flat_block = [token for sublist in block for token in sublist]
        token_embeddings = [skipgram_model.wv[token] for token in flat_block if token in skipgram_model.wv]
        if token_embeddings:
            # Average the embeddings for the basic block
            block_embedding = np.mean(token_embeddings, axis=0)
            embeddings.append(block_embedding)
        else:
            # Handle blocks without embeddings (optional, e.g., append a zero vector)
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


# Process all binaries and train the GNN
def process_binaries(directory_path, vocab_size=5000, embedding_dim=128, hidden_dim=64, output_dim=32, epochs=50):
    all_basic_blocks = []

    binary_paths = []
    
    json_file_path = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/train_test_split_classes.json'
    # Read the JSON file and load its contents as a dictionary
    with open(json_file_path, 'r') as json_file:
        data_dict = json.load(json_file)
    
    with open('/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_masterdict_with_8_pass_seqs.json', 'r') as json_file:
        master_dict = json.load(json_file)

    # Populate binary_paths
    for dirpath, _, filenames in os.walk(directory_path):
        
        # if len(binary_paths) > 50:
        #     break

        if dirpath.endswith('submissions'):
            c_directory = os.path.join(dirpath, 'C')
            if os.path.exists(c_directory) and c_directory in data_dict["classes_train"]:
                class_dict = master_dict[c_directory]
                for _ in range(0, 100):  # Adjust the range as needed
                    v2v_config = random.choice(list(class_dict['v2v'].keys()))
                    filename = random.choice(class_dict['v2v'][v2v_config])
                    filename_without_extension = os.path.splitext(filename)[0]
                    binary_filename = filename_without_extension + '.out'
                    binary_filepath = os.path.join(c_directory, v2v_config, 'unstripped', binary_filename)
                    if os.path.exists(binary_filepath):
                        binary_paths.append(binary_filepath)

    # Step 1: Extract basic blocks from all binaries
    binary_to_blocks = {}
    for binary_path in binary_paths:
        try:
            basic_blocks = extract_basic_blocks(binary_path)
            binary_to_blocks[binary_path] = basic_blocks
            all_basic_blocks.extend(basic_blocks)
        except Exception as e:
            print(f"Error processing binary {binary_path}: {e}")

    # Step 2: Train SentencePiece tokenizer on all binaries
    vocab_size = determine_vocab_size(all_basic_blocks, default_vocab_size=5000)
    sp_model_path = train_sentencepiece(all_basic_blocks, vocab_size=vocab_size)
    

    # Step 3: Tokenize basic blocks for all binaries
    all_tokenized_blocks = []
    binary_to_tokenized_blocks = {}
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)

    for binary_path, basic_blocks in binary_to_blocks.items():
        tokenized_blocks = []
        for block in basic_blocks:
            tokenized_blocks.append(sp.encode(block, out_type=str))
        
        # print(f"Sample Basic Blocks: {basic_blocks[:1]}")
        # print(f"Tokenized Blocks: {tokenized_blocks[:1]}")

        # exit()

        binary_to_tokenized_blocks[binary_path] = tokenized_blocks
        all_tokenized_blocks.extend(tokenized_blocks)

    # Step 4: Train Skip-gram model on all tokenized blocks
    sentences = [token for block in all_tokenized_blocks for token in block]
    skipgram_model = gensim.models.Word2Vec(
        sentences,
        vector_size=embedding_dim,
        window=5,
        min_count=1,
        sg=1,
        workers=4
    )

    # Step 5: Prepare data for GNN
    data_list = []
    for binary_path, tokenized_blocks in binary_to_tokenized_blocks.items():
        adj_matrix = extract_basic_block_adjacency(binary_path)
        block_embeddings = compute_basic_block_embeddings(skipgram_model, tokenized_blocks)

        # Convert to PyTorch Geometric data
        block_embeddings_tensor = torch.tensor(block_embeddings, dtype=torch.float)
        edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
        data = Data(x=block_embeddings_tensor, edge_index=edge_index)
        data_list.append(data)

    # Step 6: Train the GNN
    gnn_model = GNNModel(input_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
    loader = DataLoader(data_list, batch_size=1, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            out = gnn_model(data)
            loss = torch.nn.functional.mse_loss(out, torch.zeros_like(out))  # Dummy loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# Root directory path
directory_path = "/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset"
process_binaries(directory_path)
