# from transformers import AutoTokenizer, BertModel
# import torch

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state

from transformers import AutoTokenizer, AutoModel
import torch

# Load the tokenizer and model
model_directory = "/Pramana/VexIR2Vec/Source_Binary/trained-mlm-insts50000-e100-bs64-cpu"
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModel.from_pretrained(model_directory)


# Path to your instructions file
input_file_path = "/Pramana/VexIR2Vec/Source_Binary/Dataset/concat-vex-ll-ir-50000.txt"  # Replace with your file path

# Path to save contextual embeddings for each instruction
output_file_path = "/Pramana/VexIR2Vec/Source_Binary/Dataset/tokens/concat-vex-ll-ir-50000_token_emb.txt"  # Replace with desired output file path


with open(input_file_path, 'r', encoding='utf-8') as file:
    instructions = file.readlines()

# Initialize dictionary to store token embeddings
token_embeddings_all = {}

# Get token embeddings for each instruction
for instruction in instructions:
    
    # Tokenize the input text
    inputs = tokenizer(instruction, return_tensors="pt")

    # Pass the input through the model to get the output
    with torch.no_grad():
        outputs = model(**inputs)

    # Retrieve the hidden states from the output
    last_hidden_states = outputs.last_hidden_state

    # Access token embeddings (last layer hidden states)
    # `last_hidden_states` shape: (batch_size, sequence_length, hidden_size)
    token_embeddings = last_hidden_states[0]  # Assuming batch_size=1

    # token_embeddings contains the embeddings for each token in the input text
    # print(type(token_embeddings)) #<class 'torch.Tensor'>
    # print(token_embeddings.size()) #torch.Size([n, 768])

    # Get the tokens and their corresponding IDs
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())

    # Store unique tokens and their embeddings
    for i, token in enumerate(tokens):
        if token not in token_embeddings_all:
            token_embeddings_all[token] = token_embeddings[i].tolist()



    # Display each token and its corresponding embedding
    # for i, token in enumerate(tokens):
    #     print(f"Token: {token}")
    #     print(f"Embedding: {token_embeddings[i]}")
    #     print("\n")

# Save unique tokens and their embeddings to a file
with open(output_file_path, 'w', encoding='utf-8') as out_file:
    for token, embedding in token_embeddings_all.items():
        embedding_str = " ".join(map(str, embedding))
        out_file.write(f"{token} {embedding_str}\n")


