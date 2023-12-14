def extract_unique_tokens(input_file, output_file):
    unique_tokens = set()  # To store unique tokens

    # Read the dataset file
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        # Extract tokens from each line
        for line in lines:
            tokens = line.strip().split()  # Split tokens by spaces
            unique_tokens.update(tokens)   # Add tokens to the set

    # Write unique tokens to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        for token in sorted(unique_tokens):  # Sort tokens alphabetically
            file.write(token + '\n')

# Replace 'dataset.txt' with the path to your dataset file
input_file_path = '/Pramana/VexIR2Vec/Source_Binary/Dataset/concat-vex-ll-ir-50000.txt'

# Replace 'unique_tokens.txt' with the desired output file name
output_file_path = '/Pramana/VexIR2Vec/Source_Binary/Dataset/tokens/concat-vex-ll-ir_tokens_50000.txt'

# Extract unique tokens and store in the output file
extract_unique_tokens(input_file_path, output_file_path)
