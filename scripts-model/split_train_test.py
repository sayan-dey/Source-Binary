
#For splitting the whose csv into train data and test (inference) data

import pandas as pd
import numpy as np
import json
import random

# Function for splitting a csv into train test data
def split_csv_train_test():
    
    # Load the CSV file
    data_df = pd.read_csv('/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_data_infer.csv')

    # Shuffle the DataFrame
    # data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate the split indices
    split_idx = int(len(data_df) * 0.75)

    # Split the DataFrame into two parts
    df_part1 = data_df.iloc[:split_idx]
    df_part2 = data_df.iloc[split_idx:]

    # Save the two parts into two separate files
    df_part1.to_csv('/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_data_infer_75perc.csv', index=False)
    df_part2.to_csv('/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_data_infer_25perc.csv', index=False)


# Function for segregating the classes itself beforehand into train and test
def split_classes_train_test():
    
    # Specify the path to your JSON file containing required files to consider
    json_file_path = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/C_filedict.json'
    # Read the JSON file and load its contents as a dictionary
    with open(json_file_path, 'r') as json_file:
        data_dict = json.load(json_file)
    
    classes = list(data_dict.keys())
    
    # Shuffle the list to ensure random division
    random.shuffle(classes)

    # Determine the split index
    split_index = int(len(classes) * 0.80)

    # Split the list into two parts
    classes_train = classes[:split_index] #classes to be put in train set
    classes_test = classes[split_index:] #classes to be put in test set
    
    # Create a dictionary to store both lists
    data_to_store = {
        'classes_train': classes_train,
        'classes_test': classes_test
    }

    # File name to store the lists
    file_name = '/Pramana/VexIR2Vec/Source_Binary/COFO-Dataset/train_test_split_classes.json'

    # Write the dictionary to a file
    with open(file_name, 'w') as f:
        json.dump(data_to_store, f)

    # Print confirmation
    print(f"Lists stored in {file_name}")


if __name__ == "__main__":
    split_classes_train_test()
    
    
    
    