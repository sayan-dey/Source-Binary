
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)

from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning import distances, miners, reducers, testers
from losses import *
from argparse import ArgumentParser
import torch.nn.functional as F
from pytorch_metric_learning.regularizers import LpRegularizer

le = LabelEncoder()

def replace_pipe_with_space(input_string):
    return input_string.replace("|", " ")

def truncate_sent(lst): #for taking first 256 and last 256 elemnts of a list if size >512; else unmodified
    if len(lst) > 512:
        return lst[:256] + lst[-256:]
    else:
        return lst
    
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))




def main(args):

    #check for GPU access

    # If there's a GPU available...
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0)) #can use multiple gpus also...see vexir training script

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    print(f"device to be used: {device}")

    device = torch.device("cpu") #for debugging purpose

    #collecting vocab tokens

    i2v_vocab='/Pramana/VexIR2Vec/Source_Binary/IR2Vec/vocabulary/seedEmbeddingVocab-llvm14.txt'
    v2v_vocab='/Pramana/VexIR2Vec/checkpoint/ckpt_3M_900E_128D_0.0002LR_adam/seedEmbedding_3M_900E_128D_0.0002LR_adam'

    vocab_tokens=[]

    vfile = open(i2v_vocab, "r")
    for line in vfile:
        opc = line.split(":")[0].lower()
        vocab_tokens.append(opc)

    vfile = open(v2v_vocab, "r")
    for line in vfile:
        opc = line.split(":")[0].lower()
        vocab_tokens.append(opc)
        
    vocab_tokens.append("<INST>".lower())

    # Load the dataset into a pandas dataframe.
    df = pd.read_csv("/Pramana/VexIR2Vec/Source_Binary/Dataset/training_data/669_670.csv")

    keys = df.key.values
    le.fit(keys)
    classes = le.classes_
    uniq_key_cnt = classes.shape[0]

    embeds = df.embed.values

    print('Number of training datapoints: {:,}\n'.format(df.shape[0]))
    # print(df.key) #working fine

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    tokenizer.add_tokens(vocab_tokens) #so that none of the tokens go OOV for tokenizer


    '''
    # # Print the original embed
    # print(' Original: ', embeds[386])

    # modified_embed = replace_pipe_with_space(embeds[386])

    # print('Modified embed: ', modified_embed)

    # # Print the embed split into tokens.
    # print('Tokenized: ', tokenizer.tokenize(modified_embed))

    # # Print the embed mapped to token ids.
    # print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(modified_embed)))


    #The transformers library provides a helpful `encode` function which will handle
    #most of the parsing and data prep steps for us

    #Finding max length
    max_len = 0 #this is for padding/truncating...can adjust the length later

    input_ids = []

    # For every sentence...
    for embed in embeds:

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        embed = replace_pipe_with_space(embed)
        embed = tokenizer.encode(embed, add_special_tokens=True)
        
        # print(len(embed))
        # Update the maximum sentence length.
        max_len = max(max_len, len(embed))
        embed = truncate_sent(embed) # For truncating to 512 tokens
        # embed = torch.tensor(embed) # Convert to tensor
        # print(embed)
        input_ids.append(embed)
        
    print('Max embed length: ', max_len)

    '''

    # Tokenize all of the embeds and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every embed...
    for embed in embeds:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        
        embed = replace_pipe_with_space(embed)
        encoded_dict = tokenizer.encode_plus(
                            embed,                      # Sentence to encode.
                            add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                            max_length = 512,           # Pad & truncate all sentences.
                            padding='max_length',
                            truncation = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])



    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    keys = le.transform(keys)
    keys = torch.tensor(keys)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', embeds[0])
    print('Token IDs:', input_ids[0])
    print('attention mask: ', attention_masks[0])
    print('Label: ', keys[0])

    print(f"input_ids.shape: {input_ids.shape}")
    print(f"attention_masks.shape: {attention_masks.shape}")
    print(f"keys.shape: {keys.shape}")

    #Training & Validation Split
    #Divide up our training set to use 90% for training and 10% for validation.

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, keys)

    # Create a 90-10 train-validation split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print()
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    print()

    #We’ll also create an iterator for our dataset using the torch DataLoader
    #class. This helps save on memory during training because, unlike a for
    #loop, with an iterator the entire dataset does not need to be loaded into memory.


    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.
    batch_size = args.batch_size
    print("batch size: ", batch_size)

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )


    # Loading BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top....#replace with your model

    '''
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = uniq_key_cnt, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )  
    '''
    # model_path '/Pramana/VexIR2Vec/Source_Binary/codebert-model' #pml_loss was decreasing

    model_path = '/Pramana/VexIR2Vec/Source_Binary/trained-mlm-insts50000-e100-bs64-cpu'


    '''
    model = AutoModelForMaskedLM.from_pretrained(
                model_path,
                from_tf=bool(".ckpt" in model_path),
                config=config, #see run_mlm.py for things from here
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            )
    '''
    # model = AutoModelForMaskedLM.from_pretrained(
    #             model_path,
    #             from_tf=bool(".ckpt" in model_path),
    #         )

    model = BertModel.from_pretrained(model_path, num_labels = uniq_key_cnt)

    # Tell pytorch to run this model on the GPU.
    model.to(device)

    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        

    #Batch size: 32
    #Learning rate (Adam): 2e-5
    #Number of epochs: 4
    #eps = 1e-8,  very small number to prevent any division by zero in the implementation

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"

    #earlier lr was 2e-5
    optimizer = AdamW(model.parameters(),
                    lr = args.lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )


    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = args.epochs

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)



    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    #----------Had cut from here--------------------------

    # For each epoch...
    for epoch_i in range(0, epochs):

    # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()
        
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            
        
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)


            model.zero_grad()
            
            
            # print(b_input_ids.shape)
            # print(b_input_mask.shape)
            # print(b_labels.shape)

            # b_labels = b_labels.unsqueeze(0) #i added

            # Perform a forward pass (evaluate the model on this training batch).
        
            result = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            predictions = result.last_hidden_state
            # print(predictions.shape)

            margin=args.margin
            # Reshape the tensor to shape (batch_size, embedding_size)
            reshaped_predictions = predictions.view(predictions.size(0), -1)
            # print(reshaped_predictions.dtype) #torch.float32
            hard_pairs = miners.TripletMarginMiner(margin, type_of_triplets="semihard")(reshaped_predictions, b_labels)
            loss = pml_losses.TripletMarginLoss(margin=margin, embedding_regularizer = LpRegularizer())(reshaped_predictions, b_labels, hard_pairs)
            # print(loss)
            # exit()

            '''
            # Get the prediction for each sample
            predicted_labels = predictions[:, 0, :]
            # print(predicted_labels.shape)

            # Assuming you want to convert the predicted_labels tensor to a numpy array
            predicted_labels = predicted_labels.cpu().detach().numpy()

            # Convert predicted_labels to a list of 16 labels
            predicted_labels_list = [torch.argmax(torch.tensor(label)).item() for label in predicted_labels]

            print(f"Predicted Labels: {predicted_labels_list}")
            print(f"Actual Labels: {b_labels.tolist()}")
            
            # Convert predicted_labels_list to a tensor
            predicted_labels = torch.tensor(predicted_labels_list, dtype=torch.float32, requires_grad=True)

            b_labels = b_labels.float()

            # Calculate the mean squared error (MSE) loss
            loss = F.mse_loss(predicted_labels, b_labels) #loss not decreasing

            '''
            
            print(f"Loss: {loss.item()}\n")

            # loss = result.loss
            # logits = result.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.

            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Training Time': training_time,
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))





if __name__ == "__main__":

    parser = ArgumentParser(description='Framework for binary-source similarity.')
    parser.add_argument('-l', '--loss', required=True, help='Loss to be used.') # 'cont', 'trp' (for both offline & online)
    parser.add_argument('-lr', '--lr', required=True, type=float, help='Learning rate to be used.')
    parser.add_argument('-b', '--beta', type=float, default=0.9, help='beta1 to be used in Adam.')
    parser.add_argument('-bs', '--batch_size', type=int, required=True)
    parser.add_argument('-e', '--epochs', type=int, required=True)
    parser.add_argument('-m', '--margin', type=float, required=True)
    parser.add_argument('-cfg', '--use_cfg', type=bool, default=False, help='Use CFG if set')

    args = parser.parse_args()
    main(args)

