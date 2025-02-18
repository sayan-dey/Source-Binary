
import random
import numpy as np

SEED=1004
ADJMAT_SIZE=110

random.seed(SEED)
np.random.seed(SEED)

def pad_and_reshape_array(x, sz, INP_DIM):
    shape = x.shape[0]
    if shape <= sz:
        reqd = sz-shape
        padded_array = np.zeros((reqd, INP_DIM))
        tmp_arr = np.concatenate((x, padded_array), axis=0)
        # return np.concatenate((x, padded_array), axis=0)

    else:
        if shape > sz:
            tmp_arr = x[:sz, :]
    # return tmp_arr[np.newaxis, ...]
    # Uncomment to use for clr train data
    return tmp_arr


def determine_vocab_size(basic_blocks, default_vocab_size=5000, min_vocab_size=50):
    # Count unique symbols in the dataset
    unique_symbols = set()
    for blocks in basic_blocks:
        for block in blocks:
            unique_symbols.update(block)
    return max(min(default_vocab_size, len(unique_symbols) + 5), min_vocab_size)
