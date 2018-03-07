
from . import data_dir, resource_dir
import pandas as pd
import numpy as np
import os

_train_x = None
_train_y = None

def get_dataset(
        dataset_name,
        train_fold_seed=1234,
        cache_train=True,
):
    """
Utility for reading in different datasets and subsetting them into folds in a way
that is repeatable between sessions.

dataset_name can be either "lb", "train" or some sort of training fold specification.  "train" will load all of the training data. "lb"  will load the kaggle provided "test.csv". A training fold specification will load a consistent random subset of the data dependent on the train_fold_seed value and the number of folds and fold_index.

Training fold names should look like

"train-n-i"

where n is the total number of folds and i is the index of the desired fold. If you want to load all of the training data If you want to load all the folds that are not in a particular fold then you can prepend a "c" to the fold index. The "c" stands for complement.

For example to load a train/validation split with 20% validation set you could do the following.

train_x, train_y = get_dataset("train-5-c0")
validation_x, validation_y = get_dataset("train-5-0")

The first call to get_dataset loads everything except the 0'th fold of data for use as training data and the second call loads just exactly the 0'th fold which can then be used to estimate model quality since it wasn't used for training.

If cache_train is true then the training data is loaded only the first time this function is called and is cached thereafter. This will speed up calls to the function but will mean there are 2 sets of the training data floating around in memory which may be a problem if you have a low memory machine.
    """
    data_fold_seed = 1234
    if dataset_name == "lb":
        x_data =  pd.read_csv(os.path.join(data_dir, "raw", "test.csv")).set_index("id")
        y_data = None
        return x_data, y_data
    
    elif "train" in dataset_name:
        global _train_x
        global _train_y
        if _train_x is None:
            train_data = pd.read_csv(os.path.join(data_dir, "raw", "train.csv"))
            train_data = train_data.set_index("id")
            
            train_x = train_data[["comment_text"]].copy()
            train_y = train_data.drop("comment_text", axis=1)
            
            if cache_train:
                #cache the training data in the module so we don't have to reload it
                _train_x = train_x
                _train_y = train_y
        else:
             train_x = _train_x
             train_y = _train_y
        
        if "train" == dataset_name:
            return train_x, train_y
        
        n_folds, fold_index = dataset_name.split("-")[1:]
        
        #generate a pseudo random permutation of the training data
        r_state = np.random.RandomState(seed=train_fold_seed)
        permutation = r_state.permutation(len(_train_x))
        
        complement = False
        if fold_index[0] == "c":
            complement = True
            fold_index = fold_index[1:]
        
        n_folds = int(n_folds)
        fold_index = int(fold_index)
        
        mask = (permutation % n_folds) == fold_index
        if complement:
            mask = np.logical_not(mask)
        return train_x[mask].copy(), train_y[mask].copy() 
    else:
        raise ValueError("dataset name {} not understood".format(dataset_name))
