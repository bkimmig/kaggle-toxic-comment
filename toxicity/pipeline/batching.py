
from .. import data_dir, resource_dir
import pandas as pd
import numpy as np
import os


class BootstrapBatcher(object):
    
    def __init__(self, data_ids):
        self.data_ids = np.asarray(data_ids)
    
    def get_ids(self, batch_size):
        indexes = np.random.randint(len(self.data_ids), size=(batch_size,))
        return self.data_ids[indexes]
    
    def iterations_per_epoch(self, batch_size):
        return len(self.data_ids)//batch_size

class ShuffledList(object):
    
    def __init__(self, input_list):
        self.input_list = input_list
        self.regenerate_permutation()
    
    def regenerate_permutation(self):
        self.permutation = np.random.random(len(self.input_list)).argsort()
        self.cur_idx = 0
    
    def next(self):
        next_elem = self.input_list[self.permutation[self.cur_idx]]
        self.cur_idx += 1
        if self.cur_idx >= len(self.input_list):
            self.regenerate_permutation()
        return next_elem
    
    def __len__(self):
        return len(self.input_list)


class PermutationBatcher(object):
    
    def __init__(self, data_ids):
        self.shuffled_list = ShuffledList(data_ids)
    
    def get_ids(self, batch_size):
        return [shuffler.next() for i in range(batch_size)]
    
    def iterations_per_epoch(self, batch_size):
        return len(self.shuffled_list)//batch_size


class StratifiedBatcher(object):
    current_source_index = 0
    
    def __init__(
            self,
            source_strata,
            epoch_behavior="min",
    ):
        shufflers = [ShuffledList(source) for source in source_strata]
        n_examples = [len(shuff) for shuff in shufflers]
        if epoch_behavior == "min":
            n_eff = min(n_examples)
        elif epoch_behavior == "max":
            n_eff = max(n_examples)
        elif epoch_behavior == "mean":
            n_eff = int(np.mean(n_examples))
        else:
            raise ValueError("epoch_behavior parameter value not recognized")
        self.n_eff = n_eff
        self.shufflers = shufflers
    
    def get_ids(self, batch_size):
        n_sources = len(self.shufflers)
        n_per = batch_size//n_sources
        batch_list = []
        for i in range(batch_size):
            shuffler = self.shufflers[self.current_source_index]
            self.current_source_index = (current_source_index + 1) % n_sources
            batch_list.append(shuffler.next())
        return batch_list
    
    def iterations_per_epoch(self, batch_size):
        return self.n_eff//batch_size
