'''
Contains function to initialize word embedding matrixs.

Functions:

1. random_uniform -- A matrix of random uniform numbers taken from a 
   distribution between `low` and `high`
'''

import numpy as np

def random_uniform(num_words: int, embedding_dim: int, 
                   low: float = -0.05, high: float = 0.05) -> np.ndarray:
    '''
    :return: A matrix of size = [num_words, embedding_dim] where the matrix 
             is full of uniform random numbers selected from a distribution 
             between `low` to `high`.
    '''
    return np.random.uniform(low=low, high=high, 
                             size=(num_words, embedding_dim))