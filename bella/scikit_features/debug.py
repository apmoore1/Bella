'''
SKLearn Transformer

Classes:

1. Debug -- Allows you to debug previous transformers
'''

import pdb

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

class Debug(BaseEstimator, TransformerMixin):
    '''
    SKLearn transformer that is to be used a debugger between different
    SKLearn transformers.

    Methods:

    1. fit - Does nothing as nothing is done at fit time.
    2. fit_transform - Performs the transform method.
    3. transform - Creates a REPL terminal to inspect the input transformed data
    '''
    def __init__(self):
        pass

    def fit(self, data, y=None):
        '''Kept for consistnecy with the TransformerMixin'''

        return self

    def fit_transform(self, data, y=None):
        '''see self.transform'''

        return self.transform(data)

    def transform(self, data):
        '''
        :param data: Data from the previous transformer in the pipeline
        :type data: array
        :returns: Nothing. Creates a Python interactive shell to inspect the \
        data from the previous transformation.
        :rtype: None
        '''

        pdb.set_trace()
