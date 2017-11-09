from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from tdparse import tokenisers


class ContextTokeniser(BaseEstimator, TransformerMixin):

    def __init__(self, tokeniser=tokenisers.whitespace, lower=False):
        self.tokeniser = tokeniser
        self.lower = lower

    def fit(self, target_contexts, y=None):
        '''Kept for consistnecy with the TransformerMixin'''

        return self

    def fit_transform(self, target_contexts, y=None):
        '''see self.transform'''

        return self.transform(target_contexts)

    def transform(self, target_contexts):
        '''
        '''

        token_contexts = []
        for target_context in target_contexts:
            token_context = []
            for context in target_context:
                token_context.append(self.tokeniser(context))
            token_contexts.append(token_context)
        return token_contexts
