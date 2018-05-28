'''
Module contains a Class that is a scikit learn Transformer.

Classes:

1. ContextTokeniser - Converts a list of String lists into token lists. See the
transformer method of the class for more details.
'''
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from bella import tokenisers


class ContextTokeniser(BaseEstimator, TransformerMixin):
    '''
    Scikit learn transformer class. Converts list of String lists into tokens.

    Attributes:

    1. self.tokeniser - tokeniser function. Given a String returns a list of Strings.
    Default whitespace tokeniser.
    2. self.lower - whether to lower case the tokens. Default False.

    See :py:func:`bella.tokenisers` for more tokeniser functions that can be
    used here or create your own function.
    '''

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
        Given a list of String lists where each String represents a context per
        target span it returns those Strings as a list of Strings (tokens).

        :param target_contexts: A list of String lists e.g. \
        [['It was nice this morning', 'It was nice this morning but not \
        yesterday morning'], ['another day']] where each String is a span context \
        for a target.
        :type target_contexts: list
        :returns: A list of Strings (tokens) per span context. e.g.
        [[['It', 'was', 'nice', 'this', 'morning'], ['It', 'was', 'nice', 'this',\
        'morning', 'but', 'not', 'yesterday', 'morning']], [['another', 'day']]]
        :rtype: list
        '''

        token_contexts = []
        for target_span_contexts in target_contexts:
            token_span_contexts = []
            for span_context in target_span_contexts:
                tokens = self.tokeniser(span_context)
                if self.lower:
                    tokens = [token.lower() for token in tokens]
                token_span_contexts.append(tokens)
            token_contexts.append(token_span_contexts)
        return token_contexts
