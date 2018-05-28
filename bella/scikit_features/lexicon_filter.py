from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from bella import lexicons

class LexiconFilter(BaseEstimator, TransformerMixin):

    def __init__(self, lexicon=None, zero_token='$$$ZERO_TOKEN$$$'):
        self.lexicon = lexicon
        self.zero_token = zero_token

    def fit(self, context_tokens, y=None):
        '''Kept for consistnecy with the TransformerMixin'''

        return self

    def fit_transform(self, context_tokens, y=None):
        '''see self.transform'''

        return self.transform(context_tokens)

    def transform(self, contexts_tokens):
        lexicon_words = self.lexicon.words
        context_tokens_filtered = []
        for context_tokens in contexts_tokens:
            all_tokens_filtered = []
            for context in context_tokens:
                context_tokens = []
                for token in context:
                    if token not in lexicon_words:
                        context_tokens.append(self.zero_token)
                    else:
                        context_tokens.append(token)
                all_tokens_filtered.append(context_tokens)
            context_tokens_filtered.append(all_tokens_filtered)
        return context_tokens_filtered
