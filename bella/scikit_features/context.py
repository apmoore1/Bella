'''
Module contains a Class that is a scikit learn Transformer.
'''
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from bella import contexts


class Context(BaseEstimator, TransformerMixin):

    def __init__(self, context='left', inc_target=False):
        self.context = context
        self.inc_target = inc_target

    def fit(self, target_dicts, y=None):
        '''Kept for consistnecy with the TransformerMixin'''

        return self

    def fit_transform(self, target_dicts, y=None):
        '''see self.transform'''

        return self.transform(target_dicts)


    def transform(self, target_dicts):
        '''
        Given a list of target dictionaries containing the spans of the targets
        and the texts that are about the targets it returns the relevant left,
        right and target contexts with respect to the target word(s). Returns a
        list of contexts.

        :param target_dicts: list of dictionaries containing at least `spans` and \
        `text` keys.
        :type target_dicts: list
        :returns: a list of left, right and target contexts with respect to the \
        target word and the values in the self.context if self.context = 'lt' will \
        only return the left and target contexts and not right.
        :rtype: list
        '''

        all_context_data = []
        for target_dict in target_dicts:
            context_data = []
            context_data.extend(contexts.context(target_dict, self.context,
                                                 self.inc_target))
            all_context_data.append(context_data)
        return all_context_data
