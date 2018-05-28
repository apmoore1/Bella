'''
Module contains a Class that is a scikit learn Transformer.

Classes:

1. SyntacticContext - Converts a list of dictionaries containg text, targets, \
and target spans into text contexts defined the targets dependency tree. \
Returns a list of a list of dictionaries containing text and span.
2. DependencyChildContext - Simialr to SyntacticContext but returns a list of \
a list of Strings instead of dicts. Each String represents a targets children \
dependency relations.
3. Context - Given the output from SyntacticContext returns the left, right,
target or full text contexts. Where left and right is with respect to the
target word the text is about.
'''

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from bella import syntactic_contexts
from bella.dependency_parsers import tweebo


class SyntacticContext(BaseEstimator, TransformerMixin):
    '''
    Converts a list of dictionaries containg text, targets, and target spans
    into text contexts defined the targets dependency tree. Returns a list of a
    list of dictionaries containing text and span.

    Attributes:

    1. parser - dependency parser to use.
    2. lower - whether or not the parser should process the text in lower or \
    case or not

    Methods:

    1. fit - Does nothing as nothing is done at fit time.
    2. fit_transform - Performs the transform method.
    3. transform - Converts the list of dicts into a list of a list of \
    dicts where each each dict contains the target text and span where the text \
    is the targets full dependency tree word context.
    '''

    def __init__(self, parser=tweebo, lower=False):
        '''
        For more information on what the function does see the following
        functions documentation: dependency_context

        :param parser: Dependency parser to use.
        :param lower: Whether to lower case the words before going through \
        the dependency parser.
        :type parser: function. Default tweebo
        :type lower: bool. Default False
        '''

        self.parser = parser
        self.lower = lower

    def fit(self, target_dicts, y=None):
        '''Kept for consistnecy with the TransformerMixin'''

        return self

    def fit_transform(self, target_dicts, y=None):
        '''see self.transform'''

        return self.transform(target_dicts)


    def transform(self, target_dicts):
        '''
        Given a list of target dictionaries it returns the syntactic context of
        each target therefore returns a list of a list of dicts where each
        dict represents a targets syntactic context within the associated text.
        The syntactic context depends on the self.parser function.

        :param target_dicts: list of dictionaries
        :type target_dicts: list
        :returns: A list of a list of dicts
        :rtype: list
        '''

        return syntactic_contexts.dependency_context(target_dicts,
                                                     self.parser, self.lower)

class DependencyChildContext(BaseEstimator, TransformerMixin):
    '''
    Simialr to SyntacticContext but returns a list of a list of Strings
    instead of dicts. Each String represents a targets children dependency
    relations.

    Attributes:

    1. parser - dependency parser to use.
    2. lower - whether or not the parser should process the text in lower or \
    case or not
    3. rel_depth - The depth of the child relations (1, 1) = 1st relations

    Methods:

    1. fit - Does nothing as nothing is done at fit time.
    2. fit_transform - Performs the transform method.
    3. transform - Converts the list of dicts into a list of a list of \
    Strings which are the target words child relations within the targets \
    dependency tree.
    '''

    def __init__(self, parser=tweebo, lower=False, rel_depth=(1, 1)):
        '''
        For more information on what the function does see the following
        functions documentation: dependency_relation_context

        :param parser: Dependency parser to use.
        :param lower: Whether to lower case the words before going through \
        the dependency parser.
        :param rel_depth: Depth of the dependency relations to use as context \
        words
        :type parser: function. Default tweebo
        :type lower: bool. Default False
        :type rel_depth: tuple. Default (1, 1)
        '''
        self.parser = parser
        self.lower = lower
        self.rel_depth = rel_depth

    def fit(self, target_dicts, y=None):
        '''Kept for consistnecy with the TransformerMixin'''

        return self

    def fit_transform(self, target_dicts, y=None):
        '''see self.transform'''

        return self.transform(target_dicts)


    def transform(self, target_dicts):
        '''
        Given a list of target dictionaries it returns a list of a list of Strings
        where each String is the concatenation of the targets rel_depth child
        related words where rel_depth determines the number of child dependency
        related words to include.

        :param target_dicts: list of dictionaries
        :type target_dicts: list
        :returns: A list of a list of Strings
        :rtype: list
        '''

        return syntactic_contexts.dependency_relation_context(target_dicts,
                                                              self.parser,
                                                              self.lower,
                                                              self.rel_depth)

class Context(BaseEstimator, TransformerMixin):
    '''
    Given the output from SyntacticContext returns the left, right, target or
    full text contexts. Where left and right is with respect to the
    target word the text is about.

    Attributes:

    1. context - Defines the text context that is returned
    2. inc_target - Whether to include the target word in the context

    Methods:

    1. fit - Does nothing as nothing is done at fit time.
    2. fit_transform - Performs the transform method.
    3. transform - Converts a list of a list of dictionaries into a list of a \
    list of Strings where each String represents a the targets context.
    '''
    def __init__(self, context='left', inc_target=False):
        '''
        :param context: left, right, target or full context will be returned \
        with respect to the target word in the target sentence.
        :param inc_target: Whether to include the target word in the text \
        context.
        :type context: String. Can only be `left`, `right`, `target`, or `full`\
        . Default `left`
        :type inc_target: bool. Default False
        '''
        self.context = context
        self.inc_target = inc_target

    def fit(self, target_contexts, y=None):
        '''Kept for consistnecy with the TransformerMixin'''

        return self

    def fit_transform(self, target_contexts, y=None):
        '''see self.transform'''

        return self.transform(target_contexts)


    def transform(self, target_contexts):
        '''
        Given a list of of a list of dicts where each dict contains

        :param target_dicts: list of a list of dictionaries
        :type target_dicts: list
        :returns: A list of a list of Strings where each String represents a \
        different context as each target can have many targets within a text \
        thus multiple contexts. The context it returns depends on self.context
        :rtype: list
        '''

        return syntactic_contexts.context(target_contexts, self.context,
                                          self.inc_target)
