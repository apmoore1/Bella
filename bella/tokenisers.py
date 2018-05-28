'''
Functions that tokenise text and Returns the list of tokens all of which are
Strings.

1. Whitespace - :py:func:`bella.tokenisers.whitespace`
2. Twitter tokeniser - :py:func:`bella.tokenisers.ark_twokenize`
'''
import time

import twokenize

from bella import stanford_tools

def whitespace(text):
    '''
    Splits text based on Whitespace Returns the list of tokens.

    :param text: A string to be tokenised.
    :type text: String
    :returns: A list of tokens where each token is a String.
    :rtype: list
    '''

    if isinstance(text, str):
        return text.split()
    raise ValueError('The paramter must be of type str not {}'.format(type(text)))

def ark_twokenize(text):
    '''
    A Twitter tokeniser from `CMU Ark <https://github.com/brendano/ark-tweet-nlp>`_
    returns a list of tokens.

    This is just a wrapper of `this <https://github.com/Sentimentron/ark-twokenize-py>`_

    :param text: A string to be tokenised.
    :type text: String
    :returns: A list of tokens where each token is a String.
    :rtype: list
    '''

    if isinstance(text, str):
        return twokenize.tokenizeRawTweetText(text)
    raise ValueError('The paramter must be of type str not {}'.format(type(text)))

def stanford(text):
    return stanford_tools.tokenise(text)
