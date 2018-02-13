'''
Functions that tokenise text and Returns the list of tokens all of which are
Strings.

1. Whitespace - :py:func:`tdparse.tokenisers.whitespace`
2. Twitter tokeniser - :py:func:`tdparse.tokenisers.ark_twokenize`
'''
import time
import random

import twokenize
from pycorenlp import StanfordCoreNLP

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
    nlp = StanfordCoreNLP('http://localhost:9000')

    tokens = []
    annotations = None
    while annotations is None:
        annotations = nlp.annotate(text, properties={'annotators' : 'tokenize,ssplit',
                                                     'tokenize.language' : 'en',
                                                     'timeout' : '50000',
                                                     'outputFormat' : 'json'})
        if annotations is None:
            time.sleep(1)

    for sentence in annotations['sentences']:
        tokens.extend([token['word'] for token in sentence['tokens']])

    return tokens
