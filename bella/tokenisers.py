'''
Functions that tokenise text and returns the list of tokens all of which are
Strings.

Functions:

1. :py:func:`bella.tokenisers.whitespace` -- tokenises on whitespace.
2. :py:func:`bella.tokenisers.ark_twokenize` -- A Twitter tokeniser from
   `CMU Ark <https://github.com/brendano/ark-tweet-nlp>`_
3. :py:func:`bella.tokenisers.stanford` -- Stanford tokeniser from
   `CoreNLP <https://nlp.stanford.edu/software/tokenizer.html>`_
4. :py:func:`bella.tokenisers.moses` -- Tokeniser used in the
   `moses toolkit <https://github.com/moses-smt>`_
'''
from typing import List

from mosestokenizer import MosesTokenizer
import twokenize

from bella import stanford_tools


def whitespace(text: str) -> List[str]:
    '''
    Tokenises on whitespace.

    :param text: A string to be tokenised.
    :returns: A list of tokens where each token is a String.
    '''

    if isinstance(text, str):
        return text.split()
    raise ValueError(f'The paramter must be of type str not {type(text)}')


def ark_twokenize(text: str) -> List[str]:
    '''
    A Twitter tokeniser from
    `CMU Ark <https://github.com/brendano/ark-tweet-nlp>`_

    This is a wrapper of
    `this <https://github.com/Sentimentron/ark-twokenize-py>`_

    :param text: A string to be tokenised.
    :returns: A list of tokens where each token is a String.
    '''

    if isinstance(text, str):
        return twokenize.tokenizeRawTweetText(text)
    raise ValueError(f'The paramter must be of type str not {type(text)}')


def stanford(text: str) -> List[str]:
    '''
    Stanford tokeniser from
    `CoreNLP <https://nlp.stanford.edu/software/tokenizer.html>`_

    Requires CoreNLP server to be running.

    :param text: A string to be tokenised.
    :returns: A list of tokens where each token is a String.
    '''
    if isinstance(text, str):
        return stanford_tools.tokenise(text)
    raise ValueError(f'The paramter must be of type str not {type(text)}')


class Moses(object):
    '''
    Singleton Class instance
    '''

    instance = None

    def __new__(cls):
        if Moses.instance is None:
            Moses.instance = MosesTokenizer('en')
        return Moses.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name, value):
        return setattr(self.instance, name, value)


def moses(text: str) -> List[str]:
    '''
    Tokeniser used in the `moses toolkit <https://github.com/moses-smt>`_

    This is a wrapper of `this <https://pypi.org/project/mosestokenizer/>`_

    It expects the text not to contain any new lines therefore we split the
    text by new lines and then join the tokens in each line together.

    :param text: A string to be tokenised.
    :returns: A list of tokens where each token is a String.
    '''
    if isinstance(text, str):
        tokeniser = Moses()
        new_line_tokens = [tokeniser(new_line_text)
                           for new_line_text in text.split('\n')]
        return [tokens for new_line in new_line_tokens
                for tokens in new_line]
    raise ValueError(f'The paramter must be of type str not {type(text)}')
