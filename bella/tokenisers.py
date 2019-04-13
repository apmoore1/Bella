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
5. :py:func:`bella.tokenisers.spacy_tokeniser` -- 
   `SpaCy tokeniser <https://spacy.io/>`_
'''
from typing import List, Dict

import twokenize
import spacy
from spacy.cli.download import download as spacy_download
from spacy.cli import link
from spacy.util import get_package_path
from spacy.language import Language as SpacyModelType

from bella import stanford_tools
from bella.moses_tools import MosesTokenizer


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


def moses(text: str, aggressive_dash_splits: bool = False, 
          escape: bool = True) -> List[str]:
    '''
    Tokeniser used in the `moses toolkit <https://github.com/moses-smt>`_

    :param text: A string to be tokenised.
    :param aggressive_dash_splits: Option to trigger dash split rules
    :param escape: Whether to escape characters e.g. "'s" escaped equals 
                   "&apos;s"
    :returns: A list of tokens where each token is a String.
    '''

    if isinstance(text, str):
        moses = MosesTokenizer()
        return moses.tokenize(text, 
                              aggressive_dash_splits=aggressive_dash_splits,
                              escape=escape)
    raise ValueError(f'The paramter must be of type str not {type(text)}')

LOADED_SPACY_MODELS: Dict[str, SpacyModelType] = {}

def _get_spacy_model(language: str) -> SpacyModelType:
    """
    To avoid laoding lots of spacy models the model specific to a language 
    is loaded and saved within a Global dictionary.
    This has been mainly taken from the `AllenNLP package <https://github.
    com/allenai/allennlp/blob/master/allennlp/common/util.py>`_
    :param language: Language of the SpaCy model to load.
    :returns: The relevant SpaCy model.
    """
    if language not in LOADED_SPACY_MODELS:
        disable = ['vectors', 'textcat', 'tagger', 'parser', 'ner']
        try:
            spacy_model = spacy.load(language, disable=disable)
        except:
            print(f"Spacy models '{language}' not found.  Downloading and installing.")
            spacy_download(language)
            package_path = get_package_path(language)
            spacy_model = spacy.load(language, disable=disable)
            link(language, language, model_path=package_path)
        LOADED_SPACY_MODELS[language] = spacy_model
    return LOADED_SPACY_MODELS[language]

def spacy_tokeniser(text: str) -> List[str]:
    '''
    `SpaCy tokeniser <https://spacy.io/>`_

    Assumes the language to be English.

    :param text: A string to be tokenised.
    :returns: A list of tokens where each token is a String.
    '''
    if not isinstance(text, str):
        raise ValueError('The parameter passed has to be of type' 
                         f'String not {type(text)}')
    spacy_model = _get_spacy_model('en')

    spacy_document = spacy_model(text)
    tokens = []
    for token in spacy_document:
        if not token.is_space:
            tokens.append(token.text)
    return tokens
