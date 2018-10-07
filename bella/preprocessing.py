'''
This module contains pre-processing functions that are useful for the 
Neural Network methods.

Functions:

1. tokeniser -- Returns a Keras `keras.preprocessing.text.Tokenizer` object 
   that has been fitted on the text files given.
'''
import json
from typing import Callable, List

from keras.preprocessing.text import Tokenizer


def tokeniser(*text_fps, 
              tokeniser_function: Callable[[str], List[str]] = str.split,
              text_field: str = 'text', 
              **tokenizer_kwargs) -> Tokenizer:
    '''
    Returns a Keras `keras.preprocessing.text.Tokenizer` object that has been 
    fitted on the text files given.

    The text files have to contain json encoded data on each new line, where 
    the json data has to contain a `text` field of which this field can be 
    changed within the arguments given to this function.

    :param tokeniser_function: Tokenisation method to use default is white 
                               space splitting
    :param text_field: name of the field within the json data that contains 
                       the text. Default `text`
    :param *text_fps: File paths to the text files that have json encoded data
    :param **tokenizer_kwargs: Key Words arguments to give to the 
                               :py:class:`keras.preprocessing.text.Tokenizer`
                               object e.g. lower, filters
    '''
    def load_text_data():
        for text_fp in text_fps:
            with text_fp.open('r') as text_lines:
                for line in text_lines:
                    data = json.loads(line)
                    text = data[text_field]
                    tokens = tokeniser_function(text)
                    yield tokens
    
    tok = Tokenizer(**tokenizer_kwargs)
    tok.fit_on_texts(load_text_data())
    return tok