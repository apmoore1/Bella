'''
This module contains the following sentiment lexicons:

1. `Hu and Liu <https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf>`_ \
:py:func:`tdparse.lexicons.hu_liu`.
2. `Mohammad and Turney \
<http://saifmohammad.com/WebDocs/Mohammad-Turney-NAACL10-EmotionWorkshop.pdf>`_\
:py:func:`tdparse.lexicons.nrc_emotion`
3. `Wilson, Wiebe and Hoffman \
<https://aclanthology.coli.uni-saarland.de/papers/H05-1044/h05-1044>`_\
:py:func:`tdparse.lexicons.mpqa`

All which are avaliable through there associated functions. The return type of
all of the functions are a list of tuples. All of the tuples are a length of
two containg:

1. String: word
2. String: Value associated to that word e.g. 'positive'
'''
import csv
from functools import wraps
import os
import re

from tdparse.helper import read_config

def combine_lexicons(lexicon1, lexicon2):
    '''
    Combines two lexicons and removes words that have opposite sentiment. e.g.
    one lexicon a word is positive and in the other negative. Returns a list
    of tuples (word, value) e.g. ('great', 'positive')

    NOTE: Requires that both lexicons have the same values for positive and
    negative values.

    :param lexicon1: list of tuples containing (word, value)
    :param lexicon2: list of tuples containing (word, value)
    :type lexicon1: list
    :type lexicon2: list
    :returns: A list of tuples containing (word, value)
    :rtype: list
    '''
    def compare_lexicons(lex1, lex2):
        '''
        Given the two lexicons as dictionaries will add (word, value) tuples to
        a combined lexicon list and return that list.

        :param lex1: dictionary of word : value
        :param lex2: dictionary of word : value
        :type lex1: dict
        :type lex2: dict
        :returns: A list of tuples (word, value) where the values are not \
        contradictory between the two lexicons.
        :rtype: list
        '''

        combined_words = set(list(lex1.keys()) + list(lex2.keys()))
        combined_lexicons = []
        for word in combined_words:

            if word in lex1 and word in lex2:
                if lex1[word] != lex2[word]:
                    continue
                combined_lexicons.append((word, lex1[word]))
            elif word in lex1:
                combined_lexicons.append((word, lex1[word]))
            elif word in lex2:
                combined_lexicons.append((word, lex2[word]))
            else:
                raise KeyError('The word {} has to be in one of the lexicons'\
                               .format(word))
        return combined_lexicons


    if not isinstance(lexicon1, list) or not isinstance(lexicon2, list):
        raise TypeError('Both parameters require to be lists not {}, {}'\
                        .format(type(lexicon1), type(lexicon2)))
    word_value1 = {word : value for word, value in lexicon1}
    word_value2 = {word : value for word, value in lexicon2}

    values1 = set(word_value1.values())
    values2 = set(word_value2.values())
    if values1 != values2:
        raise ValueError('These two lexicons cannot be combined as they have '\
                         'different and non-comparable values: values1: {} '\
                         'values2: {}'.format(values1, values2))
    return compare_lexicons(word_value1, word_value2)




def parameter_check(lexicon_function):
    '''
    Decorator that checks the type values of parameters within the module.

    :param lexicon_function: One of the functions in this module.
    :type lexicon_function: function
    :returns: The lexicon function wrapped around a type check function
    :rtype: function
    '''
    @wraps(lexicon_function)
    def check(subset_values=None, lower=False):
        '''
        Function that checks that the parameters are of the correct type and then
        returns the wrapped function.

        :param subset_values: categories of words that you want to return e.g. \
        `positive`
        :param lower: wether to lower case the sentiment lexicon
        :type subset_values: set Default is None
        :type lower: bool Default False
        :returns: The wrapped function
        :rtype: function
        '''
        if subset_values is not None:
            if not isinstance(subset_values, set):
                raise TypeError('subset_values parameter has to be of type set '\
                                'and not {}'.format(type(subset_values)))
        if lower != False:
            if not isinstance(lower, bool):
                raise TypeError('lower parameter has to be of type bool not {}'\
                                .format(type(lower)))
        return lexicon_function(subset_values=subset_values, lower=lower)
    return check

@parameter_check
def hu_liu(subset_values=None, lower=False):
    '''
    Reads the path of the folder containing the Positive and Negative words from
    the config file under `lexicons.hu_liu`.

    :param subset_values: Categories of words that you want to return e.g. \
    `positive` for `positive` words only. If None then no words are subsetted.
    :type subset_values: set Default None
    :returns: Returns the lexicon as a list of tuples (word, value). Where the \
    value is either `positive` or `negative`.
    :rtype: list
    '''

    sentiment_folder = os.path.abspath(read_config('lexicons')['hu_liu'])
    values = ['positive', 'negative']
    if subset_values is not None:
        values = subset_values
    word_value = set()
    for value in values:
        file_path = os.path.join(sentiment_folder, '{}-words.txt'.format(value))
        with open(file_path, 'r', encoding='cp1252') as senti_file:
            for line in senti_file:
                if re.search('^;', line) or re.search(r'^\W+', line):
                    continue
                line = line.strip()
                if lower:
                    line = line.lower()
                word_value.add((line.strip(), value))
    return list(word_value)

@parameter_check
def nrc_emotion(subset_values=None, lower=False):
    '''
    Reads the path of the file containing the emotion words from
    the config file under `lexicons.nrc_emotion`.

    Emotion categories: 1. anger, 2. fear, 3. anticipation, 4. trust, 5. surprise,
    6. sadness, 7. joy, 8. disgust, 9. positive, and 10. negative.

    :param subset_values: Categories of words that you want to return e.g. \
    `positive` for `positive` words only. If None then no words are subsetted.
    :type subset_values: set Default None
    :returns: Returns the lexicon as a list of tuples (word, value). Where the \
    value is either one of the emotion categories.
    :rtype: list
    '''

    emotion_file_path = os.path.abspath(read_config('lexicons')['nrc_emotion'])
    word_value = set()

    with open(emotion_file_path, 'r', newline='') as emotion_file:
        tsv_reader = csv.reader(emotion_file, delimiter='\t')
        for row in tsv_reader:
            if len(row):
                word = row[0]
                category = row[1]
                association = int(row[2])
                if lower:
                    word = word.lower()
                if association:
                    if subset_values is None:
                        word_value.add((word, category))
                    elif category in subset_values:
                        word_value.add((word, category))
    return list(word_value)

@parameter_check
def mpqa(subset_values=None, lower=False):
    '''
    Reads the path of the file containing the polarity of words from
    the config file under `lexicons.mpqa`.

    polarity labels = 1. positive, 2. negative, 3. both, 4. neutral

    :param subset_values: Categories of words that you want to return e.g. \
    `positive` for `positive` words only. If None then no words are subsetted.
    :type subset_values: set Default None
    :returns: Returns the lexicon as a list of tuples (word, value). Where the \
    value is either one of the polarity labels.
    :rtype: list
    '''

    mpqa_file_path = read_config('lexicons')['mpqa']
    word_value = set()
    with open(mpqa_file_path, 'r') as mpqa_file:
        for line in mpqa_file:
            line = line.strip()
            if line:
                key_values = {}
                for data in line.split():
                    if '=' in data:
                        key, value = data.split('=')
                        key_values[key] = value
                word = key_values['word1']
                value = key_values['priorpolarity']
                if lower:
                    word = word.lower()
                if value == 'weakneg':
                    value = key_values['polarity']
                if subset_values is None:
                    word_value.add((word, value))
                elif value in subset_values:
                    word_value.add((word, value))

    return list(word_value)
