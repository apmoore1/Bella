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
import os
import re

from tdparse.helper import read_config

def hu_liu():
    '''
    Reads the path of the folder containing the Positive and Negative words from
    the config file under `lexicons.hu_liu`.

    :returns: Returns the lexicon as a list of tuples (word, value). Where the \
    value is either `positive` or `negative`.
    :rtype: list
    '''

    sentiment_folder = os.path.abspath(read_config('lexicons')['hu_liu'])
    values = ['positive', 'negative']
    word_value = []
    for value in values:
        file_path = os.path.join(sentiment_folder, '{}-words.txt'.format(value))
        with open(file_path, 'r', encoding='cp1252') as senti_file:
            for line in senti_file:
                if re.search('^;', line) or re.search(r'^\W+', line):
                    continue
                word_value.append((line.strip(), value))
    return word_value

def nrc_emotion():
    '''
    Reads the path of the file containing the emotion words from
    the config file under `lexicons.nrc_emotion`.

    Emotion categories: 1. anger, 2. fear, 3. anticipation, 4. trust, 5. surprise,
    6. sadness, 7. joy, 8. disgust, 9. positive, and 10. negative.

    :returns: Returns the lexicon as a list of tuples (word, value). Where the \
    value is either one of the emotion categories.
    :rtype: list
    '''

    emotion_file_path = os.path.abspath(read_config('lexicons')['nrc_emotion'])
    word_value = []

    with open(emotion_file_path, 'r', newline='') as emotion_file:
        tsv_reader = csv.reader(emotion_file, delimiter='\t')
        for row in tsv_reader:
            if len(row):
                word = row[0]
                category = row[1]
                association = int(row[2])
                if association:
                    word_value.append((word, category))
    return word_value

def mpqa():
    '''
    Reads the path of the file containing the polarity of words from
    the config file under `lexicons.mpqa`.

    polarity labels = 1. positive, 2. negative, 3. both, 4. neutral

    :returns: Returns the lexicon as a list of tuples (word, value). Where the \
    value is either one of the polarity labels.
    :rtype: list
    '''

    mpqa_file_path = read_config('lexicons')['mpqa']
    word_value = []
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
                if value == 'weakneg':
                    value = key_values['polarity']
                word_value.append((word, value))
    return word_value
