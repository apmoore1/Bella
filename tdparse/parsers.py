'''
Functions that parse the annotated data that is being used in this project. The
annotated dataset are the following:

1. `Li Dong <http://goo.gl/5Enpu7>`_ which links to :py:func:`tdparse.parsers.dong`
2. Semeval parser
'''
import os
import re
import xml.etree.ElementTree as ET

def dong(file_path):
    '''
    Given file path to the
    `Li Dong <https://github.com/bluemonk482/tdparse/tree/master/data/lidong>`_
    sentiment data it will parse the data and return it as a list of dictionaries.

    :param file_path: File Path to the annotated data
    :type file_path: String
    :returns: A list of dictionaries
    :rtype: list
    '''

    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError('This file does not exist {}'.format(file_path))

    sentiment_range = [-1, 0, 1]

    sentiment_data = []
    with open(file_path, 'r') as dong_file:
        sent_dict = {}
        for index, line in enumerate(dong_file):
            divisible = index + 1
            line = line.strip()
            if divisible % 3 == 1:
                sent_dict['text'] = line
            elif divisible % 3 == 2:
                sent_dict['target'] = line
            elif divisible % 3 == 0:
                sentiment = int(line)
                if sentiment not in sentiment_range:
                    raise ValueError('The sentiment has to be one of the following '\
                                     'values {} not {}'.format(sentiment_range, sentiment))
                sent_dict['sentiment'] = int(line)
                text = sent_dict['text'].lower()
                target = sent_dict['target'].lower()
                offsets = [list(match.span()) for match in re.finditer(target, text)]
                if len(target.split()) > 1:
                    joined_target = ''.join(target.split())
                    offsets.extend([list(match.span())
                                    for match in re.finditer(joined_target, text)])
                sent_dict['spans'] = offsets
                sent_dict['id'] = len(sentiment_data)
                sentiment_data.append(sent_dict)
                sent_dict = {}
            else:
                raise Exception('Problem')
    return sentiment_data

def semeval(file_path):
    '''
    :param file_path: File path to the semeval data
    :type file_path: String
    :returns: A list of dictionaries containing target, sentiment , span, \
    and text for training an aspect target sentiment classifier.
    :rtype: list
    '''

    # Converts the sentiment tags from Strings to ints
    sentiment_mapper = {'conflict' : -2, 'negative' : -1,
                        'neutral' : 0, 'positive' : 1}
    def extract_aspect_terms(aspect_terms, sentence_id):
        '''
        :param aspect_terms: An aspectTerms element within the xml tree
        :param sentence_id: Id of the sentence that the aspects came from.
        :type aspect_terms: xml.etree.ElementTree.Element
        :type sentence_id: String
        :returns: A list of dictioanries containg id, span, sentiment and \
        target
        :rtype: list
        '''

        aspect_terms_data = []
        for index, aspect_term in enumerate(aspect_terms):
            aspect_term = aspect_term.attrib
            aspect_term_data = {}
            sentiment = sentiment_mapper[aspect_term['polarity']]
            aspect_id = '{}{}'.format(sentence_id, index)
            aspect_term_data['id'] = aspect_id
            aspect_term_data['target'] = aspect_term['term']
            aspect_term_data['sentiment'] = sentiment
            aspect_term_data['spans'] = [[aspect_term['from'],
                                        aspect_term['to']]]
            aspect_terms_data.append(aspect_term_data)
        return aspect_terms_data
    def add_text(aspect_data, text):
        '''
        :param aspect_data: A list of dicts containing `span`, `target` and \
        `sentiment` keys.
        :param text: The text of the sentence that is associated to all of the \
        aspects in the aspect_data list
        :type aspect_data: list
        :type text: String
        :returns: The list of dicts in the aspect_data parameter but with a \
        `text` key with the value that the text parameter contains
        :rtype: list
        '''

        for data in aspect_data:
            data['text'] = text
        return aspect_data

    tree = ET.parse(file_path)
    sentences = tree.getroot()
    all_aspect_term_data = []
    if sentences.tag != 'sentences':
        raise ValueError('The root of all semeval xml files should '\
                         'be sentences and not {}'\
                         .format(sentences.tag))
    for sentence in sentences:
        aspect_term_data = None
        text_index = None
        sentence_id = sentence.attrib['id']
        for index, data in enumerate(sentence):
            if data.tag == 'text':
                text_index = index
            elif data.tag == 'aspectTerms':
                aspect_term_data = extract_aspect_terms(data, sentence_id)
        if aspect_term_data is None:
            continue
        if text_index is None:
            raise ValueError('A semeval sentence should always have text '\
                             'semeval file {} sentence id {}'\
                             .format(file_path, sentence.attrib['id']))
        sentence_text = sentence[text_index].text
        aspect_term_data = add_text(aspect_term_data, sentence_text)
        all_aspect_term_data.extend(aspect_term_data)
    return all_aspect_term_data
