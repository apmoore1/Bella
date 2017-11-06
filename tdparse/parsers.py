'''
Functions that parse the annotated data that is being used in this project. The
annotated dataset are the following:

1. `Li Dong <http://goo.gl/5Enpu7>`_ which links to :py:func:`tdparse.parsers.dong`
'''
import os
import re

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
