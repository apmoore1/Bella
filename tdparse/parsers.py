'''
Functions that parse the annotated data that is being used in this project. The
annotated dataset are the following:
1. `Li Dong <http://goo.gl/5Enpu7>`_ which links to :py:func:`tdparse.parsers.dong`
'''
import os

def dong(file_path):
    '''
    :param file_path: File Path to the annotated data
    :type file_path: String
    :returns: A list of dictionaries
    :rtype: list
    '''

    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError('This file does not exist {}'.format(file_path))

    sentiment_data = []
    with open(file_path, 'r') as dong_file:
        for index, line in enumerate(dong_file):
            line = line.strip()
    return sentiment_data
