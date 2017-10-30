'''
Functions that tokenise text and Returns the list of tokens all of which are
Strings.

1. Whitespace - :py:func:`tdparse.tokenisers.whitespace`
'''

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
