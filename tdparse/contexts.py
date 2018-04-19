'''
Contains functions that get different contexts from a text based on where the
target of the sentiment is within the text.


Functions:

1. right_context
2. left_context
3. target_context
4. full_context

Each function above relies on the private `_context` function.
'''

def context(target_dict, specific_context, inc_target=False):
    '''
    Returns a list of Strings based on the location of the target word in the
    text within the target dict (NOTE the target word can occur more than once
    hence why a list is returned as the context is returned for each occurence).
    specific_context can be one of the following:

    1. left - left of the target occurence.
    2. right - right of the target occurence.
    3. target - target word/words of each target occurence.
    4. full - whole text repeated for each occurence.

    The list will be length 1 if the target word only occurs once in the text.

    :param target_dict: Dictionary that contains text and the spans of the \
    target word in the text.
    :param specific_context: String specifying either the context e.g. left.
    :param inc_target: Whether to include the target word in the context text. \
    (Only applies for left and right context.)
    :type target_dict: dict
    :type specific_context: String
    :type inc_target: Boolean Default False
    :returns: A list of context strings
    :rtype: list
    '''

    acceptable_contexts = {'left', 'right', 'target', 'full'}
    if specific_context not in acceptable_contexts:
        raise ValueError('context parameter can only be one of the following {}'\
                         ' not {}'.format(acceptable_contexts, specific_context))
    text = target_dict['text']
    spans = target_dict['spans']
    contexts = []
    for span in spans:
        start_char = span[0]
        end_char = span[1]
        if specific_context == 'left':
            if inc_target:
                contexts.append(text[:end_char])
            else:
                contexts.append(text[:start_char])
        elif specific_context == 'right':
            if inc_target:
                contexts.append(text[start_char:])
            else:
                contexts.append(text[end_char:])
        elif specific_context == 'target':
            contexts.append(text[start_char:end_char])
        elif specific_context == 'full':
            contexts.append(text)
        else:
            raise ValueError('context parameter should only be `right` or '\
                             '`left` not {} there must be a logic error'\
                             .format(specific_context))
    return contexts
