'''
A set of functions which either produce contexts and related targets based on
syntactic parsing. Or functions that create
'''
import re

from tdparse.data_types import Target

def normalise_target(target):
    target = '_'.join([sub_target.capitalize() for sub_target in target.split()])
    # Found that if there is more than one `_` then the Dependency tagger 
    # can think it is more than one word sometimes
    target_split = target.split('_', 1)
    if len(target_split) == 2:
        first_target, second_target = target_split
        second_target = second_target.replace('_', '')
        target = '{}_{}'.format(first_target, second_target)
    return '${}$'.format(target)
def target_normalisation(target_dict):
    '''
    Given a target instance it normalises the target by removing whitespaces
    between target words and inserting `_`. Then inserting the normalised word
    into where the target spans appear and adding whitespace around the target
    word incase other words are joined on. Returns the text with the inserted
    normalised words and the normalised target word.

    :param target_dict: target instance.
    :type target_dict: Target
    :returns: Tuple of two Strings containing. Text with normalised targets and \
    the normalised target
    :rtype: tuple
    '''

    if not isinstance(target_dict, Target):
        raise TypeError('target_dict parameter has to be of type Target and '\
                        'not {}'.format(type(target_dict)))

    sorted_spans = sorted(target_dict['spans'], key=lambda span: span[0],
                          reverse=True)
    org_text = target_dict['text']
    target = target_dict['target']

    # captalize each word in the target and remove whitespace
    target = normalise_target(target)
    for start_index, end_index in sorted_spans:
        start_text = org_text[: start_index]
        end_text = org_text[end_index :]
        start_text += ' {} '.format(target)
        org_text = start_text + end_text
    org_text = ' '.join(org_text.split())

    return org_text

def normalise_context(target_dicts, lower):
    '''
    Given a list of target dicts and if the text should be lower cased returns
    all of the text and targets within those target dicts as lists where the
    text and targets have been normalised to ensure the targets within the
    text can be identified.

    :param target_dicts: list of dicts
    :param lower: state if the text within the dicts should be lower cased
    :type target_dicts: list
    :type lower: bool
    :returns: A tuple of length two which contain a list of normalised texts and \
    targets.
    :rtype: list
    '''

    # Normalise the target and text
    all_text = []
    all_norm_targets = []
    for target_dict in target_dicts:
        norm_target = target_dict['target']
        norm_text = target_normalisation(target_dict)
        if lower:
            norm_text = norm_text.lower()
            norm_target = norm_target.lower()
        all_text.append(norm_text)
        all_norm_targets.append(norm_target)
    return all_text, all_norm_targets

def dependency_relation_context(target_dicts, parser, lower=False,
                                n_relations=(1, 1)):
    '''
    Given a list of target dicts where each target dict has a sentence that
    contains one or more of the same target. Returns a list of a list of Strings
    where each String is associated to a target within the target sentence.
    The String is a concatenation of n_relations depth of dependency relations
    where each relation is a child of the target. e.g. n_relations = (1, 1)
    will return a String of the concatenation of the children of the target
    within the dependency tree. n_relations = (1, 2) will return the children
    of the target and the children of the children.

    :param target_dicts: list of dictionaries where each dictionary is associated \
    to a target sentence.
    :param parser: function that performs dependency parsing
    :param lower: Whether to lower case the text
    :param n_relations: The depth of the dependency relation text from the target \
    to return. Represented as a tuple of two ints the first defining the \
    starting depth the second end depth e.g. (1, 2) will return depths one and \
    two of dependency tree.
    :type target_dicts: list
    :type parser: String
    :type lower: bool. Default False.
    :type n_relations: tuple. Default (1, 1).
    :returns: A list of a list of Strings where each String represents a specific \
    target word within a target sentence dependency related text at n_relations \
    depth.
    :rtype: list
    '''

    # Normalise the target and text
    norm_texts, norm_targets = normalise_context(target_dicts, lower)
    # Get contexts
    all_dependency_tokens = parser(norm_texts)
    all_contexts = []

    for index, dependency_tokens in enumerate(all_dependency_tokens):
        contexts = []
        norm_target = norm_targets[index]


        for dependency_token in dependency_tokens:
            current_token = normalise_target(norm_target)
            if lower:
                current_token = current_token.lower()
            if dependency_token.token == current_token:
                all_related_words = dependency_token.get_n_relations(n_relations)
                related_text = ' '.join(all_related_words)
                related_text = related_text.replace(current_token, norm_target)
                contexts.append(related_text)

        rel_target = target_dicts[index]
        valid_num_targets = len(rel_target['spans'])
        if valid_num_targets != len(contexts):
            raise ValueError('The number of identified targets `{}` not equal '\
                             'to the number of targets in the data `{}`'\
                             .format(contexts, rel_target))
        # Ensure the returned data type is consistent
        if contexts == []:
            raise ValueError('This should not happen as each data type should '\
                             'have a target {}'.format(rel_target))
        all_contexts.append(contexts)
    return all_contexts

def text_and_span(text, target, current_token, target_span):

    # Find the index of target with regards to the special token representing
    # the target and other words in the text that are the target words as well
    target_token_match_spans = []
    escaped_targ_tok = re.escape(current_token)
    targ_tok_queries = [r'^{}\s'.format(escaped_targ_tok), 
                        r'\s{}$'.format(escaped_targ_tok), 
                        r'^{}$'.format(escaped_targ_tok),
                        r'\s{}\s'.format(escaped_targ_tok)]
    search_queries = targ_tok_queries + [r'\s{}\s'.format(re.escape(target))]
    for query_index, search_query in enumerate(search_queries):
        for match in re.finditer(search_query, text):
            span_match = match.span()
            # Need to remove the whitespace tokens within the span match
            if query_index == 0:
                span_match = (0, span_match[1] - 1)
            elif query_index == 1:
                span_match = (span_match[0] + 1, span_match[1])
            elif query_index == 2:
                span_match = (0, span_match[1])
            else:
                span_match = (span_match[0] + 1, span_match[1] - 1)
            target_token_match_spans.append(span_match)
    if target_token_match_spans == []:
        raise ValueError('There should be at least one Target Token `{}` in '\
                         'the text `{}`'.format(current_token, text))
    target_token_match_spans = sorted(target_token_match_spans, 
                                      key=lambda span: span[0])

    target_token_index = None
    for index, match_span in enumerate(target_token_match_spans):
        if match_span == target_span:
            target_token_index = index
    if target_token_index is None:
        #import code
        #code.interact(local=locals())
        raise ValueError('Cannot find the related default TOKEN `{}` which '\
                         'identifies the current target `{}` at span `{}` '\
                         'in text `{}` candidate spans {}'\
                         .format(current_token, target, target_span, text, 
                                 target_token_match_spans))
    updated_text = text
    for targ_tok_pattern in targ_tok_queries:
        updated_text = re.sub(targ_tok_pattern, ' {} '.format(target), updated_text)

    match_count = 0
    re_target_pattern = r'\s{}\s'.format(re.escape(target))
    for index, match in enumerate(re.finditer(re_target_pattern, updated_text)):
        if index == target_token_index:
            span_match = match.span()
            # Need to remove the whitespace tokens within the span match
            updated_target_span = (span_match[0] + 1, span_match[1] - 1)
        match_count += 1

    org_match_count = len(target_token_match_spans)
    if match_count != org_match_count:
        raise ValueError('The number of target words `{}` are not equal to the'\
                         'number that were converted `{}`. Text `{}` target `{}`'\
                         .format(org_match_count, match_count, text, target))
    return updated_text.strip(), updated_target_span



def dependency_context(target_dicts, parser, lower=False):
    '''
    Given a list of target dicts it will normalise the target word to ensure
    that it is seperated and if it is a multi word target join the target words
    together to ensure when it is processed by the dependency parser it is treated
    as a singular word.


    Given a list of target sentences returns a list of contexts where each contexts
    is associated to a target sentence and each contexts contains a target context
    for each target word in the target sentence. A target context is a dict which
    contains `text` and `span` keys where the values correspond to all the
    dependency related words as a String and the span are the indexs to the target
    word within the text.

    :param target_dicts: list of dictionaries where each dictionary is associated \
    to a target sentence.
    :param parser: function that performs dependency parsing
    :param lower: Whether to lower case the texts before processing them with the \
    parser.
    :type target_dicts: list
    :type parser: function
    :type lower: bool Default False
    :returns: A list of a list of dicts where each list is contains many contexts.
    :rtype: list
    '''

    # Normalise the target and text
    norm_texts, norm_targets = normalise_context(target_dicts, lower)
    # Get contexts
    all_dependency_tokens = parser(norm_texts)
    all_contexts = []

    for index, dependency_tokens in enumerate(all_dependency_tokens):
        contexts = []
        norm_target = norm_targets[index]


        for dependency_token in dependency_tokens:
            current_token = normalise_target(norm_target)
            if lower:
                current_token = current_token.lower()
            if dependency_token.token == current_token:
                connected_text, target_span = dependency_token\
                                              .connected_target_span()
                text_span = text_and_span(connected_text, norm_target,
                                          current_token, target_span)
                contexts.append({'text' : text_span[0],
                                 'span' : text_span[1]})
        rel_target = target_dicts[index]
        valid_num_targets = len(rel_target['spans'])
        if valid_num_targets != len(contexts):
            raise ValueError('The number of identified targets `{}` not equal '\
                             'to the number of targets in the data `{}`'\
                             .format(contexts, rel_target))
        # Ensure the returned data type is consistent
        if contexts == []:
            raise ValueError('This should not happen as each data type should '\
                             'have a target {}'.format(rel_target))
        all_contexts.append(contexts)
    return all_contexts

def context(all_context_dicts, specific_context, inc_target=False):
    '''
    Returns a list of a list of Strings based on the location of the target word
    in the text within the target dict (NOTE the target word can occur more than
    once hence why a list is returned as the context is returned for each
    occurence). Context can be one of the following:

    1. left - left of the target occurence.
    2. right - right of the target occurence.
    3. target - target word/words of each target occurence.
    4. full - whole text repeated for each occurence.

    If the target only occur once in the text then for that text occurence the
    length of the list will be one.

    :param target_dict: Dictionary that contains text and the spans of the \
    target word in the text.
    :param specific_context: String specifying the context e.g. left.
    :param inc_target: Whether to include the target word in the context text. \
    (Only applies for left and right context.)
    :type target_dict: dict
    :type specific_context: String
    :type inc_target: Boolean Default False
    :returns: A list of of a list of context strings
    :rtype: list
    '''

    acceptable_contexts = {'left', 'right', 'target', 'full'}
    if specific_context not in acceptable_contexts:
        raise ValueError('context parameter can only be one of the following {}'\
                         ' not {}'.format(acceptable_contexts, context))
    all_contexts = []
    for context_dicts in all_context_dicts:
        contexts = []
        for context_dict in context_dicts:
            text = context_dict['text']
            target_span = context_dict['span']
            start_char = target_span[0]
            end_char = target_span[1]
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
        all_contexts.append(contexts)
    return all_contexts
