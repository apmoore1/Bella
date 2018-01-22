'''
A set of functions which either produce contexts and related targets based on
syntactic parsing. Or functions that create
'''
import re

from tdparse.data_types import Target



def normalise_target(target, text, sorted_target_spans, renormalise=False):

    def add_target_to_text(target):
        target_added_text = text
        for start, end in sorted_target_spans:
            start_text = target_added_text[: start]
            end_text = target_added_text[end :]
            start_text += ' {} '.format(target)
            target_added_text = start_text + end_text
        return ' '.join(target_added_text.split())

    def replace_target_in_text(target, new_target):
        return text.replace(target, new_target)

    def check_target_unique(target, text, num_target_spans):
        escaped_target = re.escape(target)
        num_target_occurences = len(re.findall(escaped_target, text))
        if num_target_occurences != num_target_spans:
            return False
        return True

    norm_target = target.strip()
    norm_text = text
    num_target_spans = len(sorted_target_spans)

    split_target = target.split()
    num_spaces_in_target = len(split_target)
    # Converts words such LG Flat Screen into LG_FlatScreen This is done as the
    # parser won't keep a word as a word if it is LG_Flat_Screen
    if renormalise:
        norm_target = ''.join(split_target)
    elif num_spaces_in_target > 2:
        first_word = split_target[0]
        joined_rest_words = ''.join(split_target[1:])
        norm_target = '_'.join([first_word, joined_rest_words])
    # Keeps the word as a whole word.
    elif num_spaces_in_target == 2:
        temp_split_target = []
        for word_index, split_word in enumerate(split_target):
            if word_index != 0:
                split_word = split_word.replace('#', '')
            temp_split_target.append(split_word)
        norm_target = '_'.join(temp_split_target)
    # Gets rid of anything this is not a word @ or #
    norm_target = re.sub(r'[^\w@#]', '', norm_target)

    # Put the normalised targets into the text
    norm_text = add_target_to_text(norm_target)


    # Checks a word that is the normalised target word exists in the text and
    # if so changes it to a word that is not by putting dollar signs around it.
    if not check_target_unique(norm_target, norm_text, num_target_spans):
        if '@' in norm_target:
            norm_target = norm_target.replace('@', '')
        else:
            norm_target = '${}$'.format(norm_target)
        norm_text = add_target_to_text(norm_target)
        if not check_target_unique(norm_target, norm_text, num_target_spans):
            if '$' not in norm_target:
                norm_target = '${}$'.format(norm_target)
                norm_text = add_target_to_text(norm_target)
                if check_target_unique(norm_target,
                                       norm_text, num_target_spans):
                    return norm_target
            raise Exception('Normalised word {} occurs in the text more times '\
                            'than it spans {}. Text {}'\
                            .format(norm_target, num_target_spans, norm_text))
    return norm_target

def target_normalisation(target_dict, renormalise=False):
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


    target = normalise_target(target, org_text, sorted_spans, renormalise)
    for start_index, end_index in sorted_spans:
        start_text = org_text[: start_index]
        end_text = org_text[end_index :]
        start_text += ' {} '.format(target)
        org_text = start_text + end_text
    org_text = ' '.join(org_text.split())

    return org_text, target

def normalise_context(target_dicts, lower, renormalise=False):
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
        norm_text, norm_target = target_normalisation(target_dict, renormalise)
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
        for attempts in range(1, 3):
            contexts = []
            norm_target = norm_targets[index]
            # This only happens if the first normalization does not work
            if attempts == 2:
                text, norm_target = normalise_context([target_dicts[index]],
                                                      lower=lower,
                                                      renormalise=True)
                dependency_tokens = parser(text)[0]
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
                if attempts == 1:
                    continue
                raise ValueError('The number of identified targets `{}` not equal '\
                                 'to the number of targets in the data `{}`'\
                                 .format(contexts, rel_target))
            # Ensure the returned data type is consistent
            if contexts == []:
                raise ValueError('This should not happen as each data type should '\
                                 'have a target {}'.format(rel_target))
            all_contexts.append(contexts)
    return all_contexts




def text_and_span(text, norm_target, target, norm_target_span):

    def remove_duplicate_spans(sorted_spans):
        de_dup_spans = []
        for start_0, end_0 in sorted_spans:
            for start_1, end_1 in sorted_spans:
                if start_0 == start_1 and end_0 == end_1:
                    continue
                # Do not allow spans through that are within other spans
                elif start_0 > start_1 and end_0 < end_1:
                    break
            else:
                de_dup_spans.append((start_0, end_0))
        return de_dup_spans

    all_norm_match_spans = []
    norm_target = re.escape(norm_target)
    escaped_target = re.escape(target)
    real_target_span = None
    for norm_target_match in re.finditer(norm_target, text):
        match_span = norm_target_match.span()
        all_norm_match_spans.append(match_span)
        if match_span == norm_target_span:
            real_target_span = norm_target_span
    if real_target_span is None:
        raise Exception('Cannot find the normalised Target {} at span {} in '\
                        'the following text {}'\
                        .format(norm_target, norm_target_span, text))
    # Get all the unique spans of the normalised target and the actually target
    target_match_spans = [target_match.span() \
                          for target_match in re.finditer(escaped_target, text)]
    all_spans = sorted(set(all_norm_match_spans + target_match_spans),
                       key=lambda span: span[0])
    all_spans = remove_duplicate_spans(all_spans)
    index_real_target = [index for index, span in enumerate(all_spans) \
                         if span == real_target_span]
    num_real_target_spans = len(index_real_target)
    if num_real_target_spans != 1:
        raise Exception('There can only be one Target span and {} were found '\
                        'for target {} normalised {} in text {} with spans {}'\
                        .found(target, norm_target, text, norm_target_span))
    index_real_target = index_real_target[0]
    # Convert all of the normalised targets to targets
    original_text = re.sub(norm_target, target, text)
    if len(re.findall(escaped_target, original_text)) != len(all_spans):
        raise Exception('The number of targets converted is more or less than '\
                        'the number found in the normailsed text {}. Converted '\
                        'text {}. Target {}, Normalised target {}, all spans {}'\
                        .format(text, original_text, target, norm_target, all_spans))
    target_match_spans = [target_match.span() \
                          for target_match in re.finditer(escaped_target, original_text)]
    target_match_spans = sorted(target_match_spans, key=lambda span: span[0])
    real_target_span = target_match_spans[index_real_target]
    return original_text, real_target_span

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
    targets = [target_dict['target'] for target_dict in target_dicts]
    norm_texts, norm_targets = normalise_context(target_dicts, lower)
    # Get contexts
    all_dependency_tokens = parser(norm_texts)
    all_contexts = []

    for index, dependency_tokens in enumerate(all_dependency_tokens):
        for attempts in range(1, 3):
            contexts = []
            norm_target = norm_targets[index]
            # This only happens if the first normalization does not work
            if attempts == 2:
                text, norm_target = normalise_context([target_dicts[index]],
                                                      lower=lower,
                                                      renormalise=True)
                norm_target = norm_target[0]
                dependency_tokens = parser(text)[0]

            for dependency_token in dependency_tokens:
                current_target = targets[index]
                if lower:
                    current_target = current_target.lower()
                if dependency_token.token == norm_target:
                    connected_text, target_span = dependency_token\
                                                  .connected_target_span()
                    text_span = text_and_span(connected_text, norm_target,
                                              current_target, target_span)
                    contexts.append({'text' : text_span[0],
                                     'span' : text_span[1]})
            rel_target = target_dicts[index]
            valid_num_targets = len(rel_target['spans'])
            if valid_num_targets != len(contexts):
                if attempts == 1:
                    continue
                raise ValueError('The number of identified targets `{}` not equal '\
                                 'to the number of targets in the data `{}`'\
                                 ' norm target {}'\
                                 .format(contexts, rel_target, norm_target))
            # Ensure the returned data type is consistent
            if contexts == []:
                raise ValueError('This should not happen as each data type should '\
                                 'have a target {}'.format(rel_target))
            all_contexts.append(contexts)
            break
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
