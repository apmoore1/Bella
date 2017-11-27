'''
Contains functions that perform depdency parsing.
'''
from collections import defaultdict
import os
import subprocess
import tempfile

import networkx as nx
from networkx.algorithms import traversal

from tdparse.helper import read_config, full_path
from tdparse.dependency_tokens import DependencyToken

def tweebo_install(tweebo_func):
    '''
    Python decorator that ensures that
    `TweeboParser <https://github.com/ikekonglp/TweeboParser>`_ is installed,
    before running the function it wraps. Returns the given function.

    :param tweebo_func: A function that uses the Tweebo Parser.
    :type tweebo_func: function
    :returns: The given function
    :rtype: function
    '''

    tweebo_dir = full_path(read_config('depdency_parsers')['tweebo_dir'])
    # If the models file exists then Tweebo has been installed or failed to
    # install
    tweebo_models = os.path.join(tweebo_dir, 'pretrained_models.tar.gz')
    if not os.path.isfile(tweebo_models):
        install_script = os.path.join(tweebo_dir, 'install.sh')
        subprocess.run(['bash', install_script])
    return tweebo_func

def get_tweebo_dependencies(token_dep_sentence):
    '''
    NOTE:
    The DependencyToken allows easy access to all the dependency links for that
    token.

    :param token_dep_sentence: list of tuples that contain (token, linked \
    index token)
    :type token_dep_sentence: list
    :returns: A list of DependencyToken instances one for each tuple/token.
    :rtype: list
    '''

    def dep_search(dep_index, sentence, dep_info):
        '''
        This is a tail recusive function that returns a dictionary denoting
        which index in the sentence relates to the other tokens in the sentence
        and at what dependency depth.

        :param dep_index: The index of the token whos dependecies are being \
        collected
        :param sentence: List of tuples which contain (token, index of head word)
        :param dep_info: default dict whose keys are indexs from the sentence \
        and value is a default dict whose keys are dependency depths and value \
        is the sentence index related to that dependency depth.
        :type dep_index: int
        :type sentence: list
        :type dep_info: defaultdict
        :returns: Default dictionary whose keys are sentence indexs, values are \
        default dictionaries whose keys are dependency depths and value is the \
        associated sentence index to that depth.
        :rtype: defaultdict

        :Example:
        >>> sentence = [('To', -1), ('appear', 0), ('(', -2),
                        ('EMNLP', 1), ('2014', 3)]
        >>> dep_info = defaultdict(lambda: dict())
        >>> print(dep_search(4, sentence, dep_info))
        >>> {0 : {1 : 1, 2 : 3, 3 : 4},
             1 : {1 : 3, 2 : 3},
             3 : {1 : 4},
             4 : {}}
        '''

        head_index = sentence[dep_index][1]
        if head_index == -1 or head_index == -2:
            return dep_info
        prev_dep_info = dep_info[dep_index]
        head_deps = {dep_level + 1: deps
                     for dep_level, deps in prev_dep_info.items()}
        head_deps[1] = dep_index
        dep_info[head_index] = head_deps
        return dep_search(head_index, sentence, dep_info)

    dep_results = defaultdict(lambda: defaultdict(set))
    for index, _ in enumerate(token_dep_sentence):
        dep_result = dep_search(index, token_dep_sentence,
                                defaultdict(lambda: dict()))
        for token_index, dependencies in dep_result.items():
            for dep_level, dependent in dependencies.items():
                dep_results[token_index][dep_level].add(dependent)
    G = nx.Graph()
    for index, token_related_index in enumerate(token_dep_sentence):
        G.add_node(index)
        token, related_index = token_related_index
        related_index = related_index
        if related_index not in {-2, -1}:
            G.add_edge(index, related_index)

    # Convert each of the tokens in the sentence into a dependent token
    # using the results from searching through the dependencies
    dep_tokens = []
    for token_index, token_dep in enumerate(token_dep_sentence):
        token, _ = token_dep
        depth_related = dep_results[token_index]
        token_relations = defaultdict(list)

        connected_indexs = set()
        for _, node_relations in nx.bfs_successors(G, token_index):
            for node_relation in node_relations:
                connected_indexs.add(node_relation)
        if token_index in connected_indexs:
            connected_indexs.remove(token_index)
        connected_words = [token_dep_sentence[con_index][0]
                           for con_index in connected_indexs]
        # Get the tokens relations
        for depth, related_tokens_index in depth_related.items():
            for related_token_index in related_tokens_index:
                related_token = token_dep_sentence[related_token_index][0]
                token_relations[depth].append(related_token)
        dep_tokens.append(DependencyToken(token, token_relations, connected_words))
    return dep_tokens


def tweebo_post_process(processed_text):
    '''
    Given the text processed by Tweebo as a String that has a token and its
    meta data on each new line and a sentence represented by a newline. It
    finds all of the tokens related to a single sentence and returns all
    sentences as a list of DependencyToken instances produced by the
    :py:func:`get_tweebo_dependencies`.

    :param processed_text: The string ouput of Tweebo parser
    :type processed_text: String
    :returns: A list of a list of DependencyToken instances, where each list \
    represents a String instance in the texts given to the :py:func:`tweebo` \
    function.
    :rtype: list
    '''

    tokens = processed_text.split('\n')
    sentences = []
    last_token = None
    sentence = []
    for token in tokens:
        token = token.strip()
        if last_token == '' and token == '':
            continue
        elif token == '':
            if sentence == ['$$$EMPTY$$$']:
                sentences.append([])
            else:
                sentences.append(get_tweebo_dependencies(sentence))
            sentence = []
        else:
            token = token.split('\t')
            token_dep_index = int(token[6]) - 1
            token_text = token[1]
            if token_text == '$$$EMPTY$$$':
                sentence.append('$$$EMPTY$$$')
            else:
                sentence.append((token_text, token_dep_index))
        last_token = token
    return sentences

@tweebo_install
def tweebo(texts):
    '''
    Given a list of Strings will tokenise, pos tag and then dependecy parse
    the text using `Tweebo <https://github.com/ikekonglp/TweeboParser>`_
    a Tweet specific parser.

    The Tweebo parser cannot handle no strings therefore a special empty string
    symbol is required.

    If one of the texts is an empty String then an empty list will be returned
    for that index of the returned list.

    :param texts: The texts that are to be parsed
    :type text: list
    :returns: A list of of a list of DependencyToken instances. A list per text \
    in the texts argument.
    :rtype: list
    '''

    def no_text(text):
        '''
        Given a String checks if it is empty if so returns an empty_token else
        the text that was given.

        :param text: Text to be checked
        :type text: String
        :returns: The text if it is not empty or empty token if it is.
        :rtype: String
        '''

        empty_token = '$$$EMPTY$$$'
        if text.strip() == '':
            return empty_token
        return text


    with tempfile.TemporaryDirectory() as temp_dir:
        text_file_path = os.path.join(temp_dir, 'text_file.txt')
        result_file_path = os.path.join(temp_dir, 'text_file.txt.predict')
        tweebo_dir = full_path(read_config('depdency_parsers')['tweebo_dir'])
        with open(text_file_path, 'w+') as text_file:
            for text in texts:
                text = no_text(text)
                text_file.write(text)
                text_file.write('\n')
        run_script = os.path.join(tweebo_dir, 'run.sh')
        if subprocess.run(['bash', run_script, text_file_path]):
            with open(result_file_path, 'r') as result_file:
                return tweebo_post_process(result_file.read())
        else:
            raise SystemError('Could not run the Tweebo run script {}'\
                              .format(run_script))
