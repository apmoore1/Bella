'''
Contains functions that perform depdency parsing.
'''
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import networkx as nx
from ruamel.yaml import YAML
from tweebo_parser import API

from bella.dependency_tokens import DependencyToken
from bella import stanford_tools

BELLA_CONFIG_FP = Path.home().joinpath('.Bella', 'config.yaml')

class TweeboParser(object):
    '''
    Singleton Class instance
    '''

    instance = None

    @staticmethod
    def get_config() -> Tuple[str, int]:
        hostname = '0.0.0.0'
        port = 8000
        yaml = YAML()
        config_data = {}
        BELLA_CONFIG_FP.parent.mkdir(parents=True, exist_ok=True)
        if BELLA_CONFIG_FP.exists():
            with BELLA_CONFIG_FP.open('r') as config_file:
                config_data = yaml.load(config_file)
                if 'tweebo_parser' in config_data:
                    tweebo_config = config_data['tweebo_parser']
                    if 'hostname' in tweebo_config:
                        hostname = tweebo_config['hostname']
                    if 'port' in tweebo_config:
                        port = tweebo_config['port']

        config_data['tweebo_parser'] = {}
        config_data['tweebo_parser']['hostname'] = hostname
        config_data['tweebo_parser']['port'] = port
        with BELLA_CONFIG_FP.open('w') as config_file:
            yaml.dump(config_data, config_file)
        return hostname, port

    def __new__(cls):
        if TweeboParser.instance is None:
            hostname, port = cls.get_config()
            TweeboParser.instance = API(hostname=hostname, port=port, 
                                        log_errors=True)
        return TweeboParser.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name, value):
        return setattr(self.instance, name, value)


def _to_dependencies_tokens(token_dep_sentence: List[Tuple[str, int]]
                            ) -> List[DependencyToken]:
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
        :param sentence: List of tuples which contain (token, index of \
        head word)
        :param dep_info: default dict whose keys are indexs from the sentence \
        and value is a default dict whose keys are dependency depths and \
        value is the sentence index related to that dependency depth.
        :type dep_index: int
        :type sentence: list
        :type dep_info: defaultdict
        :returns: Default dictionary whose keys are sentence indexs, values \
        are default dictionaries whose keys are dependency depths and value \
        is the associated sentence index to that depth.
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
        if token_index not in connected_indexs:
            connected_indexs.add(token_index)
        connected_indexs = sorted(list(connected_indexs))
        connected_words = []
        for connected_index in connected_indexs:
            connected_word = token_dep_sentence[connected_index][0].strip()
            if connected_index == token_index:
                connected_words.append((connected_word, 'CURRENT'))
            else:
                connected_words.append((connected_word, 'CONNECTED'))
        # Get the tokens relations
        for depth, related_tokens_index in depth_related.items():
            for related_token_index in related_tokens_index:
                related_token = token_dep_sentence[related_token_index][0]
                token_relations[depth].append(related_token)
        dep_tokens.append(DependencyToken(token, token_relations,
                                          connected_words))
    return dep_tokens


def tweebo(texts: List[str]) -> List[DependencyToken]:
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
    :returns: A list of of a list of DependencyToken instances. A list per \
    text in the texts argument.
    :rtype: list
    '''

    tweebo_api = TweeboParser()
    texts = [text.replace('\n', ' ') for text in texts]
    processed_texts = tweebo_api.parse_conll(texts)

    dep_texts = []
    for processed_text in processed_texts:
        token_dep_indexs = _convert_conll(processed_text)
        dep_texts.append(_to_dependencies_tokens(token_dep_indexs))
    return dep_texts


def _convert_conll(conll_data: str) -> List[Tuple[str, int]]:
    token_dep_indexs = []
    for line in conll_data.split('\n'):
        if not line:
            continue
        column_data = line.split('\t')
        word = column_data[1].strip()
        # All the word indexs start at 1 not 0.
        # Need to take into account the previous sentence words
        dep_token_index = int(column_data[6]) - 1
        token_dep_indexs.append((word, dep_token_index))
    return token_dep_indexs


def stanford(texts: str) -> List[DependencyToken]:

    dep_texts = []
    for text in texts:
        dep_dicts, tokens_dicts = stanford_tools.dependency_parse(text)
        token_dep_indexs = []
        prev_sent_length = 0
        for sentence, _ in enumerate(tokens_dicts):
            tokens_dict = tokens_dicts[sentence]
            dep_dict = dep_dicts[sentence]
            for i in range(1, len(tokens_dict) + 1):
                word = tokens_dict[i]['word']
                # All the word indexs start at 1 not 0.
                # Need to take into account the previous sentence words
                dep_word_index = (dep_dict[i][1] - 1) + prev_sent_length
                # If True then it is the root word
                if dep_word_index + 1 == prev_sent_length:
                    dep_word_index = -1
                token_dep_indexs.append((word, dep_word_index))
            prev_sent_length += len(tokens_dict)
        dep_texts.append(_to_dependencies_tokens(token_dep_indexs))
    return dep_texts
