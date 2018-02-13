import json

from nltk.tree import Tree
from stanfordcorenlp import StanfordCoreNLP

STANFORD_NLP = StanfordCoreNLP('http://localhost', port=9000)


def tokenise(text):
    output_dict = STANFORD_NLP.annotate(text, {'annotators' : 'ssplit,tokenize',
                                               'tokenize.language' : 'English',
                                               'outputFormat' : 'json'})
    output_dict = json.loads(output_dict)
    tokens = [token['originalText'] for s in output_dict['sentences'] \
              for token in s['tokens']]
    return tokens

def constituency_parse(text):
    '''
    :param text: The text you want to parse
    :type text: String
    :returns: A list of parse trees where each tree is represented by \
    nltk.tree.Tree. Each parse tree is associated to a sentence in the text. \
    If one sentence in the text then the list will be of length 1.
    :rtype: list
    '''
    if text.strip() == '':
        raise ValueError('There has to be some text to parse. Text given {}'\
                         .format(text))
    output_dict = STANFORD_NLP.annotate(text, {'annotators' : 'pos,parse',
                                               'tokenize.language' : 'English',
                                               'outputFormat' : 'json'})
    output_dict = json.loads(output_dict)
    parse_trees = [Tree.fromstring(sent['parse']) \
                   for sent in output_dict['sentences']]
    return parse_trees
