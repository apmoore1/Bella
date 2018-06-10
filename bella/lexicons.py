'''
This module contains the Lexicon classes where each class is a sub class of
the :py:class:`bella.lexicons.Lexicon` class.

The following classes are in the module:

1. Lexicon -- Base class
2. HuLiu -- `Hu and Liu <https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf>`_
lexicon.
3. Mpqa -- `Wilson, Wiebe and Hoffman \
<https://aclanthology.coli.uni-saarland.de/papers/H05-1044/h05-1044>`_ lexicon.
4. NRC -- `Mohammad and Turney \
<http://saifmohammad.com/WebDocs/Mohammad-Turney-NAACL10-EmotionWorkshop.pdf>`_
lexicon.
'''
from collections import defaultdict
import csv
import re
from pathlib import Path


class Lexicon():
    '''
    Base class for all lexicons.

    Attributes:

    1. self.lexicon -- List of tuples where each tuple contains (word, category) \
    e.g. [('great', 'positive')]
    3. self.words -- Set of Strings which are the words in the self.lexicon list.
    2. self.name -- String name of the class.

    Methods to be overidden:

    1. self.get_lexicon -- Must return List of tuples where each tuple contains \
    (word, category). This is not used to create self.lexicon if the lexicon \
    is defined in the constructor.

    Static Methods:

    1. combine_lexicons -- Combines two Lexicon classes together and removes \
    (word, category) tuples from self.lexicon that have different category values \
    for the same word.
    '''

    def __init__(self, subset_cats=None, lower=False, name=None, lexicon=None):
        if subset_cats is not None:
            if not isinstance(subset_cats, set):
                raise TypeError('subset_cats parameter has to be of type set '\
                                'and not {}'.format(type(subset_cats)))
        if lower != False:
            if not isinstance(lower, bool):
                raise TypeError('lower parameter has to be of type bool not {}'\
                                .format(type(lower)))
        self.lexicon = lexicon
        if self.lexicon is None:
            self.lexicon = self.get_lexicon()

        if not isinstance(self.lexicon, list):
            raise TypeError('self.lexicon has to be of type list not {} This could'\
                            ' be due to the `get_lexicon` method'\
                             .format(type(self.lexicon)))
        if not isinstance(self.lexicon[0], tuple):
            raise TypeError('self.lexicon should contain tuples not {} This could'\
                            ' be due to the `get_lexicon` method'\
                            .format(type(self.lexicon[0])))

        self.lexicon = self._process_lexicon(subset_cats, lower)
        self.remove_duplicates()
        self.words = set([word for word, cat in self.lexicon])
        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__

    def remove_duplicates(self):
        duplicate_lexicon = defaultdict(list)
        for word, cat in self.lexicon:
            duplicate_lexicon[word].append(cat)
        lexicon = []
        for word, cat_list in duplicate_lexicon.items():
            if len(cat_list) > 1:
                continue
            lexicon.append((word, cat_list[0]))
        self.lexicon = lexicon

    def get_lexicon(self):
        '''
        Method to be overidden. It should be the method that all sub classes
        use to parse in their lexicon into a list of tuples format where each
        tuple contain (word, category).

        :returns: A list of tuples containing (word, category) e.g. \
        ('great', 'positive')
        :rtype: list
        '''

        return [('example', 'value')]

    def _process_lexicon(self, subset_cats, lower):
        '''
        Method not to be overidden. It process the lexicon based on the parameters.
        Returns a list of tuples where a tuple contains (word, category).

        :param subset_cats: Categories of words that you want to return e.g. \
        `positive` for `positive` words only. If None then no words are subsetted.
        :param lower: Whether to lower case the words in the lexicon.
        :type subset_values: set
        :type lower: bool
        :returns: Returns the lexicon as a list of tuples (word, category).
        :rtype: list
        '''

        temp_lexicon = self.lexicon
        lexicon = set()
        for word, cat in temp_lexicon:
            if subset_cats is not None:
                if cat not in subset_cats:
                    continue
            if lower:
                word = word.lower()
            lexicon.add((word, cat))
        return list(lexicon)

    @staticmethod
    def combine_lexicons(lexicon1, lexicon2):
        '''
        Combines two lexicons and removes words that have different categories
        for the words in the lexicons. This can be used to combine sentiment
        lexicons and remove words that have opposite sentiments. The self.name
        of the Lexicon is the combination the names of lexicon1 and lexicon2.

        NOTE: Requires that both lexicons have the same categories.

        :param lexicon1: An instance of :py:class:`bella.lexicons.Lexicon`
        :param lexicon2: An instance of :py:class:`bella.lexicons.Lexicon`
        :type lexicon1: :py:class:`bella.lexicons.Lexicon`
        :type lexicon2: :py:class:`bella.lexicons.Lexicon`
        :returns: Combines the two lexicons and removes words that have opposite \
        sentiment and returns it as a :py:class:`bella.lexicons.Lexicon`
        :rtype: :py:class:`bella.lexicons.Lexicon`
        '''
        def compare_lexicons(lex1, lex2, lex_name):
            '''
            Given the two lexicons will combine the lexicons together and remove
            words that do not have the same category.

            :param lex1: An instance of :py:class:`bella.lexicons.Lexicon`
            :param lex2: An instance of :py:class:`bella.lexicons.Lexicon`
            :param lex_name: The name to give the returned Lexicon
            :type lex1: :py:class:`bella.lexicons.Lexicon`
            :type lex2: :py:class:`bella.lexicons.Lexicon`
            :type lex_name: String
            :returns: Combines the two lexicons and removes words that have opposite \
            sentiment and returns it as a :py:class:`bella.lexicons.Lexicon`
            :rtype: :py:class:`bella.lexicons.Lexicon`
            '''

            combined_words = set(list(lex1.keys()) + list(lex2.keys()))
            combined_lexicons = []
            for word in combined_words:

                if word in lex1 and word in lex2:
                    if lex1[word] != lex2[word]:
                        continue
                    combined_lexicons.append((word, lex1[word]))
                elif word in lex1:
                    combined_lexicons.append((word, lex1[word]))
                elif word in lex2:
                    combined_lexicons.append((word, lex2[word]))
                else:
                    raise KeyError('The word {} has to be in one of the lexicons'\
                                   .format(word))
            return Lexicon(lexicon=combined_lexicons, name=lex_name)

        if not isinstance(lexicon1, Lexicon) or not isinstance(lexicon2, Lexicon):
            raise TypeError('Both parameters require to be type Lexicon not {}, {}'\
                            .format(type(lexicon1), type(lexicon2)))
        combined_lex_name = "{} {}".format(lexicon1.name, lexicon2.name)
        lexicon1 = lexicon1.lexicon
        lexicon2 = lexicon2.lexicon

        word_cat1 = {word : cat for word, cat in lexicon1}
        word_cat2 = {word : cat for word, cat in lexicon2}

        cat1 = set(word_cat1.values())
        cat2 = set(word_cat2.values())
        if cat1 != cat2:
            raise ValueError('These two lexicons cannot be combined as they have '\
                             'different and non-comparable categories: '\
                             'categories1: {} categories2: {}'\
                             .format(cat1, cat2))
        return compare_lexicons(word_cat1, word_cat2, combined_lex_name)

    def __str__(self):
        return self.name


class Mpqa(Lexicon):
    '''
    MPQA lexicon `Wilson, Wiebe and Hoffman \
    <https://aclanthology.coli.uni-saarland.de/papers/H05-1044/h05-1044>`_.
    Sub class of :py:class:`bella.lexicons.Lexicon`

    Category lables = 1. positive, 2. negative, 3. both, 4. neutral
    '''

    def __init__(self, mpqa_fp: Path, subset_cats=None, lower=False,
                 name=None, lexicon=None):
        '''
        :param mpqa_fp: File path to the mpqa lexicon.
        '''
        self.file_path = mpqa_fp
        super().__init__(subset_cats=subset_cats, lower=lower,
                         name=name, lexicon=lexicon)

    def get_lexicon(self):
        '''
        Overrides :py:func@`bella.lexicons.Lexicon.get_lexicon`
        '''

        word_cats = []
        with self.file_path.open('r') as mpqa_file:
            for line in mpqa_file:
                line = line.strip()
                if line:
                    key_values = {}
                    for data in line.split():
                        if '=' in data:
                            key, value = data.split('=')
                            key_values[key] = value
                    word = key_values['word1']
                    cat = key_values['priorpolarity']
                    if cat == 'weakneg':
                        cat = key_values['polarity']
                    word_cats.append((word, cat))
        return word_cats


class HuLiu(Lexicon):
    '''
    `Hu and Liu
    <https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf>`_
    Lexicon.
    Sub class of :py:class:`bella.lexicons.Lexicon`

    Category lables = 1. positive, 2. negative
    '''

    def __init__(self, huliu_folder_path: Path, subset_cats=None, lower=False,
                 name=None, lexicon=None):
        '''
        :param nrc_fp: File path to the Hu Liu sentiment lexicon folder that \
        contains both the positive and negative words txt files.
        '''
        self.folder_path = huliu_folder_path
        super().__init__(subset_cats=subset_cats, lower=lower,
                         name=name, lexicon=lexicon)

    def get_lexicon(self):
        '''
        Overrides :py:func@`bella.lexicons.Lexicon.get_lexicon`
        '''

        cats = ['positive', 'negative']
        word_cat = []
        for cat in cats:
            file_path = self.folder_path.joinpath('{}-words.txt'.format(cat))
            with file_path.open('r', encoding='cp1252') as senti_file:
                for line in senti_file:
                    if re.search('^;', line) or re.search(r'^\W+', line):
                        continue
                    line = line.strip()
                    word_cat.append((line.strip(), cat))
        return word_cat


class NRC(Lexicon):
    '''
    NRC emotion `Mohammad and Turney \
    <http://saifmohammad.com/WebDocs/Mohammad-Turney-NAACL10-EmotionWorkshop.pdf>`_\
    Lexicon
    Sub class of :py:class:`bella.lexicons.Lexicon`

    Category lables = 1. anger, 2. fear, 3. anticipation, 4. trust, 5. surprise,
    6. sadness, 7. joy, 8. disgust, 9. positive, and 10. negative.
    '''

    def __init__(self, nrc_fp: Path, subset_cats=None, lower=False,
                 name=None, lexicon=None):
        '''
        :param nrc_fp: File path to the NRC lexicon.
        '''
        self.file_path = nrc_fp
        super().__init__(subset_cats=subset_cats, lower=lower,
                         name=name, lexicon=lexicon)

    def get_lexicon(self):
        '''
        Overrides :py:func:`bella.lexicons.Lexicon.get_lexicon`
        '''

        word_cat = []
        with self.file_path.open('r', newline='') as emotion_file:
            tsv_reader = csv.reader(emotion_file, delimiter='\t')
            for row in tsv_reader:
                if len(row):
                    word = row[0]
                    cat = row[1]
                    association = int(row[2])
                    if association:
                        word_cat.append((word, cat))
        return word_cat
