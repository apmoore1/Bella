'''
Contains the following class:

1. DependencyToken
'''


class DependencyToken():
    '''
    Objects that contain a token and its dependency relations via it's attributes.

    Attributes:

    1. self.token -- The token stored e.g. `hello`
    2. self.relations -- The dict that contains words associated to the token \
    at different dependency relation levels e.g. relation of 1 direct dependency \
    of the token, relation 2 dependency relations of the relation 1 words.
    3. self.connected_words -- A list of tuples (word, relation) where all words \
    are within the same dependency tree and are listed in the order they appear \
    in the text that was parsed. relation can take two values `CURRENT` current \
    token and `CONNECTED` any other token.

    Methods:

    1. get_n_relations -- The lower and upper bound of the range `n` for the \
    words retrived at different relation levels associated to the `n` range.
    '''

    def __init__(self, token, relations, connected_words):
        '''
        :param token: The string that the dependency relations are associated to.
        :param relations: dict of dependency relations
        :param connected_words: list of current token and all of its syntactically\
         connected words where the list is made up of tuples where the first value \
         is the String of the connected word and the second describes its realtion \
         either `CONNECTED` for connected words or `CURRENT` stating it is the \
         current token. All words are ordered by their occurence in the text. \
         The connected words are all words in the same dependecy tree.
        :type token: String
        :type relations: dict
        :type connected_words: list
        :returns: Constructor
        :rtype: None
        '''

        if not isinstance(token, str):
            raise TypeError('token parameter has to be of type str not {}'\
                            .format(type(token)))
        if not isinstance(relations, dict):
            raise TypeError('relations parameter has to be of type dict not {}'\
                            .format(type(relations)))
        if not isinstance(connected_words, list):
            raise TypeError('connected words parameter has to be of type list not'\
                            ' {}'.format(connected_words))
        if len(connected_words) < 1:
            raise ValueError('connected words has to always contain one word which'\
                             ' is the current word')

        current_count = 0
        valid_relations = set(['CONNECTED', 'CURRENT'])
        for connected_word in connected_words:
            if not isinstance(connected_word, tuple):
                raise TypeError('connected_words parameter has to be a list of '\
                                'tuples not {}'.format(type(connected_word)))
            word, relation = connected_word
            if relation not in valid_relations:
                raise ValueError('connected relations can only be {} and not {}'\
                                 .format(valid_relations, relation))
            if not isinstance(word, str):
                raise TypeError('The word in the connected_word has to be of '\
                                'type String and not {}'.format(type(word)))
            if relation == 'CURRENT':
                current_count += 1
                if word != token:
                    raise ValueError('The token that is the CURRENT token {} in '\
                                     'connected_words should also be equal to '\
                                     'the token value {}'.format(word, token))
        if current_count != 1:
            raise ValueError('There has to be ONLY one CURRENT relation in '\
                             'the connected words {}'.format(connected_words))

        all_depths = []
        for depth_index, related_list in relations.items():
            if not isinstance(depth_index, int):
                raise TypeError('The keys in the relations dict have to be of '\
                                'type int not {}'.format(type(depth_index)))
            if not isinstance(related_list, list):
                raise TypeError('The related words in the relations dict have to '\
                                'be of type list and not {}'.format(type(related_list)))
            all_depths.append(depth_index)
        if relations != {}:
            actual_depth_range = sorted(all_depths)
            start_depth = actual_depth_range[0]
            if start_depth != 1:
                raise ValueError('The depth of the relations should always start at '\
                                 '1 and not less or greater than your start depth {}'\
                                 .format(start_depth))
            max_depth = max(all_depths)
            valid_depth_range = list(range(1, max_depth + 1))
            if valid_depth_range != actual_depth_range:
                raise ValueError('The depths (keys) in the relations dict has to be '\
                                 'incremental e.g. 1,2,3,4 and not 1,3,4 the depths '\
                                 'in your relations {} valid depths {}'\
                                 .format(actual_depth_range, valid_depth_range))
        self.token = token
        self.relations = relations
        self.connected_words = connected_words

    def get_n_relations(self, relation_range=(1, 1)):
        '''
        Given a tuple to denote the range of the depth of dependency relations to
        return. Returns a list of words.

        If the relation range is greater than the depth range then it will return
        all words up to the last depth and ignore the depths that do not exisit.

        Negative ranges can be used in a similar manner to how list indexs work.

        :param relation_range: Tuple to denote the range of the depth of \
        dependency relations to return.
        :type relation_range: tuple
        :returns: A list of words that are associated to the token at the depth \
        range given.
        :rtype: list

        :Example:
        >>> rel_token = DependencyToken('example', {1 : ['hello'], 2 : ['anything'],
                                        3 : ['lastly']})
        >>> relations = rel_token.get_n_relations((2,3))
        >>> relations == ['anything', 'lastly']
        >>> relations = rel_token.get_n_relations((-2,-1))
        >>> relations == ['anything', 'lastly']
        '''

        def negative_range(range_1, range_2):
            '''
            Given two integers where one or both are negative returns the relations
            at those range depths.

            :param range_1: start of the range
            :param range_2: end of the range
            :type range_1: int
            :type range_2: int
            :returns: Given two integers where one or both are negative returns \
            the relations at those range depths.
            :rtype: list
            '''

            def flat_list(word_lists):
                '''
                :param word_lists: A list which contains lists of Strings
                :type word_lists: list
                :returns: Converts a nested list of lists into a flat list.
                :rtype: list
                '''

                return [word for words in word_lists for word in words]

            i = 1
            all_words = []
            while True:
                related_words = self.relations.get(i, [])
                if related_words == []:
                    break
                i += 1
                all_words.append(related_words)
            if range_2 == range_1:
                if range_2 == -1:
                    return flat_list(all_words[range_1 :])
                return flat_list(all_words[range_1 : range_2 + 1])
            if range_1 > 0:
                range_1 -= 1
            if range_2 > 0:
                range_2 -= 1
            if range_2 == -1:
                return flat_list(all_words[range_1 :])
            elif range_2 < -1:
                return flat_list(all_words[range_1 : range_2 + 1])
            return flat_list(all_words[range_1 : range_2])

        if not isinstance(relation_range, tuple):
            raise TypeError('The relation_range parameter has to be of type tuple'\
                            ' and not {}'.format(type(relation_range)))
        if len(relation_range) != 2:
            raise ValueError('The relation_range tuple has to be of length 2 '\
                             'and not {}'.format(len(relation_range)))
        range_1 = relation_range[0]
        range_2 = relation_range[1]
        if not (isinstance(range_1, int)
                and isinstance(range_2, int)):
            raise ValueError('The relation_range tuple can only contain value of '\
                             'type int values {}'.format(relation_range))
        if range_1 == 0 or range_2 == 0:
            raise ValueError('relation_range values cannot be zero')
        all_related_words = []
        # Check if it is negative indexing
        if range_1 < 0:
            if range_2 < range_1:
                raise ValueError('If the first value in the range is negative it'\
                                 ' has to be less than the second value in the '\
                                 'range 1: {} 2: {}'.format(range_1, range_2))
            all_related_words = negative_range(range_1, range_2)
        elif range_2 < 0:
            all_related_words = negative_range(range_1, range_2)
        else:

            if range_1 < 1 or range_2 < 1:
                raise ValueError('The values in the relation range have to '\
                                 'be greater than one')
            if range_1 > range_2:
                raise ValueError('The second value in relation_range has to '\
                                 'be greater than or equall to the first value.'\
                                 ' Values {}'.format(relation_range))
            for i in range(range_1, range_2 + 1):
                related_words = self.relations.get(i, [])
                if related_words == []:
                    break
                all_related_words.extend(related_words)
        return all_related_words

    def connected_target_span(self, renormalise=None):
        '''
        :param renormalise: A tuple containing the normalised target and the \
        real target word e.g. (Samsung_S5, Samsung S5). If you do not require \
        this leave as default parameter None.
        :type renormalise: tuple. Default None
        :returns: It returns connected words as a String and the span of the current \
        token within that String as a tuple of length 2.
        :rtype: tuple
        '''

        connected_words = []
        word_length = {}
        target_index = -1
        for index, word_relation in enumerate(self.connected_words):
            word, relation = word_relation
            if renormalise is not None:
                normalised_target, real_word = renormalise
                if word == normalised_target:
                    word = real_word
            connected_words.append(word)
            if relation == 'CURRENT':
                target_index = index
            word_length[index] = len(word) + word_length.get(index - 1, 0)
        if target_index == -1:
            raise ValueError('target index can never be -1 error in the '\
                             'connected_words {}'.format(connected_words))
        connected_text = ' '.join(connected_words).strip()
        num_previous_chars = word_length.get(target_index - 1, 0)
        if target_index != 0:
            num_previous_chars += target_index
        if renormalise is not None:
            self.token = renormalise[1]
        target_word = self.token
        target_span = (num_previous_chars, num_previous_chars + len(target_word))
        if connected_text[target_span[0]: target_span[1]] != target_word:
            raise ValueError('Cannot get the target word `{}` within the connected '\
                             'text `{}` this could be due to the connected words '\
                             '`{}` spans: `{}`'.format(target_word, connected_text,
                                                       connected_words, target_span))
        return connected_text, target_span
