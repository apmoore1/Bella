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

    Methods:

    1. get_n_relations -- The lower and upper bound of the range `n` for the \
    words retrived at different relation levels associated to the `n` range.
    '''

    def __init__(self, token, relations):
        '''
        :param token: The string that the dependency relations are associated to.
        :param relations: dict of dependency relations
        :type token: String
        :type relations: dict
        :returns: Constructor
        :rtype: None
        '''

        if not isinstance(token, str):
            raise TypeError('token parameter has to be of type str not {}'\
                            .format(type(token)))
        if not isinstance(relations, dict):
            raise TypeError('relations parameter has to be of type dict not {}'\
                            .format(type(relations)))
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

    def get_n_relations(self, relation_range=(1, 1)):
        '''
        Given a tuple to denote the range of the depth of dependency relations to
        return. Returns a list of words.

        If the relation range is greater than the depth range then it will return
        all words up to the last depth and ignore the depths that do not exisit

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
        '''

        if not isinstance(relation_range, tuple):
            raise TypeError('The relation_range parameter has to be of type tuple'\
                            ' and not {}'.format(type(relation_range)))
        if len(relation_range) != 2:
            raise ValueError('The relation_range tuple has to be of length 2 '\
                             'and not {}'.format(len(relation_range)))
        if not (isinstance(relation_range[0], int)
                and isinstance(relation_range[1], int)):
            raise ValueError('The relation_range tuple can only contain value of '\
                             'type int values {}'.format(relation_range))
        if relation_range[0] < 1 or relation_range[1] < 1:
            raise ValueError('The values in the relation range have to be greater'\
                             ' than one')
        if not relation_range[0] <= relation_range[1]:
            raise ValueError('The second value in relation_range has to be greater'\
                             ' than or equall to the first value. Values {}'\
                             .format(relation_range))
        all_related_words = []
        for i in range(relation_range[0], relation_range[1] + 1):
            related_words = self.relations.get(i, [])
            if related_words == []:
                break
            all_related_words.extend(related_words)
        return all_related_words
